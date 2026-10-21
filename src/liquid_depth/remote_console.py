"""Authenticated browser gateway to independent instances of the formal console.

Only the gateway is exposed through HTTPS. The existing console handlers keep
their loopback Host, Origin and request-token checks. Each browser owns its own
calibration and state; rendering/inference are serialized across browsers.
"""
import argparse
from dataclasses import dataclass
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from .runtime_paths import resource_root

COOKIE = 'liquid_remote_session'
ASSETS = {'/console', '/console.js', '/console.css', '/api/status'}
MUTATIONS = {'/api/apply', '/api/initialize'}


def origin_of(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('invalid_public_url')
    if parsed.path not in ('', '/') or parsed.query or parsed.fragment:
        raise ValueError('public_url_must_be_an_origin')
    return parsed.scheme + '://' + parsed.netloc


def make_config(path):
    """Create a private access credential; never include it in repository files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('x', encoding='utf-8') as stream:
            json.dump(dict(access_key=secrets.token_urlsafe(32), public_url=''), stream)
        path.chmod(0o600)
    value = json.loads(path.read_text(encoding='utf-8'))
    if len(value.get('access_key', '')) < 32:
        raise ValueError('remote_access_key_too_short')
    return value


@dataclass
class BrowserSession:
    console: object
    server: object
    last_seen: float


class SessionPool:
    def __init__(self, work, blender, checkpoint, *, maximum=3, idle_seconds=7200):
        self.work, self.blender, self.checkpoint = Path(work), blender, checkpoint
        self.maximum, self.idle_seconds = maximum, idle_seconds
        self.sessions = {}
        self.lock = threading.RLock()
        self.computation = threading.Semaphore(1)

    @staticmethod
    def busy(session):
        with session.console.lock:
            return any(job['status'] in ('queued', 'running') for job in session.console.jobs.values())

    def lookup(self, identifier):
        with self.lock:
            session = self.sessions.get(identifier)
            if session:
                session.last_seen = time.monotonic()
            return session

    def create(self):
        from .scene_console_web import Console, serve
        with self.lock:
            now = time.monotonic()
            for identifier, session in list(self.sessions.items()):
                if now - session.last_seen > self.idle_seconds and not self.busy(session):
                    self.close(identifier)
            if len(self.sessions) >= self.maximum:
                raise ValueError('server_sessions_full')
            semaphore = self.computation

            class RemoteConsole(Console):
                def run(self, *args, **kwargs):
                    with semaphore:
                        return super().run(*args, **kwargs)

            identifier = secrets.token_urlsafe(32)
            # Data directory names are independent of the HTTP session credential.
            console = RemoteConsole(self.work / secrets.token_hex(12), self.blender, self.checkpoint)
            ready = threading.Event()
            holder = {}

            def on_ready(server):
                holder['server'] = server
                ready.set()

            threading.Thread(target=serve, args=(console, 0), kwargs={'ready_callback': on_ready},
                             daemon=True, name='private-browser-console').start()
            if not ready.wait(10):
                console.worker.shutdown(wait=False, cancel_futures=True)
                raise RuntimeError('private_console_start_timeout')
            session = BrowserSession(console, holder['server'], now)
            self.sessions[identifier] = session
            console.submit({}, initialize=True)
            return identifier, session

    def close(self, identifier):
        with self.lock:
            session = self.sessions.pop(identifier, None)
        if session:
            threading.Thread(target=session.server.shutdown, daemon=True).start()
            session.console.worker.shutdown(wait=False, cancel_futures=True)

    def close_all(self):
        for identifier in list(self.sessions):
            self.close(identifier)


def remote_markup(body):
    text = body.decode('utf-8')
    replacements = {
        'CPU离线仿真实验台': '服务器仿真实验台',
        'CPU 演示': '服务器计算',
        '本机离线运行，不需要服务器或外部网页。': '仿真与计算由服务器执行，本电脑负责显示与操作。',
        '正在连接本机服务': '正在连接仿真服务器',
        '本机仿真服务暂未响应': '仿真服务器暂未响应',
        '本机仿真服务': '仿真服务器',
    }
    for before, after in replacements.items():
        text = text.replace(before, after)
    return text.encode('utf-8')


def handler_for(pool, config_path, *, allow_insecure_local=False):
    login_page = (resource_root() / 'src/liquid_depth/ui/remote_login.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def config(self):
            return json.loads(Path(config_path).read_text(encoding='utf-8'))

        def send(self, body, status=200, kind='application/json; charset=utf-8', *, headers=()):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('X-Frame-Options', 'DENY')
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def cookie_identifier(self):
            try:
                cookie = SimpleCookie(self.headers.get('Cookie', ''))
                return cookie[COOKIE].value if COOKIE in cookie else ''
            except Exception:
                return ''

        def session(self):
            return pool.lookup(self.cookie_identifier())

        def expected_origin(self):
            host = self.headers.get('Host', '')
            if allow_insecure_local and host in ('127.0.0.1:' + str(self.server.server_port),
                                                 'localhost:' + str(self.server.server_port)):
                return 'http://' + host
            configured = self.config().get('public_url', '')
            if not configured:
                return None
            expected = origin_of(configured)
            return expected if urlsplit(expected).netloc == host else None

        def allowed_origin(self):
            expected = self.expected_origin()
            return expected is not None and self.headers.get('Origin') == expected

        def read_json(self):
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('invalid_request_length')
            return self.rfile.read(length)

        def proxy(self, session, body=None):
            path = '/' if self.path == '/console' else self.path
            port = session.server.server_port
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=30)
            headers = {'Host': '127.0.0.1:' + str(port)}
            if body is not None:
                headers.update({'Content-Type': 'application/json', 'Origin': 'http://127.0.0.1:' + str(port),
                                'X-Console-Token': self.headers.get('X-Console-Token', '')})
            try:
                connection.request(self.command, path, body=body, headers=headers)
                response = connection.getresponse()
                value, kind = response.read(), response.getheader('Content-Type', 'application/json')
                csp = response.getheader('Content-Security-Policy')
                if path in ('/', '/console.js'):
                    value = remote_markup(value)
                extra = [('X-Liquid-Execution', 'server')]
                if path == '/api/status' and response.status == 200:
                    etag = '"' + hashlib.sha256(value).hexdigest() + '"'
                    extra.append(('ETag', etag))
                    if self.headers.get('If-None-Match') == etag:
                        return self.send(b'', 304, headers=extra)
                if csp:
                    extra.append(('Content-Security-Policy', csp))
                self.send(value, response.status, kind, headers=extra)
            except (OSError, http.client.HTTPException):
                self.send({'error': 'private_console_temporarily_unavailable'}, 502)
            finally:
                connection.close()

        def do_GET(self):
            if self.path == '/healthz':
                return self.send(dict(status='ready', service='liquid-remote-console', protocol=1))
            if self.path == '/':
                return self.send(login_page, kind='text/html; charset=utf-8', headers=[
                    ('Content-Security-Policy', "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")])
            if self.path == '/favicon.ico':
                return self.send(b'', 204, 'image/x-icon')
            if self.path not in ASSETS:
                return self.send({'error': 'not_found'}, 404)
            session = self.session()
            if not session:
                if self.path == '/console':
                    return self.send(b'', 303, headers=[('Location', '/')])
                return self.send({'error': 'session_required'}, 401)
            return self.proxy(session)

        def do_POST(self):
            if self.path not in MUTATIONS | {'/session'}:
                return self.send({'error': 'not_found'}, 404)
            if not self.allowed_origin():
                return self.send({'error': 'same_origin_required'}, 403)
            try:
                body = self.read_json()
                if self.path == '/session':
                    value = json.loads(body)
                    supplied = value.get('key', '') if isinstance(value, dict) else ''
                    if not isinstance(supplied, str) or not secrets.compare_digest(supplied, self.config()['access_key']):
                        return self.send({'error': 'invalid_access_key'}, 401)
                    identifier = self.cookie_identifier()
                    session = pool.lookup(identifier)
                    if not session:
                        identifier, session = pool.create()
                    secure = '' if self.expected_origin().startswith('http://') else '; Secure'
                    cookie = f'{COOKIE}={identifier}; Path=/; HttpOnly; SameSite=Lax; Max-Age=7200{secure}'
                    return self.send({'ready': True, 'location': '/console'}, headers=[('Set-Cookie', cookie)])
                session = self.session()
                if not session:
                    return self.send({'error': 'session_required'}, 401)
                return self.proxy(session, body)
            except (ValueError, TypeError, json.JSONDecodeError):
                return self.send({'error': 'invalid_request_or_session_limit'}, 400)
            except RuntimeError:
                return self.send({'error': 'console_initialization_failed'}, 503)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--blender', required=True)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--maximum-sessions', type=int, default=3)
    parser.add_argument('--allow-insecure-local', action='store_true')
    args = parser.parse_args()
    make_config(args.config)
    import cv2
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    pool = SessionPool(args.work.resolve(), args.blender,
        resource_root() / 'assets/models/scene_router/scene_head.pt', maximum=args.maximum_sessions)
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
        handler_for(pool, args.config, allow_insecure_local=args.allow_insecure_local))
    print(json.dumps(dict(event='remote_gateway_ready', port=server.server_port)), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        pool.close_all()


if __name__ == '__main__':
    main()
