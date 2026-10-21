import http.client
import importlib.util
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import uuid

import pytest

from liquid_depth.remote_console import SessionPool, handler_for, make_config, origin_of
from http.server import ThreadingHTTPServer


@pytest.fixture
def portal(tmp_path, monkeypatch):
    import liquid_depth.scene_console_web as local

    class FakeConsole:
        """HTTP integration fixture, not a simulator or accuracy sample."""
        def __init__(self, *args, **kwargs):
            self.lock = threading.RLock()
            self.token = uuid.uuid4().hex
            self.state = {'status': 'not_initialized'}
            self.jobs = {}
            self.worker = SimpleNamespace(shutdown=lambda **kwargs: None)

        def submit(self, controls, initialize=False):
            validated = local.validate_controls(controls)
            job = uuid.uuid4().hex
            self.jobs[job] = {'status': 'complete'}
            self.state = {'status': 'ready', 'controls': validated, 'frame_id': job}
            return job

    monkeypatch.setattr(local, 'Console', FakeConsole)
    config_path = tmp_path / 'private.json'
    config = make_config(config_path)
    config['public_url'] = 'https://example.test'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    pool = SessionPool(tmp_path / 'captures', 'unused', 'unused', maximum=3)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(pool, config_path, allow_insecure_local=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, path, value=None, cookie=None, token=None, origin=True, public=False):
        headers = {}
        expected = 'https://example.test' if public else 'http://127.0.0.1:' + str(server.server_port)
        if public:
            headers['Host'] = 'example.test'
        if origin:
            headers['Origin'] = expected if origin is True else origin
        if cookie:
            headers['Cookie'] = cookie
        if token:
            headers['X-Console-Token'] = token
        body = json.dumps(value).encode() if value is not None else None
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port)
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def login(cookie=None, public=False):
        status, headers, body = request('POST', '/session', {'key': config['access_key']}, cookie, public=public)
        assert status == 200, body
        return headers['Set-Cookie'].split(';')[0], headers

    yield SimpleNamespace(request=request, login=login, pool=pool, config=config)
    server.shutdown()
    server.server_close()
    pool.close_all()
    thread.join(timeout=3)


def test_public_health_does_not_expose_private_state(portal):
    assert portal.request('GET', '/healthz')[0] == 200
    assert portal.request('GET', '/api/status')[0] == 401
    assert portal.request('GET', '/console')[0] == 303
    assert not portal.pool.sessions


def test_key_and_origin_are_required_before_allocating_models(portal):
    assert portal.request('POST', '/session', {'key': 'bad'})[0] == 401
    assert portal.request('POST', '/session', {'key': portal.config['access_key']}, origin='https://other.test')[0] == 403
    assert portal.request('POST', '/session', {'key': portal.config['access_key']}, origin=False)[0] == 403
    assert not portal.pool.sessions


def test_https_session_cookie_is_private_and_repeat_open_reuses_session(portal):
    cookie, headers = portal.login(public=True)
    assert all(item in headers['Set-Cookie'] for item in ('HttpOnly', 'SameSite=Lax', 'Secure'))
    repeated, _ = portal.login(cookie, public=True)
    assert repeated == cookie and len(portal.pool.sessions) == 1


def test_two_browsers_keep_calibration_state_and_csrf_tokens_separate(portal):
    first, _ = portal.login()
    second, _ = portal.login()
    assert first != second
    status = lambda cookie: json.loads(portal.request('GET', '/api/status', cookie=cookie)[2])
    a, b = status(first), status(second)
    assert a['csrf'] != b['csrf']
    assert portal.request('POST', '/api/apply', {'height_m': .8}, first, b['csrf'])[0] == 403
    assert portal.request('POST', '/api/apply', {'height_m': .8}, first, a['csrf'])[0] == 202
    assert status(first)['state']['controls']['height_m'] == .8
    assert status(second)['state']['controls']['height_m'] == .6
    assert portal.request('POST', '/api/apply', {'arbitrary_path': '/tmp'}, first, a['csrf'])[0] == 400


def test_ui_resources_use_same_origin_and_server_branding(portal):
    cookie, _ = portal.login()
    status, _, html = portal.request('GET', '/console', cookie=cookie)
    assert status == 200 and '服务器仿真实验台'.encode() in html
    assert 'CPU离线仿真实验台'.encode() not in html
    assert portal.request('GET', '/console.js', cookie=cookie)[0] == 200
    assert portal.request('GET', '/console.css', cookie=cookie)[0] == 200


def test_session_limit_is_bounded(portal):
    for _ in range(3):
        portal.login()
    status, _, _ = portal.request('POST', '/session', {'key': portal.config['access_key']})
    assert status == 400 and len(portal.pool.sessions) == 3


def test_launcher_keeps_access_key_in_fragment_and_requires_https(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('launcher', root / 'scripts/create_remote_launcher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / 'private.json'
    config.write_text(json.dumps({'public_url': 'https://demo.example', 'access_key': 'x' * 43}), encoding='utf-8')
    output = tmp_path / 'entry.html'
    module.create(config, output)
    text = output.read_text(encoding='utf-8')
    assert '/#key=' in text and '__ACCESS_KEY_JSON__' not in text
    assert 'location.replace(target)' in text
    for invalid in ('http://demo.example', 'https://demo.example/path', 'https://u:p@demo.example'):
        config.write_text(json.dumps({'public_url': invalid, 'access_key': 'x' * 43}), encoding='utf-8')
        with pytest.raises(ValueError):
            module.create(config, output)
