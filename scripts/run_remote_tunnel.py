"""Keep the private HTML entry synchronized with a supervised demo tunnel."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from create_remote_launcher import create
from liquid_depth.remote_console import make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cloudflared', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    make_config(args.config)
    command = [args.cloudflared, 'tunnel', '--no-autoupdate', '--url',
               f'http://127.0.0.1:{args.port}', '--protocol', 'auto', '--edge-ip-version', '4']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line.rstrip(), flush=True)
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', line)
            if not match:
                continue
            value = json.loads(args.config.read_text(encoding='utf-8'))
            endpoint = match.group(0)
            if value.get('public_url') == endpoint and args.launcher.exists():
                continue
            value['public_url'] = endpoint
            temporary = args.config.with_suffix('.updating')
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(value, stream)
            temporary.chmod(0o600)
            temporary.replace(args.config)
            create(args.config, args.launcher)
            print(json.dumps(dict(event='remote_endpoint_ready', url=endpoint)), flush=True)
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)


if __name__ == '__main__':
    raise SystemExit(main())
