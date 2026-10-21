"""Install this source snapshot as a separately supervised remote demo service."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from liquid_depth.remote_console import make_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--blender', type=Path, required=True)
    parser.add_argument('--cloudflared', type=Path, required=True)
    parser.add_argument('--supervisord', default='/usr/bin/supervisord')
    args = parser.parse_args()
    home, python, blender, tunnel = (p.resolve() for p in (args.home, args.python, args.blender, args.cloudflared))
    for path in (python, blender, tunnel):
        if not path.is_file():
            raise FileNotFoundError(path)
    home.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'private', 'captures'):
        (home / name).mkdir(exist_ok=True)
    config = home / 'private/access.json'
    make_config(config)
    supervisor = home / 'supervisord.conf'
    if supervisor.exists():
        raise RuntimeError('deployment_already_exists_use_supervisorctl_for_restart')
    text = f'''[unix_http_server]
file={home}/control.sock
chmod=0600
[supervisord]
logfile={home}/logs/supervisor.log
pidfile={home}/supervisord.pid
childlogdir={home}/logs
[rpcinterface:supervisor]
supervisor.rpcinterface_factory=supervisor.rpcinterface:make_main_rpcinterface
[supervisorctl]
serverurl=unix://{home}/control.sock
[program:liquid-remote-gateway]
command={python} -u -m liquid_depth.remote_console --work {home}/captures --config {config} --blender {blender} --port 8767
directory={ROOT}
environment=PYTHONPATH="{ROOT}/src",LIQUID_DEVICE="cuda",LIQUID_RENDER_ENGINE="eevee",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4"
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={home}/logs/gateway.log
stdout_logfile_maxbytes=10MB
priority=10
[program:liquid-remote-tunnel]
command={python} -u {ROOT}/scripts/run_remote_tunnel.py --cloudflared {tunnel} --config {config} --launcher {home}/private/liquid_remote.html --port 8767
directory={ROOT}
environment=PYTHONPATH="{ROOT}/src",PYTHONDONTWRITEBYTECODE="1"
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
redirect_stderr=true
stdout_logfile={home}/logs/tunnel.log
stdout_logfile_maxbytes=10MB
priority=20
'''
    supervisor.write_text(text, encoding='utf-8')
    supervisor.chmod(0o600)
    subprocess.run([args.supervisord, '-c', str(supervisor)], check=True)
    print('Remote demo supervisor started; private launcher will be created after the HTTPS endpoint is ready.')


if __name__ == '__main__':
    main()
