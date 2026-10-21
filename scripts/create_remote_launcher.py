"""Generate the private, transferable HTML entry for a running HTTPS gateway."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from liquid_depth.remote_console import origin_of
from liquid_depth.runtime_paths import resource_root


def create(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    url = origin_of(config['public_url'])
    if not url.startswith('https://'):
        raise ValueError('public_launcher_requires_https')
    template = (resource_root() / 'src/liquid_depth/ui/remote_launch.html').read_text(encoding='utf-8')
    encode = lambda value: json.dumps(value, ensure_ascii=False).replace('<', '\\u003c')
    text = template.replace('__REMOTE_URL_JSON__', encode(url)).replace('__ACCESS_KEY_JSON__', encode(config['access_key']))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    output.chmod(0o600)
    print(json.dumps(dict(event='html_entry_created', bytes=output.stat().st_size), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    create(args.config, args.output)
