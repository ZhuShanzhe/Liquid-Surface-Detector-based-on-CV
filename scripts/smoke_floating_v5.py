"""Exercise real no-USB CLI and prove an unrelated scene cannot load a missing profile."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp'); ROOT = BASE/'liquid-depth-data/synthetic/floating-apex-v5-test-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016/no-usb-example'


def main():
    OUT.mkdir(exist_ok=False)
    item = install(165, ROOT, ROOT/'dry', 202610160821, max_annotation_attempts=6, source_slot_stride=1)
    (OUT/'calibration.json').write_text(json.dumps(item['bundle'], indent=2))
    np.save(OUT/'dry_depth_m.npy', item['dry'])
    folder = ROOT/'00165'; frame = json.loads((folder/'frames.json').read_text())[0]
    common = [sys.executable, 'scripts/measure_floating_apex_v5.py', '--bundle', str(OUT/'calibration.json'),
        '--dry', str(OUT/'dry_depth_m.npy'), '--depth', frame['raw_path']]
    for name, extra in [('float_on', ['--scene', 'floating_objects', '--rgb', str(folder/'rgb.png'), '--enable-floating-peak-vision']),
        ('float_off', ['--scene', 'floating_objects']),
        ('ordinary_blocked', ['--scene', 'ordinary_diffuse', '--profile', '/nonexistent/never-load.json', '--enable-floating-peak-vision'])]:
        command = common+extra
        output = subprocess.run(command, capture_output=True, text=True, check=True)
        result = json.loads(output.stdout)
        assert result['visual_module_executed'] == (name == 'float_on')
        assert not result['liquid_visual_guidance_used']
        (OUT/f'{name}.json').write_text(json.dumps(result, indent=2))
        (OUT/f'{name}-command.json').write_text(json.dumps(command, indent=2))
    on = json.loads((OUT/'float_on.json').read_text()); off = json.loads((OUT/'float_off.json').read_text())
    assert on['liquid_depth_m'] == off['liquid_depth_m']
    (OUT/'smoke-audit.json').write_text(json.dumps({'all_three_cli_calls_passed': True, 'liquid_on_off_identical': True,
        'other_scene_rejected_visual_even_with_missing_profile': True}, indent=2))
    print('NO_USB_SMOKE_COMPLETE', flush=True)


if __name__ == '__main__': main()
