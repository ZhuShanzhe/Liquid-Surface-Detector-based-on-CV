"""Exercise the public entry without USB, including lazy scene protection."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp')
ROOT = BASE/'liquid-depth-data/synthetic/floating-apex-v5-validation-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-surface-v7-20261016/no-usb-example'


def main():
    OUT.mkdir(exist_ok=False)
    item = install(51, ROOT, ROOT/'dry', 202610160419, max_annotation_attempts=6, source_slot_stride=1)
    (OUT/'calibration.json').write_text(json.dumps(item['bundle'], indent=2))
    np.save(OUT/'dry_depth_m.npy', item['dry'])
    folder = ROOT/'00051'
    frame = json.loads((folder/'frames.json').read_text())[0]
    common = [sys.executable, 'scripts/measure_surface_v7.py', '--bundle', str(OUT/'calibration.json'),
        '--dry', str(OUT/'dry_depth_m.npy'), '--depth', frame['raw_path']]
    enabled = ['--scene', 'floating_objects', '--rgb', str(folder/'rgb.png'), '--enable-floating-peak-vision']
    outputs = {}
    for mode, extra in [('on', enabled), ('guards', enabled+['--guards-only']),
        ('off', ['--scene', 'floating_objects']),
        ('ordinary_blocked', ['--scene', 'ordinary_diffuse', '--enable-floating-peak-vision', '--profile', '/nonexistent/no-model.json'])]:
        process = subprocess.run(common+extra, capture_output=True, text=True, check=True)
        result = json.loads(process.stdout)
        outputs[mode] = result
        assert result['visual_module_executed'] == (mode in ('on', 'guards'))
        assert not result['liquid_visual_guidance_used']
        (OUT/f'{mode}.json').write_text(json.dumps(result, indent=2))
        (OUT/f'{mode}-command.json').write_text(json.dumps(common+extra, indent=2))
    assert len({v['liquid_depth_m'] for v in outputs.values()}) == 1
    (OUT/'scope-audit.json').write_text(json.dumps({'four_cli_calls_passed': True, 'liquid_identical': True,
        'other_scene_does_not_load_missing_model': True}, indent=2))
    print('V7_SMOKE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
