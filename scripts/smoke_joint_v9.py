"""Public causal-sequence entry plus off and nonfloating lazy-load tests."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp')
ROOT = BASE/'liquid-depth-data/synthetic/floating-joint-v9-validation-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016/no-usb-example'


def main():
    OUT.mkdir(exist_ok=False)
    item = install(51, ROOT, ROOT/'dry', 202610164119, max_annotation_attempts=6, source_slot_stride=1)
    (OUT/'calibration.json').write_text(json.dumps(item['bundle'], indent=2))
    np.save(OUT/'dry_depth_m.npy', item['dry'])
    folder = ROOT/'00051'
    frames = json.loads((folder/'frames.json').read_text())
    sequence = [{'depth': f['raw_path'], 'rgb': str(folder/'rgb.png'), 'frame_id': int(f['repeat']), 'timestamp_s': int(f['repeat'])*.1} for f in frames]
    (OUT/'sequence.json').write_text(json.dumps(sequence, indent=2))
    common = [sys.executable, 'scripts/measure_joint_v9.py', '--bundle', str(OUT/'calibration.json'),
        '--dry', str(OUT/'dry_depth_m.npy'), '--sequence', str(OUT/'sequence.json'), '--stream-id', 'smoke_fixed_installation']
    results = {}
    for mode, extra in [('on', ['--enable-floating-peak-vision']), ('single', ['--enable-floating-peak-vision', '--stateless']),
        ('off', []), ('other_blocked', ['--scene', 'ordinary_diffuse', '--enable-floating-peak-vision', '--profile', '/nonexistent/model.json'])]:
        process = subprocess.run(common+extra, capture_output=True, text=True, check=True)
        rows = [json.loads(line) for line in process.stdout.splitlines()]
        assert len(rows) == 4
        for r in rows:
            assert r['visual_module_executed'] == (mode in ('on', 'single'))
            assert not r['liquid_visual_guidance_used']
        results[mode] = rows
        (OUT/f'{mode}.json').write_text(json.dumps(rows, indent=2))
        (OUT/f'{mode}-command.json').write_text(json.dumps(common+extra, indent=2))
    assert all(len({rows[i]['liquid_depth_m'] for rows in results.values()}) == 1 for i in range(4))
    (OUT/'scope-audit.json').write_text(json.dumps({'four_cli_sequences_passed': True, 'frames_per_sequence': 4,
        'liquid_identical': True, 'other_scene_does_not_load_missing_model': True}, indent=2))
    print('V9_SMOKE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
