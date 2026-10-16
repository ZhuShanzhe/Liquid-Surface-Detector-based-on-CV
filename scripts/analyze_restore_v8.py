"""Read-only error analysis and current-entry smoke; never retune frozen V8."""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/floating-v8-restored-20261016'
TEST = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016/independent-test'
DATA = BASE/'liquid-depth-data/synthetic/floating-joint-v9-test-20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((REPO/'configs/application_route.json').read_text())
    assert config['selected_cli'] == 'scripts/measure_rank_v8.py'
    assert config['policy_variant'] == 'selected'
    assert not config['default_visual_enabled']
    manifest = json.loads((BASE/'liquid-depth-artifacts/floating-rank-v8-20261016/final-delivery-manifest.json').read_text())
    frozen = {p: sha(REPO/p) == h for p, h in manifest['source_sha256'].items()}
    assert all(frozen.values()), frozen
    protocol = json.loads((TEST/'protocol.json').read_text())
    weight_checks = {p: sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items()
                     if p.endswith(('.joblib', '.pt'))}
    assert all(weight_checks.values()), weight_checks
    rows = [json.loads(line) for line in (TEST/'records.jsonl').read_text().splitlines()]
    wet = [r for r in rows if r['mode'] == 'v8' and r['actual_visible_floating_object']]
    failures = []
    for r in wet:
        error_mm = 1000*(r['highest_height_m']-r['truth']['highest_height_m'])
        if abs(error_mm) <= max(50, 50*r['truth']['highest_height_m']):
            continue
        aux = r['diagnostics']['floating_peak_auxiliary']
        c = r['offline_instance_audit']['chosen']
        bypass = aux['v8_policy'].get('bypass')
        reason = ('selected_bypass_'+bypass if bypass else 'rejected_then_fallback') if not aux['v8_used'] else (
            'wrong_region_or_echo' if not c['physical'] else 'lower_instance_selected'
            if c['gap_to_visible_top_mm'] > 20 else 'highest_instance_metric_error')
        failures.append({'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'band': r['band'],
                         'signed_error_mm': error_mm, 'reason': reason, 'audit': r['offline_instance_audit']})
    adopted = [r for r in wet if r['offline_instance_audit']['chosen'] is not None]
    summary = json.loads((TEST/'summary.json').read_text())
    result = {'source': str(TEST), 'test_seed': 202610164913, 'wet_frames': len(wet),
              'independent_wet_installations': len({r['case'] for r in wet}),
              'failure_counts': dict(Counter(f['reason'] for f in failures)),
              'failure_sensor_counts': dict(Counter(f['sensor'] for f in failures)),
              'failures': sorted(failures, key=lambda f: -abs(f['signed_error_mm'])),
              'adopted_frames': len(adopted),
              'adopted_physical': sum(r['offline_instance_audit']['chosen']['physical'] for r in adopted),
              'adopted_highest_within_20mm': sum(r['offline_instance_audit']['chosen']['physical'] and
                   r['offline_instance_audit']['chosen']['gap_to_visible_top_mm'] <= 20 for r in adopted),
              'summary': {k: v for k, v in summary.items() if k.endswith('|v8')},
              'frozen_v8_source_checks': frozen, 'weights_unchanged': weight_checks,
              'application_config_sha256': sha(REPO/'configs/application_route.json')}
    # Invoke the selected public CLI, not an in-process testing-only backend.
    smoke = []
    for sensor in ['tof', 'active_stereo', 'structured_light']:
        row = next(r for r in wet if r['sensor'] == sensor and r['repeat'] == 0)
        case = row['case']
        item = install(case, DATA, DATA/'dry', 202610164913, max_annotation_attempts=6, source_slot_stride=1)
        folder = OUT/f'smoke-{sensor}'
        folder.mkdir(exist_ok=True)
        (folder/'calibration.json').write_text(json.dumps(item['bundle'], indent=2))
        np.save(folder/'dry.npy', item['dry'])
        frames = json.loads((DATA/f'{case:05d}'/'frames.json').read_text())
        frame = next(f for f in frames if f['repeat'] == 0)
        common = [sys.executable, config['selected_cli'], '--bundle', str(folder/'calibration.json'),
                  '--dry', str(folder/'dry.npy'), '--depth', frame['raw_path'], '--scene', 'floating_objects',
                  '--profile', config['profile'], '--policy-variant', config['policy_variant']]
        outputs = {}
        for name, flags in [('off', []), ('on', ['--enable-floating-peak-vision', '--rgb', str(DATA/f'{case:05d}'/'rgb.png')])]:
            cmd = common+flags
            proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True)
            value = json.loads(proc.stdout)
            assert not value['liquid_visual_guidance_used']
            assert value['visual_module_executed'] == (name == 'on')
            outputs[name] = value
            (folder/f'{name}.json').write_text(json.dumps(value, indent=2))
            (folder/f'{name}-command.json').write_text(json.dumps(cmd, indent=2))
        assert outputs['on']['liquid_depth_m'] == outputs['off']['liquid_depth_m'] == row['liquid_depth_m']
        difference = abs(outputs['on']['highest_height_m']-row['highest_height_m'])
        assert difference < 1e-7, difference
        smoke.append({'sensor': sensor, 'case': case, 'liquid_identical': True,
                      'highest_difference_m_to_frozen_test': difference})
    result['public_entry_smoke'] = smoke
    (OUT/'analysis-and-restoration.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ['summary', 'failures']}), flush=True)
    print('WORST', json.dumps(result['failures'][:5]), flush=True)
    print('V8_RESTORED_AND_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
