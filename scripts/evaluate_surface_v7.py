"""Freeze parameters first, then compare V4/V5/checks/facets on a new seed."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from evaluate_visual_route_v1 import metrics

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_apex_v5 import FloatingApexMeterV5
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_surface_v6 import FloatingSurfaceMeterV6
from liquid_depth.floating_surface_v7 import FloatingSurfaceMeterV7
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'; APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
PREVIOUS = BASE/'liquid-depth-artifacts/floating-surface-v6-20261016'; OUT = BASE/'liquid-depth-artifacts/floating-surface-v7-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-surface-v7-test-20261016'; SEED = 202610162711


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root, cases, seed, predictor, previous=False):
    rows = []; audits = []
    for slot, case in enumerate(cases):
        folder = root/f'{case:05d}'; item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        audits.append(item['audit']); base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
        inners = {'v4': OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib'),
            'v5': FloatingApexMeterV5(base, item['dry'], predictor, APEX/'apex-verifier.joblib', OLD/'region-model/verifier.joblib')}
        for name in ['v7_guards', 'v7']:
            inner = FloatingSurfaceMeterV7(base, item['dry'], predictor, OUT/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
            if name == 'v7_guards': inner.bundle = {**inner.bundle, 'policy_by_sensor': inner.bundle['guard_policy_by_sensor']}
            inners[name] = inner
        if previous:
            inners['v6'] = FloatingSurfaceMeterV6(base, item['dry'], predictor, PREVIOUS/'surface-reliability.joblib',
                OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
        routes = {name: FloatingPeakOnlyMeter(legacy, lambda b=inner: b) for name, inner in inners.items()}
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); outputs = {}
            for name, route in routes.items():
                start = time.perf_counter(); result = route.measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
                outputs[name] = (result, 1000*(time.perf_counter()-start))
            assert all(v[0]['liquid_depth_m'] == outputs['v4'][0]['liquid_depth_m'] for v in outputs.values())
            assert not any(v[0]['liquid_visual_guidance_used'] for v in outputs.values())
            # Offline labels and truth used after all measurements, never for eligibility.
            semantic = np.load(folder/'labels_offline.npz')['semantic']; has_float = bool((semantic == 3).any())
            for name, (result, elapsed) in outputs.items():
                rows.append({'case': case, 'repeat': int(frame['repeat']), 'mode': name, 'sensor': item['meta']['sensor_family'],
                    'band': item['meta']['band'], 'truth': item['meta']['truth'], 'camera_to_bottom_m': item['meta']['camera_to_bottom_m'],
                    'actual_visible_floating_object': has_float, 'highest_height_m': result['highest_height_m'],
                    'liquid_depth_m': result['liquid_depth_m'], 'algorithm_ms': elapsed, 'diagnostics': result['diagnostics']})
        if slot % 10 == 0: print('V7_EVALUATED', slot, case, flush=True)
    return rows, audits


def summarize(rows):
    summary = {}
    for subset in ['all_category_frames', 'actual_visible_floats', 'empty_control']:
        selected = [r for r in rows if subset == 'all_category_frames' or
            (r['actual_visible_floating_object'] if subset == 'actual_visible_floats' else not r['actual_visible_floating_object'])]
        for stratum in ['all']+[f'sensor={s}' for s in ['tof', 'active_stereo', 'structured_light']]+[f'band={b}' for b in sorted({r['band'] for r in rows})]:
            for mode in ['v4', 'v5', 'v7_guards', 'v7']:
                group = [r for r in selected if r['mode'] == mode and (stratum == 'all' or stratum == f"sensor={r['sensor']}" or stratum == f"band={r['band']}")]
                if not group: continue
                errors = np.array([abs(r['highest_height_m']-r['truth']['highest_height_m']) for r in group])
                latency = [r['algorithm_ms'] for r in group]
                summary[f'{subset}|{stratum}|{mode}'] = {'highest': metrics(group, 'highest_height_m', True), 'liquid': metrics(group),
                    'max_error_mm': float(1000*errors.max()), 'p99_mm': float(1000*np.percentile(errors, 99)),
                    'errors_ge_1m': int((errors >= 1).sum()), 'latency_median_ms': float(np.median(latency)), 'latency_p95_ms': float(np.percentile(latency, 95))}
    return summary


def main():
    torch.set_num_threads(4); cv2.setNumThreads(1); dest = OUT/'independent-test'; dest.mkdir(exist_ok=False)
    sources = [REPO/'src/liquid_depth/floating_surface_v7.py', REPO/'scripts/fit_surface_v7.py', Path(__file__),
        OUT/'model.joblib', OUT/'validation-selection.json', OLD/'pixel-model/best.pt', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib']
    protocol = {'frozen_sha256': {str(p): sha(p) for p in sources}, 'test_seed': SEED, 'test_installations': 198, 'test_frames': 396,
        'train_seed': 202610160317, 'validation_seed': 202610160419, 'all_frames_retained': True, 'liquid_unchanged': True,
        'visual_scope': 'floating_objects_highest_height_only', 'no_zero_snap_no_height_clipping': True,
        'highest_tolerance': 'max(50mm, 5% actual bottom-relative highest height)',
        'liquid_tolerance': 'max(20mm, 2% actual bottom-relative liquid height)'}
    (dest/'protocol.json').write_text(json.dumps(protocol, indent=2))
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    rows, audit = run(ROOT, [3+6*i for i in range(198)], SEED, predictor)
    (dest/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    summary = summarize(rows); (dest/'summary.json').write_text(json.dumps(summary, indent=2)); (dest/'calibration-audit.json').write_text(json.dumps(audit, indent=2))
    diagnostics = []
    for old_root, cases, seed in [(BASE/'liquid-depth-data/synthetic/floating-surface-v6-test-20261016', [129], 202610161337),
        (BASE/'liquid-depth-data/synthetic/floating-apex-v5-test-20261016', [63, 1023], 202610160821),
        (BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015', [621, 633, 639, 645, 657], 202610155927)]:
        replay, _ = run(old_root, cases, seed, predictor, previous=True)
        for row in replay: row['source_root'] = str(old_root)
        diagnostics += replay
    (dest/'prior-failure-diagnostics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in diagnostics))
    assert all(sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items())
    print('V7_EVALUATION_COMPLETE', json.dumps({k: v for k, v in summary.items() if 'actual_visible_floats|all|' in k}), flush=True)


if __name__ == '__main__': main()
