"""Frozen physical-surface experiment on a fresh independent rendering seed."""
import hashlib
import json
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from evaluate_visual_route_v1 import metrics

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_apex_v5 import FloatingApexMeterV5, score_v5
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_surface_v6 import FloatingSurfaceMeterV6, select_surface
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'; APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-surface-v6-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-surface-v6-test-20261016'
SEED = 202610161337


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_run(root, cases, seed, predictor):
    rows = []; audits = []
    previous = joblib.load(OLD/'region-model/verifier.joblib'); apex = joblib.load(APEX/'apex-verifier.joblib')
    for slot, case in enumerate(cases):
        folder = root/f'{case:05d}'; item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        audits.append(item['audit']); base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
        old_inner = OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib')
        v5_inner = FloatingApexMeterV5(base, item['dry'], predictor, APEX/'apex-verifier.joblib', OLD/'region-model/verifier.joblib')
        v6_inner = FloatingSurfaceMeterV6(base, item['dry'], predictor, OUT/'surface-reliability.joblib',
            OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
        meters = {'v4': FloatingPeakOnlyMeter(legacy, lambda b=old_inner: b),
            'v5': FloatingPeakOnlyMeter(legacy, lambda b=v5_inner: b),
            'surface_reliable': FloatingPeakOnlyMeter(legacy, lambda b=v6_inner: b)}
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); outputs = {}
            for name, meter in meters.items():
                start = time.perf_counter(); result = meter.measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
                outputs[name] = (result, 1000*(time.perf_counter()-start))
            reconstructed = score_v5(v6_inner._last_reconstructed, apex['classifier'], previous)
            chosen, accepted = select_surface(reconstructed, apex['policy_by_sensor'][base.bundle['sensor_family']])
            liquid = outputs['v4'][0]['liquid_depth_m']
            ablation_peak = outputs['v4'][0]['highest_height_m'] if chosen is None else max(liquid, chosen['height_m'])
            ablation = {**outputs['v4'][0], 'highest_height_m': ablation_peak,
                'diagnostics': {'surface_used': chosen is not None, 'accepted_surfaces': len(accepted)}}
            outputs['surface_old_score'] = (ablation, None)
            assert all(v[0]['liquid_depth_m'] == liquid for v in outputs.values())
            # Only NOW access labels and metric targets, after all four measurements.
            semantic = np.load(folder/'labels_offline.npz')['semantic']; has_float = bool((semantic == 3).any())
            for mode, (result, elapsed) in outputs.items():
                rows.append({'case': case, 'repeat': int(frame['repeat']), 'mode': mode, 'sensor': item['meta']['sensor_family'],
                    'band': item['meta']['band'], 'truth': item['meta']['truth'], 'camera_to_bottom_m': item['meta']['camera_to_bottom_m'],
                    'actual_visible_floating_object': has_float, 'highest_height_m': result['highest_height_m'],
                    'liquid_depth_m': result['liquid_depth_m'], 'algorithm_ms': elapsed, 'diagnostics': result['diagnostics']})
        if slot % 10 == 0: print('SURFACE_EVALUATED', slot, case, flush=True)
    return rows, audits


def summarize(rows):
    summary = {}
    for subset in ['all_category_frames', 'actual_visible_floats', 'empty_control']:
        selected = [r for r in rows if subset == 'all_category_frames' or
            (r['actual_visible_floating_object'] if subset == 'actual_visible_floats' else not r['actual_visible_floating_object'])]
        for stratum in ['all']+[f'sensor={s}' for s in ['tof', 'active_stereo', 'structured_light']]+[f'band={b}' for b in sorted({r['band'] for r in rows})]:
            for mode in ['v4', 'v5', 'surface_old_score', 'surface_reliable']:
                group = [r for r in selected if r['mode'] == mode and (stratum == 'all' or stratum == f"sensor={r['sensor']}" or stratum == f"band={r['band']}")]
                if not group: continue
                e = [abs(r['highest_height_m']-r['truth']['highest_height_m']) for r in group]
                latency = [r['algorithm_ms'] for r in group if r['algorithm_ms'] is not None]
                summary[f'{subset}|{stratum}|{mode}'] = {'highest': metrics(group, 'highest_height_m', True),
                    'liquid': metrics(group), 'max_error_mm': float(1000*max(e)), 'p99_mm': float(1000*np.percentile(e, 99)),
                    'errors_ge_1m': sum(int(v >= 1) for v in e),
                    'latency_median_ms': float(np.median(latency)) if latency else None,
                    'latency_p95_ms': float(np.percentile(latency, 95)) if latency else None}
    return summary


def main():
    torch.set_num_threads(4); cv2.setNumThreads(1)
    destination = OUT/'independent-test'; destination.mkdir(exist_ok=False)
    paths = [REPO/'src/liquid_depth/floating_surface_v6.py', REPO/'scripts/fit_surface_v6.py', Path(__file__),
        OUT/'surface-reliability.joblib', OUT/'validation-selection.json', APEX/'apex-verifier.joblib',
        OLD/'pixel-model/best.pt', OLD/'region-model/verifier.joblib']
    protocol = {'frozen_sha256': {str(p): sha(p) for p in paths}, 'test_seed': SEED, 'installations': 198, 'frames': 396,
        'train_seed': 202610160317, 'validation_seed': 202610160419, 'all_frames_retained': True,
        'highest_tolerance': 'max(50mm, 5% actual bottom-relative highest height)',
        'liquid_tolerance': 'max(20mm, 2% actual bottom-relative liquid depth)',
        'no_zero_snap_no_height_clipping': True, 'visual_scope': 'floating_objects_highest_only',
        'surface_old_score_is_shared_evidence_ablation_not_separately_timed': True}
    (destination/'protocol.json').write_text(json.dumps(protocol, indent=2))
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    rows, audits = paired_run(ROOT, [3+6*i for i in range(198)], SEED, predictor)
    (destination/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    summary = summarize(rows); (destination/'summary.json').write_text(json.dumps(summary, indent=2))
    (destination/'calibration-audit.json').write_text(json.dumps(audits, indent=2))
    # Previous known failures are diagnostic replay, not this experiment's final test.
    prior_root = BASE/'liquid-depth-data/synthetic/floating-apex-v5-test-20261016'
    replay, _ = paired_run(prior_root, [63, 1023], 202610160821, predictor)
    (destination/'previous-extremes-diagnostic.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in replay))
    original_root = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'
    replay, _ = paired_run(original_root, [621, 633, 639, 645, 657], 202610155927, predictor)
    (destination/'old-nine-diagnostic.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in replay))
    assert all(sha(Path(p)) == digest for p, digest in protocol['frozen_sha256'].items())
    print('SURFACE_EVALUATION_COMPLETE', json.dumps({k: v for k, v in summary.items() if 'actual_visible_floats|all|' in k}), flush=True)


if __name__ == '__main__': main()
