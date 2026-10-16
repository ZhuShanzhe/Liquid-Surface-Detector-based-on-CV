"""Freeze first, then paired held-out evaluation. No truth enters a measuring route."""
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
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
OUT = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
DATA = BASE/'liquid-depth-data/synthetic/floating-apex-v5-test-20261016'
SEED = 202610160821


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cases(root, cases, seed, predictor):
    records = []; audit = []
    for index, case in enumerate(cases):
        folder = root/f'{case:05d}'
        item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        audit.append(item['audit']); base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
        v4 = FloatingPeakOnlyMeter(legacy, lambda b=base, dry=item['dry']: OptionalFloatMeterV3(b, dry, predictor, OLD/'region-model/verifier.joblib'))
        backend = FloatingApexMeterV5(base, item['dry'], predictor, OUT/'apex-verifier.joblib', OLD/'region-model/verifier.joblib')
        v5 = FloatingPeakOnlyMeter(legacy, lambda b=backend: b)
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); outputs = {}
            for name, route, enabled in [('geometry', v4, False), ('v4', v4, True), ('v5', v5, True)]:
                start = time.perf_counter()
                result = route.measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=enabled)
                elapsed = 1000*(time.perf_counter()-start)
                outputs[name] = (result, elapsed)
            assert outputs['v4'][0]['liquid_depth_m'] == outputs['v5'][0]['liquid_depth_m'] == outputs['geometry'][0]['liquid_depth_m']
            assert not outputs['v5'][0]['liquid_visual_guidance_used']
            # Offline truth and component matching begin only after ALL outputs are fixed.
            labels = np.load(folder/'labels_offline.npz'); gt = labels['semantic'] == 3
            _, components = cv2.connectedComponents(gt.astype(np.uint8))
            true_height = world_points(labels['float_depth_m'], item['true'])[..., 2]-item['true'].bottom_z_m
            apex_component = max(range(1, int(components.max())+1), key=lambda k: true_height[components == k].max(), default=0)
            offline_candidates = []
            for c in backend._last_scored:
                apex_overlap = int((c['mask'] & (components == apex_component)).sum()) if apex_component else 0
                offline_candidates.append({'pixels': c['pixels'], 'variant': c['variant'], 'object_score': c['object_score'],
                    'height_raw_m': c['height_raw_m'], 'correction_m': c['correction_m'],
                    'highest_object_overlap_pixels': apex_overlap,
                    'highest_object_purity': apex_overlap/max(c['pixels'], 1)})
            for name, (result, elapsed) in outputs.items():
                records.append({'case': case, 'repeat': int(frame['repeat']), 'mode': name, 'scene': 'floating_objects',
                    'sensor': item['meta']['sensor_family'], 'band': item['meta']['band'],
                    'camera_to_bottom_m': item['meta']['camera_to_bottom_m'], 'truth': item['meta']['truth'],
                    'actual_visible_floating_object': bool(gt.any()), 'liquid_depth_m': result['liquid_depth_m'],
                    'highest_height_m': result['highest_height_m'], 'algorithm_ms': elapsed,
                    'visual_guidance_used': result['visual_guidance_used'], 'diagnostics': result['diagnostics'],
                    'offline_candidate_audit': offline_candidates if name == 'v5' else []})
        if index % 10 == 0: print('EVALUATED', index, case, flush=True)
    return records, audit


def summarize(rows):
    groups = {}
    for subset in ['all_category_frames', 'actual_visible_floats', 'empty_control']:
        selected = [r for r in rows if subset == 'all_category_frames' or
            (r['actual_visible_floating_object'] if subset == 'actual_visible_floats' else not r['actual_visible_floating_object'])]
        for stratum in ['all']+[f'sensor={s}' for s in ['tof', 'active_stereo', 'structured_light']]+[f'band={b}' for b in sorted({r['band'] for r in rows})]:
            for mode in ['geometry', 'v4', 'v5']:
                branch = [r for r in selected if r['mode'] == mode and (stratum == 'all' or
                    stratum == f"sensor={r['sensor']}" or stratum == f"band={r['band']}")]
                if not branch: continue
                latency = [r['algorithm_ms'] for r in branch]
                groups[f'{subset}|{stratum}|{mode}'] = {'highest': metrics(branch, 'highest_height_m', True),
                    'liquid': metrics(branch), 'algorithm_latency_median_ms': float(np.median(latency)),
                    'algorithm_latency_p95_ms': float(np.percentile(latency, 95)),
                    'peak_tolerance_failures': sum(int(abs(r['highest_height_m']-r['truth']['highest_height_m']) >
                        max(.05, .05*r['truth']['highest_height_m'])) for r in branch)}
    return groups


def bootstrap(rows):
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    wet = [r for r in rows if r['mode'] == 'v5' and r['actual_visible_floating_object']]
    cases = sorted({r['case'] for r in wet}); rng = np.random.default_rng(160821)
    deltas = []
    for _ in range(2000):
        sampled = rng.choice(cases, len(cases)); mae = []; passed = []
        for case in sampled:
            for r in (x for x in wet if x['case'] == case):
                truth = r['truth']['highest_height_m']; old = lookup[(r['case'], r['repeat'], 'v4')]
                a = abs(r['highest_height_m']-truth); b = abs(old['highest_height_m']-truth); tol = max(.05, .05*truth)
                mae.append(1000*(a-b)); passed.append(100*(int(a <= tol)-int(b <= tol)))
        deltas.append([np.mean(mae), np.mean(passed)])
    return {'sampling_unit': 'installation', 'replicates': 2000,
        'new_minus_old_mae_mm_95pct_ci': np.percentile(np.asarray(deltas)[:, 0], [2.5, 97.5]).tolist(),
        'new_minus_old_pass_percentage_points_95pct_ci': np.percentile(np.asarray(deltas)[:, 1], [2.5, 97.5]).tolist()}


def main():
    torch.set_num_threads(4); cv2.setNumThreads(1)
    destination = OUT/'independent-test'; destination.mkdir(exist_ok=False)
    paths = [REPO/'src/liquid_depth/floating_apex_v5.py', REPO/'src/liquid_depth/floating_apex_application_v5.py',
        REPO/'scripts/fit_floating_v5.py', Path(__file__), OUT/'apex-verifier.joblib', OUT/'validation-selection.json',
        OLD/'pixel-model/best.pt', OLD/'region-model/verifier.joblib']
    protocol = {'frozen_sha256': {str(p): sha(p) for p in paths}, 'final_test_seed': SEED,
        'train_seed': 202610160317, 'validation_seed': 202610160419, 'test_installations': 198, 'repeats': 2,
        'highest_tolerance': 'max(50mm, 5% true bottom-relative highest height)',
        'liquid_tolerance': 'max(20mm, 2% true bottom-relative liquid height)',
        'all_test_frames_retained': True, 'liquid_route_changed': False, 'visual_scope': 'floating_objects_highest_only',
        'no_zero_snap_or_height_clipping': True, 'old_nine_frames_not_training_or_validation': True}
    (destination/'protocol.json').write_text(json.dumps(protocol, indent=2))
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    rows, audit = run_cases(DATA, [3+6*i for i in range(198)], SEED, predictor)
    (destination/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    summary = summarize(rows)
    (destination/'summary.json').write_text(json.dumps(summary, indent=2))
    (destination/'calibration-audit.json').write_text(json.dumps(audit, indent=2))
    (destination/'paired-bootstrap.json').write_text(json.dumps(bootstrap(rows), indent=2))
    failures = [r for r in rows if r['mode'] == 'v5' and abs(r['highest_height_m']-r['truth']['highest_height_m']) > max(.05, .05*r['truth']['highest_height_m'])]
    (destination/'failures.json').write_text(json.dumps(failures, indent=2))
    old_data = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'
    replay, _ = run_cases(old_data, [621, 633, 639, 645, 657], 202610155927, predictor)
    (destination/'old-nine-diagnostic-replay.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in replay))
    assert all(sha(Path(p)) == digest for p, digest in protocol['frozen_sha256'].items())
    print('EVALUATION_COMPLETE', json.dumps({k: v for k, v in summary.items() if '|all|' in k}), flush=True)


if __name__ == '__main__': main()
