"""Frozen paired evaluation on a fresh 528-frame simulation, no frame exclusion."""
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
from liquid_depth.floating_rank_v8 import FloatingRankMeterV8
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-rank-v8-test-20261016'
SEED = 202610163811
MODES = ['v4', 'v5', 'v8_no_rank', 'v8', 'v8_research']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    result = {}
    for subset in ['all_category_frames', 'actual_visible_floats', 'empty_control']:
        for stratum in ['all']+[f'sensor={s}' for s in ['tof', 'active_stereo', 'structured_light']]+[f'band={i}' for i in range(11)]:
            for mode in MODES:
                selected = [r for r in rows if r['mode'] == mode and
                    (subset == 'all_category_frames' or r['actual_visible_floating_object'] == (subset == 'actual_visible_floats')) and
                    (stratum == 'all' or stratum == f"sensor={r['sensor']}" or stratum == f"band={r['band']}")]
                if not selected:
                    continue
                errors = np.array([abs(r['highest_height_m']-r['truth']['highest_height_m'])*1000 for r in selected])
                latency = [r['algorithm_ms'] for r in selected]
                result[f'{subset}|{stratum}|{mode}'] = {'highest': metrics(selected, 'highest_height_m', True), 'liquid': metrics(selected),
                    'max_error_mm': float(errors.max()), 'p99_mm': float(np.percentile(errors, 99)),
                    'latency_median_ms': float(np.median(latency)), 'latency_p95_ms': float(np.percentile(latency, 95)),
                    'latency_max_ms': float(max(latency)), 'latency_over_500ms_pct': 100*float(np.mean(np.asarray(latency) > 500))}
    return result


def offline_audit(inner, labels, origins, true):
    """Called after all routes measured; cannot change a measurement or policy."""
    gt = labels['semantic'] == 3
    _, cc = cv2.connectedComponents(gt.astype(np.uint8))
    heights = world_points(labels['float_depth_m'], true)[..., 2]-true.bottom_z_m
    tops = {i: float(heights[cc == i].max()) for i in range(1, int(cc.max())+1)}
    top = max(tops.values(), default=0.)

    def inspect(c):
        mask = c['mask']
        ids, counts = np.unique(cc[mask], return_counts=True)
        matched, overlap = max([(int(i), int(n)) for i, n in zip(ids, counts, strict=True) if i], key=lambda p: p[1], default=(0, 0))
        physical = overlap/max(c['pixels'], 1) >= .6 and (origins[mask] == 2).mean() >= .6
        t = tops.get(matched)
        return {'physical': bool(physical), 'component_top_m': t,
            'gap_to_visible_top_mm': 1000*(top-t) if t is not None else None,
            'height_error_to_component_mm': 1000*(c['height_m']-t) if t is not None else None,
            'predicted_risk_mm': c['risk_m']*1000, 'predicted_top_gap_mm': c['top_gap_m']*1000}
    candidates = [inspect(c) for c in inner._last_scored]
    return {'visible_top_m': top, 'candidate_count': len(candidates),
        'highest_instance_candidate_present': any(c['physical'] and c['gap_to_visible_top_mm'] <= 20 for c in candidates),
        'chosen': inspect(inner._last_chosen) if inner._last_chosen is not None else None}


def run(root, cases, seed, predictor, *, audit=True):
    rows = []
    audits = []
    warmed = False
    for slot, case in enumerate(cases):
        item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        audits.append(item['audit'])
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
        inners = {'v4': OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib'),
            'v5': FloatingApexMeterV5(base, item['dry'], predictor, APEX/'apex-verifier.joblib', OLD/'region-model/verifier.joblib')}
        for mode in MODES[2:]:
            inner = FloatingRankMeterV8(base, item['dry'], predictor, OUT/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
            sensor = base.bundle['sensor_family']
            if mode == 'v8_no_rank':
                inner.policy_override = {**inner.bundle['research_policy_by_sensor'][sensor], 'gap_weight': 0.}
            elif mode == 'v8_research':
                inner.policy_override = inner.bundle['research_policy_by_sensor'][sensor]
            inners[mode] = inner
        routes = {name: FloatingPeakOnlyMeter(legacy, lambda b=inner: b) for name, inner in inners.items()}
        folder = root/f'{case:05d}'
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path'])
            if not warmed:
                # A complete untimed measurement initializes CUDA kernels and CPU pools.
                routes['v4'].measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
                warmed = True
            outputs = {}
            rotation = (slot+int(frame['repeat'])) % len(MODES)
            order = MODES[rotation:]+MODES[:rotation]
            for name in order:
                start = time.perf_counter()
                result = routes[name].measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
                outputs[name] = result, 1000*(time.perf_counter()-start)
            assert all(v[0]['liquid_depth_m'] == outputs['v4'][0]['liquid_depth_m'] for v in outputs.values())
            assert not any(v[0]['liquid_visual_guidance_used'] for v in outputs.values())
            labels = np.load(folder/'labels_offline.npz')
            has_float = bool((labels['semantic'] == 3).any())
            origins = np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
            for name, (result, latency) in outputs.items():
                rows.append({'case': case, 'repeat': int(frame['repeat']), 'mode': name, 'sensor': item['meta']['sensor_family'],
                    'band': item['meta']['band'], 'truth': item['meta']['truth'], 'camera_to_bottom_m': item['meta']['camera_to_bottom_m'],
                    'actual_visible_floating_object': has_float, 'highest_height_m': result['highest_height_m'],
                    'liquid_depth_m': result['liquid_depth_m'], 'algorithm_ms': latency, 'diagnostics': result['diagnostics'],
                    'offline_instance_audit': offline_audit(inners[name], labels, origins, item['true']) if audit and name.startswith('v8') else None})
        if slot % 10 == 0:
            print('V8_EVALUATED', slot, case, flush=True)
    return rows, audits


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    dest = OUT/'independent-test'
    dest.mkdir(exist_ok=False)
    paths = [REPO/'src/liquid_depth/floating_rank_v8.py', REPO/'scripts/fit_rank_v8.py', Path(__file__),
        OUT/'model.joblib', OUT/'validation-selection.json', OLD/'pixel-model/best.pt', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib']
    protocol = {'frozen_sha256': {str(p): sha(p) for p in paths}, 'test_seed': SEED, 'test_installations': 264, 'test_frames': 528,
        'train_seed': 202610160317, 'validation_seed': 202610160419, 'all_frames_retained': True,
        'scope': 'floating_objects_highest_height_only', 'highest_tolerance': 'max(50mm, 5% true bottom-relative highest height)',
        'liquid_tolerance': 'max(20mm, 2% true bottom-relative liquid depth)', 'timing_order': 'cyclic_rotation_after_warmup',
        'no_zero_snap_no_height_clipping': True}
    (dest/'protocol.json').write_text(json.dumps(protocol, indent=2))
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    rows, audit = run(ROOT, [3+6*i for i in range(264)], SEED, predictor)
    (dest/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    summary = summarize(rows)
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    (dest/'calibration-audit.json').write_text(json.dumps(audit, indent=2))
    replay = []
    for root, cases, seed in [(BASE/'liquid-depth-data/synthetic/floating-surface-v7-test-20261016', [315, 477, 507], 202610162711),
        (BASE/'liquid-depth-data/synthetic/floating-apex-v5-test-20261016', [1023], 202610160821)]:
        previous, _ = run(root, cases, seed, predictor)
        for row in previous:
            row['source_root'] = str(root)
        replay += previous
    (dest/'historical-diagnostics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in replay))
    assert all(sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items())
    print('V8_EVALUATION_COMPLETE', json.dumps({k: v for k, v in summary.items() if 'actual_visible_floats|all|' in k}), flush=True)


if __name__ == '__main__':
    main()
