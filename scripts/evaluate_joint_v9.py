"""Paired frozen single/causal evaluation; GT is used only after measurements."""
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
from liquid_depth.floating_joint_v9 import FloatingJointMeterV9
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_rank_v8 import FloatingRankMeterV8
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
V8 = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-joint-v9-test-20261016'
MODES = ['v4', 'v5', 'v8', 'v9_single', 'v9']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    result = {}
    for subset in ['all_frames', 'actual_visible_floats', 'empty_control']:
        strata = ['all']+[f'sensor={s}' for s in ['tof', 'active_stereo', 'structured_light']]+[f'band={i}' for i in range(11)]+[f'repeat={i}' for i in range(4)]
        for stratum in strata:
            for mode in MODES:
                group = [r for r in rows if r['mode'] == mode and
                    (subset == 'all_frames' or r['actual_visible_floating_object'] == (subset == 'actual_visible_floats')) and
                    (stratum == 'all' or stratum == f"sensor={r['sensor']}" or stratum == f"band={r['band']}" or stratum == f"repeat={r['repeat']}")]
                if not group:
                    continue
                e = np.array([abs(r['highest_height_m']-r['truth']['highest_height_m'])*1000 for r in group])
                latency = np.array([r['algorithm_ms'] for r in group])
                result[f'{subset}|{stratum}|{mode}'] = {'highest': metrics(group, 'highest_height_m', True), 'liquid': metrics(group),
                    'max_error_mm': float(e.max()), 'p99_mm': float(np.percentile(e, 99)),
                    'latency_median_ms': float(np.median(latency)), 'latency_p95_ms': float(np.percentile(latency, 95)),
                    'latency_max_ms': float(latency.max()), 'latency_over_500ms_pct': 100*float(np.mean(latency > 500))}
    return result


def offline_audits(inners, folder, repeat, true):
    labels = np.load(folder/'labels_offline.npz')
    sem = labels['semantic']
    _, cc = cv2.connectedComponents((sem == 3).astype(np.uint8))
    heights = world_points(labels['float_depth_m'], true)[..., 2]-true.bottom_z_m
    tops = np.full(int(cc.max())+1, -np.inf)
    np.maximum.at(tops, cc.ravel(), heights.ravel())
    tops[0] = -np.inf
    top = float(tops[1:].max()) if len(tops) > 1 else 0.
    origins = np.load(folder/f'origin_offline_{repeat:02d}.npy').ravel()
    flat_cc = cc.ravel()
    cache = {}

    def inspect(c):
        key = tuple(c['indices'])
        if key not in cache:
            values = flat_cc[c['indices']]
            counts = np.bincount(values, minlength=len(tops))
            counts[0] = 0
            matched = int(counts.argmax())
            physical = matched > 0 and counts[matched]/len(values) >= .6 and (origins[c['indices']] == 2).mean() >= .6
            target = float(tops[matched]) if matched else None
            cache[key] = bool(physical), target
        physical, target = cache[key]
        return {'physical': physical, 'component_top_m': target, 'gap_to_visible_top_mm': 1000*(top-target) if target is not None else None,
            'height_error_to_frame_mm': 1000*(c['height_m']-top), 'predicted_risk_mm': 1000*c['risk_m']}
    audits = {}
    for name, inner in inners.items():
        if not name.startswith(('v8', 'v9')):
            continue
        c = inner._last_chosen
        audits[name] = {'chosen': inspect(c) if c is not None else None, 'visible_top_m': top}
    return bool((sem == 3).any()), audits


def scalar_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run(root, cases, seed, predictor, *, historical=False, checkpoint=None):
    rows, audits = [], []
    for slot, case in enumerate(cases):
        item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        audits.append(item['audit'])
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
        inners = {'v4': OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib'),
            'v5': FloatingApexMeterV5(base, item['dry'], predictor, APEX/'apex-verifier.joblib', OLD/'region-model/verifier.joblib'),
            'v8': FloatingRankMeterV8(base, item['dry'], predictor, V8/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')}
        for name in ['v9_single', 'v9']:
            inners[name] = FloatingJointMeterV9(base, item['dry'], predictor, OUT/'model.joblib', V8/'model.joblib',
                OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib', temporal=name == 'v9')
        if historical:
            inner = FloatingRankMeterV8(base, item['dry'], predictor, V8/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
            inner.policy_override = inner.bundle['research_policy_by_sensor'][base.bundle['sensor_family']]
            inners['v8_research'] = inner
        routes = {n: FloatingPeakOnlyMeter(legacy, lambda b=inner: b) for n, inner in inners.items()}
        folder = root/f'{case:05d}'
        rgb = cv2.imread(str(folder/'rgb.png'))
        frames = json.loads((folder/'frames.json').read_text())
        # Warm CUDA without touching V9 memory or consuming a future burst frame.
        routes['v4'].measure('floating_objects', depth_m=np.load(frames[0]['raw_path']), rgb_bgr=rgb, visual_enabled=True)
        for frame in frames:
            raw = np.load(frame['raw_path'])
            repeat = int(frame['repeat'])
            inners['v9'].set_frame_context(str(root)+str(case), repeat, repeat*.1)
            outputs = {}
            names = list(routes)
            rotate = (slot+repeat) % len(names)
            for name in names[rotate:]+names[:rotate]:
                start = time.perf_counter()
                result = routes[name].measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
                outputs[name] = result, 1000*(time.perf_counter()-start)
            assert len({v[0]['liquid_depth_m'] for v in outputs.values()}) == 1
            assert not any(v[0]['liquid_visual_guidance_used'] for v in outputs.values())
            has_float, offline = offline_audits(inners, folder, repeat, item['true'])
            for name, (result, elapsed) in outputs.items():
                row = {'case': case, 'repeat': repeat, 'mode': name, 'sensor': item['meta']['sensor_family'],
                    'band': item['meta']['band'], 'truth': item['meta']['truth'], 'camera_to_bottom_m': item['meta']['camera_to_bottom_m'],
                    'actual_visible_floating_object': has_float, 'highest_height_m': result['highest_height_m'],
                    'liquid_depth_m': result['liquid_depth_m'], 'algorithm_ms': elapsed, 'diagnostics': result['diagnostics'],
                    'offline_instance_audit': offline.get(name)}
                rows.append(row)
                if checkpoint is not None:
                    checkpoint.write(json.dumps(row, allow_nan=False, default=scalar_json)+'\n')
                    checkpoint.flush()
        if slot % 10 == 0:
            print('V9_EVALUATED', slot, case, flush=True)
    return rows, audits


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    dest = OUT/'independent-test'
    dest.mkdir(exist_ok=False)
    files = [REPO/'src/liquid_depth/floating_joint_v9.py', REPO/'scripts/fit_joint_v9.py', Path(__file__), OUT/'model.joblib',
        OUT/'validation-selection.json', V8/'model.joblib', OLD/'pixel-model/best.pt', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib']
    protocol = {'frozen_sha256': {str(p): sha(p) for p in files}, 'test_seed': 202610164913, 'installations': 99, 'frames': 396,
        'validation_seed': 202610164119, 'train_seed': 202610160317, 'four_exposures_per_installation': True,
        'fixed_scene_same_RGB_independent_depth_noise': True, 'timestamp_spacing_s': .1, 'causal_no_future_frames': True,
        'highest_tolerance': 'max(50mm, 5% physical bottom-relative highest height)',
        'liquid_tolerance': 'max(20mm, 2% physical bottom-relative liquid depth)',
        'all_frames_retained_including_cold_start': True, 'timing': 'warm_models_cyclic_method_order_no_IO_no_capture',
        'visual_scope': 'floating_objects_highest_height_only'}
    (dest/'protocol.json').write_text(json.dumps(protocol, indent=2))
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    with (dest/'records.partial.jsonl').open('x') as checkpoint:
        rows, audits = run(ROOT, [3+6*i for i in range(99)], 202610164913, predictor, checkpoint=checkpoint)
    (dest/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False, default=scalar_json)+'\n' for r in rows))
    summary = summarize(rows)
    (dest/'summary.json').write_text(json.dumps(summary, indent=2))
    (dest/'calibration-audit.json').write_text(json.dumps(audits, indent=2))
    replay = []
    for root, cases, seed in [(BASE/'liquid-depth-data/synthetic/floating-rank-v8-test-20261016', [303, 1011, 1227], 202610163811),
        (BASE/'liquid-depth-data/synthetic/floating-apex-v5-test-20261016', [1023], 202610160821)]:
        previous, _ = run(root, cases, seed, predictor, historical=True)
        for r in previous:
            r['source_root'] = str(root)
        replay += previous
    (dest/'historical-diagnostics.jsonl').write_text(''.join(json.dumps(r, allow_nan=False, default=scalar_json)+'\n' for r in replay))
    assert all(sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items())
    print('V9_EVALUATION_COMPLETE', json.dumps({k: v for k, v in summary.items() if 'actual_visible_floats|all|' in k}), flush=True)


if __name__ == '__main__':
    main()
