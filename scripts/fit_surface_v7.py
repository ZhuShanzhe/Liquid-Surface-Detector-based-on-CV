"""Train metric-candidate reliability and select checks/facets on validation only."""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_apex_v5 import observed_peak_roi, score_v5, select_v5
from liquid_depth.floating_surface_v7 import fast_candidates, fit_candidate_models, select_models
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'; OUT = BASE/'liquid-depth-artifacts/floating-surface-v7-20261016'
SPECS = {'train': (202610160317, 198), 'validation': (202610160419, 99)}


def capture(split, predictor, previous, apex):
    cache = OUT/f'{split}-models.joblib'
    if cache.exists(): return joblib.load(cache)
    seed, count = SPECS[split]; root = BASE/f'liquid-depth-data/synthetic/floating-apex-v5-{split}-20261016'; rows = []
    for slot in range(count):
        case = 3+6*slot; folder = root/f'{case:05d}'; item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        frozen = FrozenSymmetricGeometryMeter(base); rgb = cv2.imread(str(folder/'rgb.png'))
        fallback = OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib')
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); liquid = frozen.measure(depth_m=raw)['liquid_depth_m']
            roi = observed_peak_roi(base, raw); pred = predictor.predict(observed_features(rgb, raw, item['dry'], roi, base.bundle))
            candidates = score_v5(fast_candidates(base, item['dry'], rgb, raw, pred, liquid, roi), apex['classifier'], previous)
            fraction = apex['policy_by_sensor'][base.bundle['sensor_family']]['correction_fraction']
            for c in candidates: c['corrected_raw_m'] = c['height_raw_m']+fraction*c['correction_m']
            models = fit_candidate_models(candidates, raw, pred, base, piecewise=True)
            old_peak = fallback.measure(depth_m=raw, rgb_bgr=rgb, visual_mode='auto')['highest_height_m'] if split == 'validation' else None
            chosen, _ = select_v5(candidates, apex['policy_by_sensor'][base.bundle['sensor_family']])
            fast_v5_peak = old_peak if chosen is None else max(liquid, chosen['height_m'])
            # Targets are training/validation supervision, never inference features.
            labels = np.load(folder/'labels_offline.npz'); gt = labels['semantic'] == 3
            _, cc = cv2.connectedComponents(gt.astype(np.uint8))
            true_heights = world_points(labels['float_depth_m'], item['true'])[..., 2]-item['true'].bottom_z_m
            tops = {i: float(true_heights[cc == i].max()) for i in range(1, int(cc.max())+1)}
            origins = np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
            for m in models:
                mask = m['mask']; values, counts = np.unique(cc[mask], return_counts=True)
                matched, overlap = max([(int(v), int(n)) for v, n in zip(values, counts, strict=True) if v], key=lambda v: v[1], default=(0, 0))
                physical = float((origins[mask] == 2).mean())
                target = tops.get(matched, 0.); metric_good = matched > 0 and abs(m['height_m']-target) <= max(.05, .05*abs(target))
                m['positive'] = int(overlap/max(m['pixels'], 1) >= .55 and physical >= .55 and metric_good)
                m['target_error_m'] = abs(m['height_m']-target) if matched else None
                if split == 'train':
                    for key in ['mask', 'indices']: m.pop(key, None)
            rows.append({'case': case, 'repeat': int(frame['repeat']), 'sensor': base.bundle['sensor_family'],
                'truth': item['meta']['truth']['highest_height_m'], 'has_float': bool(gt.any()), 'liquid': liquid,
                'fallback': old_peak, 'fast_v5_peak': fast_v5_peak, 'models': models})
        if slot % 10 == 0: print('V7_CAPTURE', split, slot, sum(len(r['models']) for r in rows), flush=True)
    joblib.dump(rows, cache, compress=3); return rows


def assess(rows, policy):
    errors = []; passed = 0; dry_used = 0; used = 0
    for row in rows:
        chosen, _ = select_models(row['models'], policy)
        peak = row['fallback'] if chosen is None else max(row['liquid'], chosen['height_m'])
        used += int(chosen is not None)
        if row['has_float']:
            e = abs(peak-row['truth']); errors.append(e); passed += int(e <= max(.05, .05*row['truth']))
        else: dry_used += int(chosen is not None)
    e = np.asarray(errors)
    return {'policy': policy, 'mae_m': float(e.mean()), 'p95_m': float(np.percentile(e, 95)), 'max_m': float(e.max()),
        'pass_pct': 100*passed/len(e), 'used': used, 'dry_used': dry_used,
        'objective': float(e.mean()+.35*np.percentile(e, 95)+.03*e.max()+.10*dry_used/max(sum(not r['has_float'] for r in rows), 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4); cv2.setNumThreads(1)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    previous = joblib.load(OLD/'region-model/verifier.joblib'); apex = joblib.load(APEX/'apex-verifier.joblib')
    train = capture('train', predictor, previous, apex); val = capture('validation', predictor, previous, apex)
    examples = [m for r in train for m in r['models']]; x = np.stack([m['quality_features'] for m in examples]); y = np.asarray([m['positive'] for m in examples])
    assert x.shape[1] == 58 and len(np.unique(y)) == 2
    classifier = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, min_samples_leaf=24, l2_regularization=4., random_state=72711)
    classifier.fit(x, y, sample_weight=compute_sample_weight('balanced', y))
    for row in val:
        if row['models']:
            for m, p in zip(row['models'], classifier.predict_proba(np.stack([m['quality_features'] for m in row['models']]))[:, 1], strict=True): m['quality_score'] = float(p)
    selected = {}; guards = {}; choices = {}
    for sensor in ['tof', 'active_stereo', 'structured_light']:
        rows = [r for r in val if r['sensor'] == sensor]
        reference = assess(rows, {'threshold': 1.01, 'max_cv': 3., 'min_cells': 4, 'interval_factor': 1., 'mode': 'guarded_raw'})
        choices[sensor] = []
        for mode in ['guarded_raw', 'surface', 'hybrid']:
            for threshold in [.5, .7, .85, .95, .98]:
                for max_cv in [2., 3.]:
                    for cells in [3, 4]:
                        for interval in [1., 2.]:
                            result = assess(rows, {'threshold': threshold, 'max_cv': max_cv, 'min_cells': cells, 'interval_factor': interval, 'mode': mode})
                            result['tail_guard'] = result['max_m'] <= reference['max_m']+.05 and result['p95_m'] <= reference['p95_m']+.01
                            choices[sensor].append(result)
        eligible = [reference]+[c for c in choices[sensor] if c['tail_guard']]
        selected[sensor] = min(eligible, key=lambda c: c['objective'])
        guards[sensor] = min([c for c in eligible if c['policy']['mode'] == 'guarded_raw'], key=lambda c: c['objective'])
        print('V7_SELECTED', sensor, json.dumps(selected[sensor]), 'GUARDS', json.dumps(guards[sensor]), flush=True)
    joblib.dump({'classifier': classifier, 'policy_by_sensor': {k: v['policy'] for k, v in selected.items()},
        'guard_policy_by_sensor': {k: v['policy'] for k, v in guards.items()}, 'feature_count': 58,
        'train_models': len(y), 'train_positives': int(y.sum()), 'test_seen': False}, OUT/'model.joblib', compress=3)
    (OUT/'validation-selection.json').write_text(json.dumps({'selected': selected, 'guards': guards, 'choices': choices,
        'train_models': len(y), 'train_positives': int(y.sum()), 'train_seed': SPECS['train'][0], 'validation_seed': SPECS['validation'][0], 'test_used': False}, indent=2))
    print('V7_FIT_COMPLETE', flush=True)


if __name__ == '__main__': main()
