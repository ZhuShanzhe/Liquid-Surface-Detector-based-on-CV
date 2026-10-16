"""Train component height correction, OOF metric risk, and highest-instance gap."""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_apex_v5 import observed_peak_roi
from liquid_depth.floating_rank_v8 import (
    candidate_features,
    fast_v5_peak,
    proposals,
    score_candidates,
    select_ranked,
)
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter

BASE = Path('/root/autodl-tmp')
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'


def capture(split, predictor, previous, apex):
    path = OUT/f'{split}-candidates.joblib'
    if path.exists():
        return joblib.load(path)
    seed, count = {'train': (202610160317, 198), 'validation': (202610160419, 99)}[split]
    root = BASE/f'liquid-depth-data/synthetic/floating-apex-v5-{split}-20261016'
    rows = []
    for slot in range(count):
        case = 3+6*slot
        folder = root/f'{case:05d}'
        item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        frozen = FrozenSymmetricGeometryMeter(base)
        fallback = OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib')
        geometry = UnevenSurfaceMeter(base, item['dry'])
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path'])
            liquid = frozen.measure(depth_m=raw)['liquid_depth_m']
            roi = observed_peak_roi(base, raw)
            pred = predictor.predict(observed_features(rgb, raw, item['dry'], roi, base.bundle))
            original, candidates = proposals(base, item['dry'], rgb, raw, pred, liquid, roi, previous, apex)
            x = candidate_features(candidates, base, item['dry'], raw, pred)
            v4 = fallback.measure(depth_m=raw, rgb_bgr=rgb, visual_mode='auto')['highest_height_m'] if split == 'validation' else None
            v5 = fast_v5_peak(original, apex, base.bundle['sensor_family'], liquid,
                lambda geometry=geometry, raw=raw: geometry.measure(raw)['highest_visible_height_m']) if split == 'validation' else None
            # Offline targets are attached only after observations and features are fixed.
            labels = np.load(folder/'labels_offline.npz')
            gt = labels['semantic'] == 3
            _, cc = cv2.connectedComponents(gt.astype(np.uint8))
            heights = world_points(labels['float_depth_m'], item['true'])[..., 2]-item['true'].bottom_z_m
            tops = {i: float(heights[cc == i].max()) for i in range(1, int(cc.max())+1)}
            global_top = max(tops.values(), default=0.)
            origin = np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
            for c, features in zip(candidates, x, strict=True):
                mask = c['mask']
                values, counts = np.unique(cc[mask], return_counts=True)
                matched, overlap = max([(int(i), int(n)) for i, n in zip(values, counts, strict=True) if i], key=lambda p: p[1], default=(0, 0))
                c['physical_target'] = int(overlap/max(c['pixels'], 1) >= .6 and (origin[mask] == 2).mean() >= .6)
                c['component_top'] = tops.get(matched, 0.)
                c['top_gap'] = global_top-c['component_top']
                c['rank_features'] = features
                c.pop('mask')
            rows.append({'case': case, 'repeat': int(frame['repeat']), 'sensor': base.bundle['sensor_family'],
                'candidates': candidates, 'liquid': liquid, 'v4': v4, 'v5': v5,
                'truth': item['meta']['truth']['highest_height_m'], 'has_float': bool(gt.any())})
        if slot % 10 == 0:
            print('V8_CAPTURE', split, slot, sum(len(r['candidates']) for r in rows), flush=True)
    joblib.dump(rows, path, compress=3)
    return rows


def regressor(loss='absolute_error', quantile=None):
    return HistGradientBoostingRegressor(loss=loss, quantile=quantile, max_iter=180, max_leaf_nodes=15,
        min_samples_leaf=24, l2_regularization=6., early_stopping=False, random_state=83811)


def assess(rows, policy):
    errors = []
    passes = 0
    used = 0
    dry_used = 0
    for r in rows:
        chosen = None if policy.get('bypass') else select_ranked(r['scored'], policy)[0]
        peak = r[policy['fallback']] if chosen is None else max(r['liquid'], chosen['height_m'])
        used += int(chosen is not None)
        if r['has_float']:
            e = abs(peak-r['truth'])
            errors.append(e)
            passes += int(e <= max(.05, .05*r['truth']))
        else:
            dry_used += int(chosen is not None)
    e = np.asarray(errors)
    return {'policy': policy, 'mae_mm': float(e.mean()*1000), 'p95_mm': float(np.percentile(e, 95)*1000),
        'max_mm': float(e.max()*1000), 'pass_pct': 100*passes/len(e), 'used': used, 'dry_used': dry_used,
        'objective': float(e.mean()+.25*np.percentile(e, 95)+.05*e.max()+.05*dry_used/max(len(rows)-len(e), 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    previous = joblib.load(OLD/'region-model/verifier.joblib')
    apex = joblib.load(APEX/'apex-verifier.joblib')
    train = capture('train', predictor, previous, apex)
    val = capture('validation', predictor, previous, apex)
    examples = [(r, c) for r in train for c in r['candidates']]
    x = np.stack([c['rank_features'] for _, c in examples])
    y = np.asarray([c['physical_target'] for _, c in examples])
    positive = y.astype(bool)
    scale = np.asarray([max(.01, c['noise_m']) for _, c in examples])
    raw_h = np.asarray([c['height_raw_m'] for _, c in examples])
    target_h = np.asarray([c['component_top'] for _, c in examples])
    correction_y = (target_h-raw_h)/scale
    groups = np.asarray([r['case'] for r, _ in examples])
    cases = np.unique(groups)
    np.random.default_rng(83811).shuffle(cases)
    fold_lookup = {int(case): i % 3 for i, case in enumerate(cases)}
    folds = np.asarray([fold_lookup[int(g)] for g in groups])
    oof = np.zeros(len(y))
    for fold in range(3):
        use = positive & (folds != fold)
        model = regressor().fit(x[use], correction_y[use])
        oof[folds == fold] = model.predict(x[folds == fold])
        print('V8_OOF', fold, int(use.sum()), flush=True)
    correction = regressor().fit(x[positive], correction_y[positive])
    physical = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=15, min_samples_leaf=24,
        l2_regularization=6., early_stopping=False, random_state=83811)
    physical.fit(x, y, sample_weight=compute_sample_weight('balanced', y))
    oof_h = raw_h+oof*scale
    risk_x = np.column_stack([x, oof, oof_h])
    risk_y = abs(oof_h-target_h)/scale
    risk = regressor('quantile', .8).fit(risk_x[positive], risk_y[positive])
    gaps = np.asarray([c['top_gap'] for _, c in examples])/scale
    top_gap = regressor().fit(risk_x[positive], gaps[positive])
    bundle = {'object_classifier': physical, 'correction': correction, 'risk': risk, 'top_gap': top_gap,
        'feature_count': 67, 'risk_feature_count': 69, 'training_candidates': len(y), 'physical_positives': int(y.sum()),
        'oof_folds_by_installation': 3, 'train_seed': 202610160317, 'validation_seed': 202610160419, 'test_seen': False}
    for r in val:
        r['scored'] = score_candidates(r['candidates'], np.stack([c['rank_features'] for c in r['candidates']]) if r['candidates'] else np.empty((0, 67)), bundle)
    selected = {}
    research = {}
    records = {}
    for sensor in ['tof', 'active_stereo', 'structured_light']:
        rows = [r for r in val if r['sensor'] == sensor]
        baselines = [assess(rows, {'bypass': b, 'fallback': b}) for b in ['v4', 'v5']]
        results = []
        for threshold in [.7, .9, .98]:
            for pixels in [4, 8]:
                for risk_cap in [.08, .25]:
                    for risk_weight, gap_weight in [(0., 0.), (.5, 0.), (.5, .5), (1., 1.)]:
                        for fallback in ['v4', 'v5']:
                            p = {'physical_threshold': threshold, 'min_pixels': pixels, 'min_cells': 2,
                                'max_risk_m': risk_cap, 'risk_weight': risk_weight, 'gap_weight': gap_weight, 'fallback': fallback}
                            results.append(assess(rows, p))
        reference = min(baselines, key=lambda a: a['objective'])
        eligible = [r for r in results if r['max_mm'] <= reference['max_mm']+20 and r['p95_mm'] <= reference['p95_mm']+5]
        selected[sensor] = min(baselines+eligible, key=lambda a: a['objective'])
        research[sensor] = min(results, key=lambda a: a['objective'])
        records[sensor] = baselines+results
        print('V8_SELECTED', sensor, json.dumps(selected[sensor]), 'RESEARCH', json.dumps(research[sensor]), flush=True)
    bundle['policy_by_sensor'] = {s: r['policy'] for s, r in selected.items()}
    bundle['research_policy_by_sensor'] = {s: r['policy'] for s, r in research.items()}
    joblib.dump(bundle, OUT/'model.joblib', compress=3)
    joblib.dump(val, OUT/'validation-scored.joblib', compress=3)
    selection = {'selected': selected, 'research': research, 'all_validation_results': records,
        'training_candidates': len(y), 'physical_positives': int(y.sum()), 'test_used': False,
        'oof_installs_disjoint': True, 'train_seed': 202610160317, 'validation_seed': 202610160419}
    (OUT/'validation-selection.json').write_text(json.dumps(selection, indent=2))
    print('V8_FIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
