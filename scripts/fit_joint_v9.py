"""Train joint identity/error risk including wrong regions, then causal validation."""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight

from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_apex_v5 import observed_peak_roi
from liquid_depth.floating_joint_v9 import (
    CausalSupport,
    choose_supported,
    instance_candidates,
    score_joint,
    support_check,
)
from liquid_depth.floating_rank_v8 import candidate_features, proposals
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp')
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
V8 = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-joint-v9-validation-20261016'


def train():
    rows = joblib.load(V8/'train-candidates.joblib')
    examples = [(r, c) for r in rows for c in r['candidates']]
    x = np.stack([c['rank_features'] for _, c in examples])
    y = np.asarray([c['physical_target'] for _, c in examples])
    scales = np.asarray([max(.01, c['noise_m']) for _, c in examples])
    raw_h = np.asarray([c['height_raw_m'] for _, c in examples])
    comp_h = np.asarray([c['component_top'] for _, c in examples])
    truth = np.asarray([r['truth'] for r, _ in examples])
    groups = np.asarray([r['case'] for r, _ in examples])
    cases = np.unique(groups)
    np.random.default_rng(83811).shuffle(cases)
    fold_lookup = {int(c): i % 3 for i, c in enumerate(cases)}
    folds = np.asarray([fold_lookup[int(c)] for c in groups])
    oof = np.zeros(len(y))
    for fold in range(3):
        use = (y == 1) & (folds != fold)
        model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=24, l2_regularization=6., early_stopping=False, random_state=83811)
        model.fit(x[use], ((comp_h-raw_h)/scales)[use])
        oof[folds == fold] = model.predict(x[folds == fold])
        print('V9_OOF', fold, flush=True)
    v8 = joblib.load(V8/'model.joblib')
    h = raw_h+oof*scales
    old_risk_x = np.column_stack([x, oof, h])
    risk = np.maximum(0., v8['risk'].predict(old_risk_x))
    gap = np.maximum(0., v8['top_gap'].predict(old_risk_x))
    features = np.column_stack([x, oof, h, risk, gap]).astype(np.float32)
    weights = compute_sample_weight('balanced', y)
    small = np.asarray([c['pixels'] < 12 for _, c in examples])
    weights *= np.where((y == 0) & small, 4., 1.)
    identity = HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=15, min_samples_leaf=24,
        l2_regularization=8., early_stopping=False, random_state=94913)
    identity.fit(features, y, sample_weight=weights)
    # All candidates, including wrong regions/echoes and lower objects, have a
    # frame-highest metric error. No undefined fake component target for negatives.
    joint_y = abs(h-truth)/scales
    joint = HistGradientBoostingRegressor(loss='quantile', quantile=.9, max_iter=220, max_leaf_nodes=15,
        min_samples_leaf=24, l2_regularization=8., early_stopping=False, random_state=94913)
    joint.fit(features, joint_y)
    bundle = {'identity': identity, 'joint_risk': joint, 'feature_count': 71, 'training_candidates': len(y),
        'positive_identity': int(y.sum()), 'wrong_region_or_echo_candidates': int((y == 0).sum()),
        'small_negative_weight': 4., 'risk_target': 'absolute error to physical frame highest for ALL candidates',
        'oof_installation_folds': 3, 'train_seed': 202610160317, 'validation_seed': 202610164119, 'test_seen': False}
    joblib.dump(bundle, OUT/'heads.joblib', compress=3)
    return bundle, v8


def capture(bundle, v8):
    cache = OUT/'validation-observations.joblib'
    if cache.exists():
        return joblib.load(cache)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    previous = joblib.load(OLD/'region-model/verifier.joblib')
    apex = joblib.load(APEX/'apex-verifier.joblib')
    rows = []
    for slot in range(99):
        case = 3+6*slot
        folder = ROOT/f'{case:05d}'
        item = install(case, ROOT, ROOT/'dry', 202610164119, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        frozen = FrozenSymmetricGeometryMeter(base)
        fallback = OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib')
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path'])
            liquid = frozen.measure(depth_m=raw)['liquid_depth_m']
            roi = observed_peak_roi(base, raw)
            pred = predictor.predict(observed_features(rgb, raw, item['dry'], roi, base.bundle))
            _, candidates = proposals(base, item['dry'], rgb, raw, pred, liquid, roi, previous, apex)
            x = candidate_features(candidates, base, item['dry'], raw, pred)
            scored = score_joint(candidates, x, v8, bundle)
            for c in scored:
                c.pop('mask')
            old = fallback.measure(depth_m=raw, rgb_bgr=rgb, visual_mode='auto')['highest_height_m']
            rows.append({'case': case, 'repeat': int(frame['repeat']), 'sensor': base.bundle['sensor_family'],
                'bundle': item['bundle'], 'raw': raw, 'scored': scored, 'liquid': liquid, 'fallback': old,
                'truth': item['meta']['truth']['highest_height_m'],
                'has_float': bool((np.load(folder/'labels_offline.npz')['semantic'] == 3).any()), 'fit_cache': {}})
        if slot % 10 == 0:
            print('V9_VALIDATION_CAPTURE', slot, flush=True)
    joblib.dump(rows, cache, compress=3)
    return rows


def assess(rows, policy, *, temporal):
    memory = CausalSupport()
    errors = []
    passes = used = weak = dry_used = 0
    for row in rows:
        valid, _ = memory.begin((str(row['case']), row['repeat'], row['repeat']*.1) if temporal else None, row['raw'])
        instances = instance_candidates(row['scored'], policy)
        cal = height_calibration(row['bundle'])
        chosen, _, _ = choose_supported(instances, policy,
            lambda c, r=row, k=cal: support_check(c, r['raw'], k, r['bundle'], fit_cache=r['fit_cache']), memory, valid)
        peak = row['fallback'] if chosen is None else max(row['liquid'], chosen['height_m'])
        used += int(chosen is not None)
        weak += int(chosen is not None and chosen['support_kind'] == 'causal_small_object')
        if row['has_float']:
            e = abs(peak-row['truth'])
            errors.append(e)
            passes += int(e <= max(.05, .05*row['truth']))
        else:
            dry_used += int(chosen is not None)
    e = np.asarray(errors)
    return {'policy': policy, 'temporal': temporal, 'mae_mm': float(1000*e.mean()), 'p95_mm': float(1000*np.percentile(e, 95)),
        'max_mm': float(1000*e.max()), 'pass_pct': 100*passes/len(e), 'used': used, 'small_temporal_used': weak, 'dry_used': dry_used,
        'objective': float(e.mean()+.25*np.percentile(e, 95)+.1*e.max()+.1*dry_used/max(len(rows)-len(e), 1))}


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    bundle, v8 = train()
    rows = capture(bundle, v8)
    selected = {}
    results = {}
    for sensor in ['tof', 'active_stereo', 'structured_light']:
        subset = [r for r in rows if r['sensor'] == sensor]
        options = []
        for threshold in [.8, .95]:
            for risk_cap in [.08, .25]:
                for weight in [.5, 1.]:
                    for confirmations in [2, 3]:
                        policy = {'identity_threshold': threshold, 'risk_cap_m': risk_cap, 'risk_weight': weight, 'confirm_frames': confirmations}
                        options.append(assess(subset, policy, temporal=True))
        selected[sensor] = min(options, key=lambda r: r['objective'])
        selected[sensor]['stateless_same_policy'] = assess(subset, selected[sensor]['policy'], temporal=False)
        results[sensor] = options
        print('V9_SELECTED', sensor, json.dumps(selected[sensor]), flush=True)
    bundle['policy_by_sensor'] = {s: r['policy'] for s, r in selected.items()}
    joblib.dump(bundle, OUT/'model.joblib', compress=3)
    (OUT/'validation-selection.json').write_text(json.dumps({'selected': selected, 'all_results': results,
        'train_candidates': bundle['training_candidates'], 'wrong_region_or_echo_candidates': bundle['wrong_region_or_echo_candidates'],
        'test_used': False, 'four_frame_validation_seed': 202610164119}, indent=2))
    print('V9_FIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
