"""Fit return-origin reliability from existing training split; choose policy on validation only."""
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
from liquid_depth.floating_apex_v5 import candidates_v5, observed_peak_roi, score_v5
from liquid_depth.floating_surface_v6 import reconstruct_candidates, select_surface
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-surface-v6-20261016'
SPECS = {'train': (202610160317, 198), 'validation': (202610160419, 99)}


def capture(split, predictor, previous, apex):
    path = OUT/f'{split}-surface-cache.joblib'
    if path.exists(): return joblib.load(path)
    seed, count = SPECS[split]; root = BASE/f'liquid-depth-data/synthetic/floating-apex-v5-{split}-20261016'
    rows = []
    for slot in range(count):
        case = 3+6*slot; folder = root/f'{case:05d}'
        item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        frozen = FrozenSymmetricGeometryMeter(base); rgb = cv2.imread(str(folder/'rgb.png'))
        fallback = OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib')
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); liquid = frozen.measure(depth_m=raw)['liquid_depth_m']
            roi = observed_peak_roi(base, raw); prediction = predictor.predict(observed_features(rgb, raw, item['dry'], roi, base.bundle))
            candidates = candidates_v5(base, item['dry'], rgb, raw, prediction, liquid, roi)
            scored = score_v5(candidates, apex['classifier'], previous)
            reconstructed = reconstruct_candidates(scored, raw, prediction, base)
            old_peak = fallback.measure(depth_m=raw, rgb_bgr=rgb, visual_mode='auto')['highest_height_m'] if split == 'validation' else None
            # Semantic and physical return origin labels are training-only, accessed after proposals/fits.
            semantic = np.load(folder/'labels_offline.npz')['semantic']; gt = semantic == 3
            origin = np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
            for c in reconstructed:
                m = c['mask']; semantic_purity = float(gt[m].mean()); true_return_fraction = float((origin[m] == 2).mean())
                c['positive'] = int(semantic_purity >= .55 and true_return_fraction >= .55)
                c['label_semantic_purity'] = semantic_purity; c['label_true_return_fraction'] = true_return_fraction
                if split == 'train': del c['mask']
            rows.append({'case': case, 'repeat': int(frame['repeat']), 'sensor': item['meta']['sensor_family'],
                'truth': item['meta']['truth']['highest_height_m'], 'has_float': bool(gt.any()),
                'liquid': liquid, 'old_peak': old_peak, 'candidates': reconstructed})
        if slot % 10 == 0: print('SURFACE_CAPTURE', split, slot, sum(len(r['candidates']) for r in rows), flush=True)
    joblib.dump(rows, path, compress=3); return rows


def stats(rows, policy, classifier):
    errors = []; false_dry = 0; used = 0; passing = 0
    for row in rows:
        chosen, _ = select_surface(row['candidates'], policy, classifier)
        peak = row['old_peak'] if chosen is None else max(row['liquid'], chosen['height_m'])
        if row['has_float']:
            error = abs(peak-row['truth']); errors.append(error); passing += int(error <= max(.05, .05*row['truth']))
        else: false_dry += int(chosen is not None)
        used += int(chosen is not None)
    e = np.asarray(errors); p95 = float(np.percentile(e, 95)); worst = float(e.max())
    return {'policy': policy, 'mae_m': float(e.mean()), 'p95_m': p95, 'max_m': worst,
        'pass_pct': 100*passing/len(e), 'false_dry': false_dry, 'used': used,
        'objective': float(e.mean()+.35*p95+.10*false_dry/max(sum(not r['has_float'] for r in rows), 1))}


def main():
    OUT.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4); cv2.setNumThreads(1)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    previous = joblib.load(OLD/'region-model/verifier.joblib'); apex = joblib.load(APEX/'apex-verifier.joblib')
    training = capture('train', predictor, previous, apex); validation = capture('validation', predictor, previous, apex)
    examples = [c for r in training for c in r['candidates']]
    x = np.stack([c['surface_features'] for c in examples]); y = np.asarray([c['positive'] for c in examples])
    assert x.shape[1] == 55 and len(np.unique(y)) == 2
    classifier = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=12, min_samples_leaf=20,
        l2_regularization=3., random_state=61337)
    classifier.fit(x, y, sample_weight=compute_sample_weight('balanced', y))
    selected = {}; choices = {}
    for sensor in ['tof', 'active_stereo', 'structured_light']:
        rows = [r for r in validation if r['sensor'] == sensor]
        reference = stats(rows, {'threshold': 1.01, 'interval_factor': 1.}, classifier); choices[sensor] = []
        # A bounded maximum-error guard is chosen BEFORE fresh test access.
        for threshold in [.5, .7, .85, .95, .98]:
            for factor in [1., 2., 4., 8.]:
                result = stats(rows, {'threshold': threshold, 'interval_factor': factor}, classifier)
                result['guard_pass'] = result['p95_m'] <= reference['p95_m']+.01 and result['max_m'] <= reference['max_m']+.05
                choices[sensor].append(result)
        selected[sensor] = min([reference]+[c for c in choices[sensor] if c['guard_pass']], key=lambda c: c['objective'])
        print('SURFACE_SELECTED', sensor, json.dumps(selected[sensor]), flush=True)
    joblib.dump({'classifier': classifier, 'policy_by_sensor': {k: v['policy'] for k, v in selected.items()},
        'feature_count': 55, 'train_candidates': len(y), 'train_positives': int(y.sum()),
        'positive_definition': 'semantic_float_fraction>=.55 AND true_float_return_origin_fraction>=.55',
        'test_seen': False}, OUT/'surface-reliability.joblib', compress=3)
    (OUT/'validation-selection.json').write_text(json.dumps({'selected': selected, 'choices': choices,
        'train_seed': SPECS['train'][0], 'validation_seed': SPECS['validation'][0], 'test_used': False,
        'positive_candidates': int(y.sum()), 'training_candidates': len(y)}, indent=2))
    print('SURFACE_FIT_COMPLETE', flush=True)


if __name__ == '__main__': main()
