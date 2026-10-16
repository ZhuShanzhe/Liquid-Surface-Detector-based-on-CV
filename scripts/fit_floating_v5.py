"""Independent region fitting and validation-only shrinkage selection. Never reads final test."""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from liquid_depth.float_evidence_v3 import (
    MultiScaleEvidencePredictor,
    choose_peak,
    region_candidates,
    score_regions,
)
from liquid_depth.floating_apex_v5 import candidates_v5, observed_peak_roi, score_v5, select_v5
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter

BASE = Path('/root/autodl-tmp')
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
OUT = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
SPECS = {'train': (202610160317, 198), 'validation': (202610160419, 99)}


def capture(split, predictor, previous):
    cache = OUT/f'{split}-candidate-cache.joblib'
    if cache.exists(): return joblib.load(cache)
    seed, count = SPECS[split]
    root = BASE/f'liquid-depth-data/synthetic/floating-apex-v5-{split}-20261016'
    rows = []
    for slot in range(count):
        case = 3+6*slot; folder = root/f'{case:05d}'
        item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        frozen = FrozenSymmetricGeometryMeter(base); geometric = UnevenSurfaceMeter(base, item['dry'])
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); liquid = frozen.measure(depth_m=raw)['liquid_depth_m']
            roi = observed_peak_roi(base, raw)
            prediction = predictor.predict(__import__('liquid_depth.rgbd_evidence_model', fromlist=['observed_features']).observed_features(
                rgb, raw, item['dry'], roi, item['bundle']))
            candidates = candidates_v5(base, item['dry'], rgb, raw, prediction, liquid, roi)
            geometric_peak = geometric.measure(raw)['highest_visible_height_m']
            if split == 'validation':
                from liquid_depth.rgbd_evidence_model import observed_features
                original = predictor.predict(observed_features(rgb, raw, item['dry'], base.ordinary_container_roi, item['bundle']))
                old_candidates = region_candidates(base, item['dry'], rgb, raw, original, liquid)
                chosen, _ = choose_peak(score_regions(old_candidates, previous), previous['policy'])
                old_peak = geometric_peak if chosen is None else max(liquid, chosen['height_m'])
            else: old_peak = None
            # Labels are accessed AFTER the proposals and observed baseline are fixed.
            gt = np.load(folder/'labels_offline.npz')['semantic'] == 3
            _, labels = cv2.connectedComponents(gt.astype(np.uint8))
            for c in candidates:
                values, counts = np.unique(labels[c['mask']], return_counts=True)
                overlap = max([int(n) for v, n in zip(values, counts, strict=True) if v > 0], default=0)
                c['positive'] = int(overlap/max(c['pixels'], 1) >= .55)
                if split == 'train': del c['mask']
            rows.append({'case': case, 'repeat': int(frame['repeat']), 'sensor': item['bundle']['sensor_family'],
                'truth': item['meta']['truth']['highest_height_m'], 'has_float': bool(gt.any()),
                'liquid': liquid, 'geometric': geometric_peak, 'old_peak': old_peak, 'candidates': candidates})
        if slot % 10 == 0: print('CAPTURE', split, slot, 'candidates', sum(len(r['candidates']) for r in rows), flush=True)
    joblib.dump(rows, cache, compress=3)
    return rows


def policy_stats(rows, policy):
    errors = []; wet_errors = []; false_count = 0; used = 0
    for row in rows:
        chosen, _ = select_v5(row['scored'], policy)
        peak = row['geometric'] if chosen is None else max(row['liquid'], chosen['height_m'])
        error = abs(peak-row['truth']); errors.append(error)
        if row['has_float']: wet_errors.append(error)
        else: false_count += int(chosen is not None)
        used += int(chosen is not None)
    e = np.asarray(wet_errors or errors)
    objective = float(e.mean()+.35*np.percentile(e, 95)+.1*false_count/max(sum(not r['has_float'] for r in rows), 1))
    return {'policy': policy, 'objective': objective, 'wet_mae_m': float(e.mean()), 'wet_p95_m': float(np.percentile(e, 95)),
        'false_dry_frames': false_count, 'used_frames': used}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--capture-only', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    previous = joblib.load(OLD/'region-model/verifier.joblib')
    train = capture('train', predictor, previous)
    validation = capture('validation', predictor, previous)
    if args.capture_only: return
    examples = [c for r in train for c in r['candidates']]
    x = np.stack([c['features'] for c in examples]); y = np.asarray([c['positive'] for c in examples])
    assert x.shape[1] == 43
    classifier = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, min_samples_leaf=12,
        l2_regularization=2., random_state=50317)
    classifier.fit(x, y, sample_weight=compute_sample_weight('balanced', y))
    for row in validation: row['scored'] = score_v5(row['candidates'], classifier, previous)
    choices = {}; selected = {}
    for sensor in ['tof', 'active_stereo', 'structured_light']:
        rows = [r for r in validation if r['sensor'] == sensor]; choices[sensor] = []
        for fraction in [0., .25, .5, .75, 1.]:
            for threshold in [.55, .7, .85, .95]:
                for factor in [2., 4.]:
                    for q in [.5, 1.]:
                        policy = {'correction_fraction': fraction, 'threshold': threshold, 'interval_factor': factor,
                            'consensus_quantile': q}
                        choices[sensor].append(policy_stats(rows, policy))
        selected[sensor] = min(choices[sensor], key=lambda v: v['objective'])
        old_e = np.asarray([abs(r['old_peak']-r['truth']) for r in rows if r['has_float']])
        selected[sensor]['old_v4_wet_mae_m'] = float(old_e.mean())
        selected[sensor]['old_v4_wet_p95_m'] = float(np.percentile(old_e, 95))
        print('SELECTED', sensor, json.dumps(selected[sensor]), flush=True)
    bundle = {'classifier': classifier, 'policy_by_sensor': {k: v['policy'] for k, v in selected.items()},
        'feature_count': 43, 'training_candidates': len(y), 'training_positive_candidates': int(y.sum()),
        'training_seed': SPECS['train'][0], 'validation_seed': SPECS['validation'][0], 'test_accessed': False}
    joblib.dump(bundle, OUT/'apex-verifier.joblib', compress=3)
    (OUT/'validation-selection.json').write_text(json.dumps({'selected': selected, 'choices': choices,
        'test_used': False, 'train_installations': 198, 'validation_installations': 99}, indent=2))
    print('FIT_COMPLETE', len(y), int(y.sum()), flush=True)


if __name__ == '__main__': main()
