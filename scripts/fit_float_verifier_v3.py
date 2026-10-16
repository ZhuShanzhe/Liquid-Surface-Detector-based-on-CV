"""Region classification, noise-aware apex correction and validation-only policy."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.class_weight import compute_sample_weight

from liquid_depth.float_evidence_v3 import (
    MultiScaleEvidencePredictor,
    choose_peak,
    region_candidates,
    score_regions,
)
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter


def capture_rows(root, model, seed, output):
    cache = output/'region-cache.joblib'
    if cache.exists(): return joblib.load(cache)
    predictor = MultiScaleEvidencePredictor(model, 'cuda'); rows = []
    for case in range(594):
        folder = root/f'{case:05d}'; item = install(case, root, root/'dry', seed, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        liquid_meter = FrozenSymmetricGeometryMeter(base); peak_meter = UnevenSurfaceMeter(base, item['dry'])
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); liquid = liquid_meter.measure(depth_m=raw)['liquid_depth_m']
            prediction = predictor.predict(observed_features(rgb, raw, item['dry'], base.ordinary_container_roi, item['bundle']))
            candidates = region_candidates(base, item['dry'], rgb, raw, prediction, liquid)
            geometric = peak_meter.measure(raw)['highest_visible_height_m']
            # Offline labels appear only after all observed features/proposals are computed.
            offline = np.load(folder/'labels_offline.npz'); gt = offline['semantic'] == 3
            n, labels = cv2.connectedComponents(gt.astype(np.uint8))
            trueheight = world_points(offline['float_depth_m'], item['true'])[..., 2]-item['true'].bottom_z_m
            targets = {i: float(trueheight[labels == i].max()) for i in range(1, n)}
            for candidate in candidates:
                values, counts = np.unique(labels[candidate['mask']], return_counts=True)
                choices = [(int(v), int(c)) for v, c in zip(values, counts, strict=True) if v > 0]
                match, overlap = max(choices, key=lambda v: v[1]) if choices else (0, 0)
                positive = match > 0 and overlap/max(candidate['pixels'], 1) >= .35
                candidate['positive'] = int(positive)
                candidate['target_correction'] = (targets[match]-candidate['height_raw_m'])/candidate['noise_m'] if positive else 0.
                del candidate['mask']
            rows.append({'case': case, 'repeat': frame['repeat'], 'scene': item['meta']['scenario'],
                'truth': item['meta']['truth']['highest_height_m'], 'has_visible_float': bool(gt.any()),
                'liquid': liquid, 'geometric': geometric, 'candidates': candidates})
        if case % 30 == 0: print('REGIONS', case, sum(len(r['candidates']) for r in rows), flush=True)
    joblib.dump(rows, cache, compress=3)
    return rows


def evaluate_policy(rows, policy):
    groups = defaultdict(list); false_frames = 0; negative_frames = 0; used = 0
    for row in rows:
        chosen, _ = choose_peak(row['scored'], policy)
        peak = row['geometric'] if chosen is None else max(row['liquid'], chosen['height_m'])
        groups[row['scene']].append(abs(peak-row['truth']))
        used += chosen is not None
        if not row['has_visible_float']:
            negative_frames += 1; false_frames += chosen is not None
    stats = {k: {'mae': float(np.mean(v)), 'p95': float(np.percentile(v, 95))} for k, v in groups.items()}
    objective = np.mean([v['mae']+.35*v['p95'] for v in stats.values()])+.10*false_frames/max(negative_frames, 1)
    return {'policy': policy, 'scene_metrics': stats, 'false_float_frames': false_frames,
            'negative_frames': negative_frames, 'used_frames': used, 'objective': float(objective)}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(exist_ok=True, parents=True); torch.set_num_threads(4); cv2.setNumThreads(1)
    rows = capture_rows(a.root, a.model, 202610155361, a.output)
    train = [c for row in rows if row['case'] < 396 for c in row['candidates']]
    x = np.stack([c['features'] for c in train]); y = np.asarray([c['positive'] for c in train])
    positive = y == 1; targets = np.asarray([c['target_correction'] for c in train])[positive]
    classifier = HistGradientBoostingClassifier(max_iter=180, max_leaf_nodes=15, min_samples_leaf=12, l2_regularization=2., random_state=361)
    classifier.fit(x, y, sample_weight=compute_sample_weight('balanced', y))
    models = {'classifier': classifier}
    for key, quantile in [('regressor', .5), ('lower', .1), ('upper', .9)]:
        model = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, max_iter=140,
            max_leaf_nodes=12, min_samples_leaf=12, l2_regularization=2., random_state=361)
        model.fit(x[positive], targets); models[key] = model
    validation = [row for row in rows if row['case'] >= 396]
    for row in validation: row['scored'] = score_regions(row['candidates'], models)
    reference = evaluate_policy(validation, {'threshold': 1.01, 'interval_factor': 1.})
    choices = []
    for factor in [1., 2., 4., 8.]:
        for threshold in [.5, .7, .85, .9, .95, .98, .995]:
            choice = evaluate_policy(validation, {'threshold': threshold, 'interval_factor': factor})
            choice['tail_guard_passed'] = all(choice['scene_metrics'][key]['p95'] <=
                reference['scene_metrics'][key]['p95']+.010 for key in ['ordinary', 'glare', 'floating_objects'])
            choices.append(choice)
    eligible = [v for v in choices if v['tail_guard_passed'] and v['used_frames'] >= 8]
    selected = min(eligible+[reference], key=lambda v: v['objective'])
    models['policy'] = selected['policy']; models['feature_count'] = int(x.shape[1])
    models['train_candidates'] = len(train); models['train_positive_candidates'] = int(positive.sum())
    joblib.dump(models, a.output/'verifier.joblib', compress=3)
    (a.output/'validation-policy.json').write_text(json.dumps({'selected': selected, 'reference': reference,
        'choices': choices, 'test_used': False, 'train_candidates': len(train),
        'train_positive_candidates': int(positive.sum()), 'training_installations': 396, 'validation_installations': 198}, indent=2))
    print('VERIFIER_COMPLETE', json.dumps(selected), flush=True)


if __name__ == '__main__': main()
