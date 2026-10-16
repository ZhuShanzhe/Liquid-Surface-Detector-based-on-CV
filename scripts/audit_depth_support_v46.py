"""Offline point-survival audit for <=50% controlled surface-depth loss.

Simulator truth is used ONLY to synthesize the controlled corruption, identify
true surface support for diagnosis, and score. The deployment route receives
RGB, raw depth, model mask, and estimated empty-vessel geometry only.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from evaluate_liquid_height_v21 import Predictor
from evaluate_other_scenes_v34 import predicted
from evaluate_shared_empty_v45 import install
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics, world_points
from liquid_depth.ordinary_surface_v35 import interior_ray_roi
from liquid_depth.scene_route_policy_v45 import surface_candidate
from liquid_depth.surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel, support


LEVELS = (0, 10, 25, 50)
NOISE = {'active_stereo': (.0006, .0011),
         'structured_light': (.0008, .0015),
         'tof': (.0015, .0015)}


def finite_valid(depth):
    return np.isfinite(depth) & (depth > 0)


def controlled_raw(base, target, visible, sensor, loss_pct, seed):
    """Fill synthetic target returns, then remove exactly loss_pct of pixels.

    This is an optimistic point-availability diagnostic, NOT a physical sensor
    model or an independent accuracy benchmark.
    """
    rng = np.random.default_rng(seed)
    output = np.array(base, copy=True)
    sigma0, sigma2 = NOISE[sensor]
    valid_target = visible & finite_valid(target)
    sigma = sigma0 + sigma2 * target[valid_target] ** 2
    output[valid_target] = np.maximum(1e-4,
        target[valid_target] + rng.normal(size=int(valid_target.sum())) * sigma)
    yy, xx = np.nonzero(valid_target)
    if loss_pct:
        # A contiguous stripe plus repeat-specific phase mimics spatial holes.
        phase = (xx + (seed % output.shape[1])) % output.shape[1]
        order = np.lexsort((yy, phase))
        chosen = order[:round(len(order) * loss_pct / 100)]
        output[yy[chosen], xx[chosen]] = 0
    return output


def route_output(bundle, rgb, raw, mask):
    result = surface_candidate(bundle, rgb, raw, mask, scene='depth_failure')
    value = result['numeric_candidate_m']
    return result, None if value is None or not np.isfinite(value) else float(value)


def counterfactual(bundle, rgb, raw, mask):
    cal = height_calibration(bundle)
    noise = None if bundle['sensor_family'] == 'tof' else StereoNoiseModel.simulation_proxy(bundle['sensor_family'])
    result = RefinedSurfaceEstimator(mode='sensor', stereo_noise=noise).estimate(
        rgb, raw, {'mask': mask}, cal.intrinsics, cal.camera_to_world_cv,
        cal.bottom_z_m, cal.area_grid(), cal.radii_m)
    return result['level_m'] is not None, result['quality_flags']


def point_chain(bundle, raw, predicted_mask, oracle_visible):
    cal = height_calibration(bundle)
    valid = finite_valid(raw)
    rim = float(bundle['rim_z_m'])
    opening = replace(cal, bottom_z_m=rim - 1e-5)
    roi = interior_ray_roi(opening, rim, raw.shape, .90)
    xyz = world_points(raw, cal)
    normalized = xyz[..., :2] / np.asarray(cal.radii_m)
    inside = (abs(normalized) ** cal.footprint_exponent).sum(axis=-1) <= 1.05 ** cal.footprint_exponent
    margin = max(.03, .02 * rim)
    plausible = (xyz[..., 2] >= cal.bottom_z_m - margin) & (xyz[..., 2] <= rim + margin)
    container = valid & roi & inside & plausible
    mask = np.asarray(predicted_mask, bool)
    pre_geometry = valid & roi & mask
    selected = container & mask
    eroded = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & valid
    data, support_reason = support(raw, selected, cal.intrinsics, cal.camera_to_world_cv, cal.radii_m)
    sampled = 0 if data is None else len(data[0])
    # The oracle-visible mask is evaluator-only: never used in the route above.
    oracle = np.asarray(oracle_visible, bool)
    return {
        'image_pixels': int(raw.size), 'full_valid': int(valid.sum()),
        'opening_pixels': int(roi.sum()), 'opening_valid': int((valid & roi).sum()),
        'container_valid': int(container.sum()),
        'predicted_mask_pixels': int(mask.sum()),
        'pre_geometry_mask_valid': int(pre_geometry.sum()),
        'container_mask_valid': int(selected.sum()),
        'eroded_fit_valid': int(eroded.sum()), 'sampled_fit_points': int(sampled),
        'support_reason': support_reason,
        'oracle_surface_pixels': int(oracle.sum()),
        'oracle_surface_valid': int((oracle & valid).sum()),
        'oracle_predicted_overlap': int((oracle & mask).sum()),
        'oracle_valid_predicted_overlap': int((oracle & valid & mask).sum()),
        'oracle_valid_in_container': int((oracle & container).sum()),
        'mask_roi_pixels': int((mask & roi).sum()),
    }


def diagnosis(bundle, rgb, raw, predicted_mask, oracle_mask, chain, route_reason):
    cal = height_calibration(bundle)
    rim = float(bundle['rim_z_m'])
    opening = replace(cal, bottom_z_m=rim - 1e-5)
    roi = interior_ray_roi(opening, rim, raw.shape, .90)
    no_geometry_ok, no_geometry_reason = counterfactual(bundle, rgb, raw, predicted_mask)
    roi_only_ok, roi_only_reason = counterfactual(bundle, rgb, raw, predicted_mask & roi)
    oracle_ok, oracle_reason = counterfactual(bundle, rgb, raw, oracle_mask)
    if roi_only_ok:
        label = 'pointwise_geometry_filter_or_gate'
    elif no_geometry_ok:
        label = 'opening_roi_or_local_gate'
    elif oracle_ok:
        label = 'model_mask_or_mask_erosion'
    elif chain['oracle_surface_valid'] < 24:
        label = 'true_surface_evidence_under_24'
    else:
        label = 'sensor_likelihood_or_sparse_spatial_support'
    return {'category': label, 'no_geometry_numeric': no_geometry_ok,
            'roi_only_numeric': roi_only_ok, 'oracle_mask_numeric': oracle_ok,
            'no_geometry_reasons': no_geometry_reason,
            'roi_only_reasons': roi_only_reason,
            'oracle_mask_reasons': oracle_reason, 'route_reasons': route_reason}


def summarize(rows):
    n = len(rows)
    def pct(value, total):
        return float(100 * value / total) if total else None
    def mean_frac(num, den):
        return float(np.mean([r['counts'][num] / r['counts'][den]
            for r in rows if r['counts'][den] > 0])) if rows else None
    outputs = [r for r in rows if r['prediction_m'] is not None]
    errors = np.asarray([abs(r['prediction_m'] - r['true_height_m']) for r in outputs])
    passes = sum(abs(r['prediction_m'] - r['true_height_m']) <=
                 max(.005, .02 * r['true_height_m']) for r in outputs)
    return {
        'frames': n, 'numeric_frames': len(outputs),
        'numeric_coverage_pct': pct(len(outputs), n),
        'all_frame_pass_pct': pct(passes, n),
        'conditional_mae_mm': float(1000 * errors.mean()) if len(errors) else None,
        'full_valid_of_image_mean_pct': pct(mean_frac('full_valid', 'image_pixels'), 1),
        'opening_valid_of_opening_mean_pct': pct(mean_frac('opening_valid', 'opening_pixels'), 1),
        'container_valid_of_opening_mean_pct': pct(mean_frac('container_valid', 'opening_pixels'), 1),
        'mask_valid_of_container_mean_pct': pct(mean_frac('container_mask_valid', 'container_valid'), 1),
        'fit_valid_of_mask_valid_mean_pct': pct(mean_frac('eroded_fit_valid', 'container_mask_valid'), 1),
        'oracle_surface_valid_mean_pct': pct(mean_frac('oracle_surface_valid', 'oracle_surface_pixels'), 1),
        'mask_recall_oracle_mean_pct': pct(mean_frac('oracle_predicted_overlap', 'oracle_surface_pixels'), 1),
        'geometry_retention_mask_valid_mean_pct': pct(mean_frac('container_mask_valid', 'pre_geometry_mask_valid'), 1),
        'no_output_reasons': dict(Counter('|'.join(r['route_reasons']) for r in rows if r['prediction_m'] is None)),
        'diagnoses': dict(Counter(r['diagnosis']['category'] for r in rows if r['diagnosis'] is not None)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wet-root', type=Path, required=True)
    parser.add_argument('--dry-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=202610122159)
    parser.add_argument('--cases', type=int, default=99)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = Predictor('v8')
    records = []
    for case in range(args.cases):
        item = install(case, args.wet_root, args.dry_root, args.seed)
        folder = args.wet_root / f'{case * 6 + 5:05d}'
        meta = json.loads((folder / 'metadata.json').read_text())
        if meta['scenario'] != 'depth_failure':
            raise ValueError('unexpected_scene_slot')
        rgb = cv2.imread(str(folder / 'rgb.png'))
        with np.load(folder / 'geometry.npz') as geom:
            visible = geom['mask'].astype(bool)
            target = geom['target'].astype(np.float32)
        cal = replace(HeightCalibration.from_simulation(meta),
                      bottom_z_m=float(meta['interior_bottom_z_m']))
        truth = reference_statistics(meta, cal)['mean_m']
        frames = [f for f in json.loads((folder / 'frames.json').read_text())
                  if f['variant'] == 'base' and f['repeat'] < args.repeats]
        for frame in frames:
            base = np.load(frame['raw_path'], allow_pickle=False)
            scenarios = [('natural_base', None, base)]
            for loss in LEVELS:
                scenarios.append(('controlled_proxy', loss, controlled_raw(
                    base, target, visible, meta['sensor_family'], loss,
                    args.seed + case * 10007 + frame['repeat'] * 509)))
            predictions = predicted(model, [rgb] * len(scenarios), [s[2] for s in scenarios])
            for (regime, loss, raw), prediction in zip(scenarios, predictions):
                mask = prediction['mask']
                route, value = route_output(item['bundle'], rgb, raw, mask)
                chain = point_chain(item['bundle'], raw, mask, visible)
                diag = None if value is not None else diagnosis(
                    item['bundle'], rgb, raw, mask, visible, chain, route['reasons'])
                records.append({
                    'case': case, 'repeat': frame['repeat'], 'sensor': meta['sensor_family'],
                    'band': meta['band'], 'regime': regime, 'additional_loss_pct': loss,
                    'true_height_m': truth, 'prediction_m': value,
                    'actual_surface_loss_pct': 100 * (1 - chain['oracle_surface_valid'] /
                        max(chain['oracle_surface_pixels'], 1)),
                    'counts': chain, 'route_reasons': route['reasons'], 'diagnosis': diag,
                })
        print(json.dumps({'case_done': case + 1, 'records': len(records)}), flush=True)
    with (args.output / 'records.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    groups = defaultdict(list)
    for row in records:
        key = row['regime'] + ('/' + str(row['additional_loss_pct']) if row['additional_loss_pct'] is not None else '')
        groups[key].append(row)
        groups[key + '/' + row['sensor']].append(row)
        if row['regime'] == 'natural_base':
            loss = row['actual_surface_loss_pct']
            band = '0-10' if loss <= 10 else '10-25' if loss <= 25 else '25-50' if loss <= 50 else 'over50'
            groups['natural_realized_loss/' + band].append(row)
    summary = {key: summarize(group) for key, group in sorted(groups.items())}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    sources = [Path(__file__), Path(__file__).with_name('evaluate_shared_empty_v45.py'),
               Path(__file__).parents[1] / 'src/liquid_depth/scene_route_policy_v45.py',
               Path(__file__).parents[1] / 'src/liquid_depth/empty_scene_routes_v44.py']
    (args.output / 'protocol.json').write_text(json.dumps({
        'scope': 'offline point-support diagnosis, not production qualification',
        'controlled_loss': list(LEVELS),
        'controlled_proxy': 'oracle surface depth plus parametric noise; oracle mask used only in synthesis/diagnosis',
        'natural_base': 'existing sensor-simulation raw data, not cleaned or backfilled',
        'no_zero_snap_or_height_clip': True,
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, indent=2))
    print('COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
