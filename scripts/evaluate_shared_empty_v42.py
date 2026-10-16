"""Held-out scene evaluation with one dry RGB-D geometry per installation.

Only simulator metadata is used to emulate noisy operator outline/distance input.
Wet inference receives neither true geometry nor target masks. Oracle geometry is
an evaluator-only comparator. All heights are relative to the interior bottom.
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
from evaluate_other_scenes_v34 import predicted, refined
from experiment_empty_geometry_v41 import annotation
from liquid_depth.empty_geometry_v41 import build_geometry, height_calibration
from liquid_depth.illumination import adaptive_exposure_correction
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics
from liquid_depth.ordinary_surface_v40 import prepare, calibrate, estimate


def install(case, wet_root, dry_root, seed):
    folder = wet_root / f'{case * 6:05d}'
    meta = json.loads((folder / 'metadata.json').read_text())
    dry = np.load(dry_root / f'{case:04d}' / 'dry_depth_m.npy', allow_pickle=False)
    original = replace(HeightCalibration.from_simulation(meta),
                       bottom_z_m=float(meta['interior_bottom_z_m']))
    pose = original.camera_to_world_cv.copy()
    pose[2, 3] -= original.bottom_z_m
    true = replace(original, camera_to_world_cv=pose, bottom_z_m=0.)
    rim = float(meta['container_rim_z_m'] - meta['interior_bottom_z_m'])
    shape = {'cylindrical': 'circle', 'elliptical': 'ellipse',
             'rectangular_tank': 'rectangle'}[meta['container_shape']]
    rng = np.random.default_rng(seed + case * 8071)
    spec = annotation(true, rim, shape, rng)
    sensor = meta['sensor_family']
    spec['sensor_family'] = sensor
    spec['baseline_m'] = {'active_stereo': .055, 'structured_light': .035,
                          'tof': 0.}[sensor]
    bottom_distance = float(true.camera_to_world_cv[2, 3])
    spec['metric_anchor'] = {'kind': 'perpendicular_distance',
                             'distance_m': float(bottom_distance + rng.normal(
                                 0, .001 + .001 * bottom_distance))}
    bundle = build_geometry(dry, true.intrinsics, spec)
    fitted = height_calibration(bundle)
    audit = {'case': case, 'shape': shape, 'sensor': sensor,
             'dry_valid_pixels': int((dry > 0).sum()),
             'bottom_distance_error_mm': 1000 * (
                 bundle['bottom_plane_camera']['perpendicular_distance_m'] - bottom_distance),
             'normal_error_deg': float(np.degrees(np.arccos(np.clip(
                 np.dot(np.asarray(bundle['bottom_plane_camera']['up']),
                        true.camera_to_world_cv[2, :3]), -1, 1)))),
             'geometry_quality': bundle['quality']}
    ordinary = None
    if sensor != 'tof':
        try:
            ctx = prepare(fitted, bundle['rim_z_m'], dry.shape, bundle['baseline_m'])
            ordinary = (ctx, calibrate(dry, ctx))
            audit['ordinary_dry_observed_pixels'] = ordinary[1].observed_pixels
        except ValueError as exc:
            audit['ordinary_reference_error'] = str(exc)
    return {'meta': meta, 'dry': dry, 'bundle': bundle, 'fitted': fitted,
            'true': true, 'ordinary': ordinary, 'audit': audit}


def summarize(rows):
    n = len(rows)
    q = [r for r in rows if r['prediction_m'] is not None]
    ae = np.array([abs(r['prediction_m'] - r['true_height_m']) for r in q])
    rel = np.array([abs(r['prediction_m'] - r['true_height_m']) /
                    r['true_height_m'] for r in q if r['true_height_m'] > 0])
    ok = sum(abs(r['prediction_m'] - r['true_height_m']) <=
             max(.005, .02 * r['true_height_m']) for r in q)
    bins = [0, .005, .01, .02, .05, .1, .2, np.inf]
    rel_bins = [0, .005, .01, .02, .03, .05, .1, np.inf]
    return {'frames': n, 'installations': len({r['case'] for r in rows}),
            'numeric_frames': len(q), 'numeric_coverage_pct': 100 * len(q) / n if n else None,
            'mae_mm': float(1000 * ae.mean()) if len(ae) else None,
            'p95_mm': float(1000 * np.percentile(ae, 95)) if len(ae) else None,
            'abs_rel_pct': float(100 * rel.mean()) if len(rel) else None,
            'numeric_pass_pct': 100 * ok / len(q) if q else None,
            'all_frame_pass_pct': 100 * ok / n if n else None,
            'absolute_bins_all_pct': (100 * np.histogram(ae, bins)[0] / n).tolist() if n else None,
            'relative_bins_numeric_pct': (100 * np.histogram(rel, rel_bins)[0] / len(rel)).tolist() if len(rel) else None,
            'no_output_reasons': dict(Counter(r['reason'] for r in rows if r['prediction_m'] is None))}


def add(rows, frame, route, geometry, result):
    stats = result.get('statistics')
    prediction = None if not stats else float(stats['mean_m'])
    if prediction is not None and not np.isfinite(prediction):
        prediction = None
    reasons = result.get('quality_flags', result.get('reasons', []))
    rows.append({**frame, 'route': route, 'geometry': geometry,
                 'prediction_m': prediction,
                 'reason': '|'.join(reasons) if prediction is None else None})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wet-root', type=Path, required=True)
    p.add_argument('--dry-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=202610120341)
    p.add_argument('--cases', type=int, default=99)
    p.add_argument('--repeats', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    audits, installations = [], {}
    for case in range(args.cases):
        try:
            installations[case] = install(case, args.wet_root, args.dry_root, args.seed)
            audits.append(installations[case]['audit'])
        except (ValueError, FileNotFoundError) as exc:
            audits.append({'case': case, 'initialization_error': str(exc)})
    (args.output / 'installation-audit.json').write_text(json.dumps(audits, indent=2))
    model8, model6 = Predictor('v8'), Predictor('v6')
    rows = []
    for case in range(args.cases):
        folders = [args.wet_root / f'{case * 6 + i:05d}' for i in range(6)]
        if case not in installations:
            continue
        item = installations[case]
        for folder in folders:
            meta = json.loads((folder / 'metadata.json').read_text())
            rgb = cv2.imread(str(folder / 'rgb.png'))
            frames = [f for f in json.loads((folder / 'frames.json').read_text())
                      if f['repeat'] < args.repeats]
            raw = [np.load(f['raw_path'], allow_pickle=False) for f in frames]
            p8 = predicted(model8, [rgb] * len(frames), raw)
            need6 = meta['scenario'] in ('glare', 'depth_failure')
            p6 = predicted(model6, [rgb] * len(frames), raw) if need6 else None
            enhanced = None
            if meta['scenario'] == 'low_light':
                bright = adaptive_exposure_correction(rgb, {'enabled': True, 'clahe_clip_limit': 1.5}).image_bgr
                enhanced = predicted(model8, [bright] * len(frames), raw)
            truth = reference_statistics(meta, replace(
                HeightCalibration.from_simulation(meta),
                bottom_z_m=float(meta['interior_bottom_z_m'])))['mean_m']
            for i, frame in enumerate(frames):
                key = {'case': case, 'slot': frame['slot'], 'scenario': meta['scenario'],
                       'sensor': meta['sensor_family'], 'band': meta['band'],
                       'variant': frame['variant'], 'repeat': frame['repeat'],
                       'true_height_m': truth}
                mode = 'partition' if meta['scenario'] == 'uneven_surface' else 'sensor'
                for geo in ('ideal', 'empty_fitted'):
                    cal = item['true'] if geo == 'ideal' else item['fitted']
                    add(rows, key, 'refined_v8', geo, refined(
                        rgb, raw[i], p8[i]['mask'], cal, key['sensor'], mode))
                    if enhanced is not None:
                        add(rows, key, 'exposure_refined_v8', geo, refined(
                            rgb, raw[i], enhanced[i]['mask'], cal, key['sensor'], mode))
                    if p6 is not None:
                        add(rows, key, 'refined_v6', geo, refined(
                            rgb, raw[i], p6[i]['mask'], cal, key['sensor'], mode))
                if item['ordinary'] is not None:
                    ctx, ref = item['ordinary']
                    add(rows, key, 'diffuse_v40_naive', 'empty_fitted',
                        estimate(raw[i], ctx, ref, 'paired'))
        print(json.dumps({'case_done': case + 1, 'cases': args.cases,
                          'rows': len(rows)}), flush=True)
    with (args.output / 'records.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + '\n')
    groups = defaultdict(list)
    for r in rows:
        scene = r['scenario'] + ('/' + r['variant'] if r['scenario'] == 'depth_failure' else '')
        for name in (scene, scene + '/' + str(r['band']), scene + '/' + r['sensor']):
            groups[f"{name}|{r['route']}|{r['geometry']}"].append(r)
    summary = {key: summarize(values) for key, values in sorted(groups.items())}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    sources = [Path(__file__), Path(__file__).with_name('generate_shared_empty_v42.py'),
               Path(__file__).parents[1] / 'src/liquid_depth/empty_geometry_v41.py']
    (args.output / 'protocol.json').write_text(json.dumps({
        'one_dry_RGBD_per_installation': True, 'wet_RGBD_dataset': str(args.wet_root),
        'dry_RGBD_dataset': str(args.dry_root), 'operator_click_sigma_px': .3,
        'operator_metric_sigma_m': '0.001+0.001*perpendicular_distance',
        'test_seed': args.seed, 'test_cases': args.cases, 'repeats': args.repeats,
        'truth_used_only_for_operator_simulation_and_scoring': True,
        'no_height_clipping_or_zero_snap': True, 'numeric_diagnostic_not_product_acceptance': True,
        'source_hashes': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
    }, indent=2))
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
