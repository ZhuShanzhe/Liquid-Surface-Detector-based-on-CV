"""Small image-pair integration pilot: V41 empty geometry -> five-prior meter.

The V33 optics simulator has no RGB-D channel. Its dry depth here is a clearly
labelled proxy, not a rendered sensor capture or a hardware validation.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from experiment_empty_geometry_v41 import annotation
from experiment_range_v31 import load
from liquid_depth.empty_geometry_v41 import build_geometry, make_refraction_meter
from liquid_depth.liquid_height_metrics import HeightCalibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-condition', type=int, default=2)
    parser.add_argument('--tests-per-config', type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    configs = json.loads(args.manifest.read_text())['configs']
    selected = []
    for condition in ('transparent', 'translucent_mild'):
        selected += [c for c in configs if c['condition'] == condition][:args.per_condition]
    rows, audit = [], []
    for index, config in enumerate(selected):
        ims, masks, cal, heights = load(config)
        k = np.array([[cal['fx'], 0., cal['cx']], [0., cal['fx'], cal['cy']], [0., 0., 1.]])
        pose = np.diag([1., -1., -1., 1.])
        pose[2, 3] = cal['bottom_m']
        model = HeightCalibration(k, pose, 0.,
                                  (cal['container_width_m'] / 2,
                                   cal['container_length_m'] / 2), 16.)
        rng = np.random.default_rng(42241 + index * 179)
        spec = annotation(model, cal['container_max_height_m'], 'rectangle', rng)
        spec['baseline_m'] = cal['baseline_m']
        spec['sensor_family'] = 'active_stereo'
        spec['metric_anchor'] = {'kind': 'perpendicular_distance',
                                 'distance_m': float(cal['bottom_m'] +
                                                     rng.normal(0, .001 + .001 * cal['bottom_m']))}
        # This separate image simulator does not emit depth. The dry-depth proxy
        # is a noisy observable, rather than exact ground-truth passed to V41.
        dry = np.zeros((cal['height'], cal['width']), np.float32)
        visible = cv2.erode((masks['left'] > 0).astype('uint8'), np.ones((3, 3), 'uint8')).astype(bool)
        dry[visible] = cal['bottom_m'] + rng.normal(0, .0008 + .0012 * cal['bottom_m']**2,
                                                   int(visible.sum()))
        dry[(dry <= 0) | (dry > 10)] = 0
        try:
            bundle = build_geometry(dry, k, spec)
            prior = [{ch: ims[j, ch] for ch in ('left', 'right')} for j in range(5)]
            method = 'baseline_v32' if config['condition'] == 'transparent' else 'midpoint_dense'
            meter = make_refraction_meter(bundle, k, masks, prior, heights,
                                          config['condition'], method,
                                          left_intrinsics=k)
        except ValueError as exc:
            audit.append({'config_id': config['id'], 'init_error': str(exc)})
            continue
        audit.append({'config_id': config['id'], 'condition': config['condition'],
                      'method': method, 'dry_valid_pixels': int((dry > 0).sum()),
                      'distance_error_mm': 1000 * (
                          bundle['bottom_plane_camera']['perpendicular_distance_m'] - cal['bottom_m']),
                      'bottom_model': 'proxy_not_rendered_RGBD',
                      'geometry_quality': bundle['quality']})
        # Fixed index schedule spans the entire stored test sequence, instead of
        # accidentally evaluating only shallow levels at its beginning.
        uniform_indices = [j for j, cap in enumerate(config['tests'])
                           if cap.get('population') == 'uniform']
        if not uniform_indices:
            raise ValueError('uniform_test_population_missing')
        uniform_budget = max(1, args.tests_per_config - 4)
        selected_indices = np.unique(np.r_[
            [0, 1, 4, 8],
            np.asarray(uniform_indices)[np.linspace(
                0, len(uniform_indices) - 1,
                min(uniform_budget, len(uniform_indices)), dtype=int)],
        ]).astype(int)
        for j in selected_indices:
            cap = config['tests'][int(j)]
            result = meter.measure({ch: ims[j + 5, ch] for ch in ('left', 'right')})
            with np.load(cap['evaluator_truth']) as gt:
                truth = float(gt['height_m'])
            value = result.get('height_m')
            rows.append({'config_id': config['id'], 'condition': config['condition'],
                         'sample_id': int(j), 'true_height_m': truth,
                         'prediction_m': float(value) if value is not None else None,
                         'reasons': result['reasons']})
        print('CONFIG', config['id'], len(rows), flush=True)
    (args.output / 'records.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    (args.output / 'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False))
    summary = {}
    for condition in ('transparent', 'translucent_mild'):
        q = [x for x in rows if x['condition'] == condition]
        valid = [x for x in q if x['prediction_m'] is not None]
        e = np.array([abs(x['prediction_m'] - x['true_height_m']) for x in valid])
        summary[condition] = {'frames': len(q), 'outputs': len(valid),
                              'mae_mm': float(e.mean() * 1000) if len(e) else None,
                              'all_frame_pass_pct': 100 * sum(
                                  abs(x['prediction_m'] - x['true_height_m']) <=
                                  max(.005, .02 * x['true_height_m']) for x in valid) /
                                  len(q) if q else None}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    (args.output / 'protocol.json').write_text(json.dumps({
        'scope': 'geometry-adapter-pilot-not-optical-performance-certification',
        'dry_depth': 'procedural_noisy_proxy_not_rendered_RGBD',
        'five_prior_images': True, 'known_heights_used_only_in_calibration': True,
        'no_gate_no_zero_snap_no_clip': True,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }, indent=2))
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
