"""Reproduce the supported ordinary, homogeneous, and partial-loss routes."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from evaluate_shared_empty_v45 import install
from evaluate_uniform_loss_v48 import LEVELS, delete_to_actual_loss
from experiment_range_v31 import load as load_transparent
from experiment_sr_v33 import configs as transparent_configs

from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics
from liquid_depth.supported_scene_meter import FormalSceneMeter


def metrics(rows):
    n = len(rows)
    numeric = [r for r in rows if r['predicted_height_m'] is not None]
    error = np.array([abs(r['predicted_height_m'] - r['true_height_m'])
                      for r in numeric], dtype=float)
    passed = sum(abs(r['predicted_height_m'] - r['true_height_m']) <=
                 max(.020, .02 * r['true_height_m']) for r in numeric)
    return {'frames': n, 'numeric': len(numeric),
            'coverage_pct': 100 * len(numeric) / n if n else None,
            'all_frame_pass_pct': 100 * passed / n if n else None,
            'mae_mm': float(1000 * error.mean()) if len(error) else None,
            'p95_mm': float(1000 * np.percentile(error, 95)) if len(error) else None}


def transparent_rows(configs_per_condition, frames_per_config):
    available = transparent_configs('test')
    rows = []
    for condition in ('transparent', 'translucent_mild'):
        pool = sorted((c for c in available if c['condition'] == condition),
                      key=lambda c: c['id'])
        indices = np.linspace(0, len(pool)-1, min(configs_per_condition, len(pool)))
        for config_index in dict.fromkeys(np.rint(indices).astype(int)):
            config = pool[config_index]
            images, masks, calibration, heights = load_transparent(config)
            scene = ('homogeneous_transparent' if condition == 'transparent'
                     else 'homogeneous_translucent')
            prior = [{ch: images[i, ch] for ch in ('left', 'right')}
                     for i in range(5)]
            meter = FormalSceneMeter.from_five_priors(
                calibration, masks, prior, heights, scene=scene)
            sample_index = np.linspace(0, len(config['tests'])-1,
                                       min(frames_per_config, len(config['tests'])))
            for index in dict.fromkeys(np.rint(sample_index).astype(int)):
                frame = {ch: images[index+5, ch] for ch in ('left', 'right')}
                result = meter.measure(scene, stereo_images=frame)
                with np.load(config['tests'][index]['evaluator_truth']) as truth:
                    true_height = float(truth['height_m'])
                rows.append({'scene': scene, 'config_id': config['id'],
                             'sample': int(index), 'true_height_m': true_height,
                             'predicted_height_m': result['liquid_depth_m'],
                             'route': result['route']})
            print("transparent_config", config["id"], flush=True)
    return rows


def depth_rows(args):
    ordinary, partial, exclusions = [], [], []
    for case in range(args.cases):
        folder = args.wet_root / f'{case*6+5:05d}'
        meta = json.loads((folder / 'metadata.json').read_text())
        cal = replace(HeightCalibration.from_simulation(meta),
                      bottom_z_m=float(meta['interior_bottom_z_m']))
        truth = float(reference_statistics(meta, cal)['mean_m'])
        with np.load(folder / 'geometry.npz') as geometry:
            surface = geometry['mask'].astype(bool)
        try:
            item = install(case, args.wet_root, args.dry_root, args.seed)
            profile = ({'disparity_step_px': args.disparity_step_px,
                        'disparity_noise_px': args.disparity_noise_px}
                       if meta['sensor_family'] == 'structured_light' else None)
            meter = FormalSceneMeter.from_empty_geometry(
                item['bundle'], item['dry'],
                max_return_depth_m=args.max_return_depth_m,
                quantized_disparity_profile=profile)
        except (ValueError, FileNotFoundError) as exc:
            exclusions.append({'case': case, 'reason': 'geometry:' + str(exc)})
            for rep in range(args.repeats):
                ordinary.append({'scene': 'ordinary_diffuse', 'case': case,
                                 'repeat': rep, 'sensor': meta['sensor_family'],
                                 'true_height_m': truth, 'predicted_height_m': None,
                                 'route': 'geometry_initialization_failed'})
            continue
        for rep in range(args.repeats):
            raw = np.load(folder / f'raw_native_{rep:02d}.npy', allow_pickle=False)
            result = meter.measure('ordinary_diffuse', depth_m=raw)
            ordinary.append({'scene': 'ordinary_diffuse', 'case': case,
                             'repeat': rep, 'sensor': meta['sensor_family'],
                             'band': meta['band'], 'true_height_m': truth,
                             'predicted_height_m': result['liquid_depth_m'],
                             'route': result['route']})
            native_loss = 100 * (1 - int((surface & (raw > 0)).sum()) /
                                 max(int(surface.sum()), 1))
            if native_loss > 50:
                exclusions.append({'case': case, 'repeat': rep,
                                   'reason': 'native_loss_above_50pct',
                                   'native_loss_pct': native_loss})
                continue
            samples = [('native', native_loss, raw)]
            for target in LEVELS[1:]:
                if target + .01 < native_loss:
                    continue
                samples.append(('controlled', float(target),
                                delete_to_actual_loss(
                                    raw, surface, target,
                                    args.seed + case * 10007 + rep * 509)))
            for kind, loss_pct, frame in samples:
                result = meter.measure('partial_depth_loss', depth_m=frame)
                partial.append({'scene': 'partial_depth_loss', 'case': case,
                                'repeat': rep, 'kind': kind,
                                'actual_loss_pct': loss_pct,
                                'sensor': meta['sensor_family'],
                                'band': meta['band'], 'true_height_m': truth,
                                'predicted_height_m': result['liquid_depth_m'],
                                'route': result['route']})
        print("depth_case", case + 1, flush=True)
    return ordinary, partial, exclusions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wet-root', type=Path, required=True)
    parser.add_argument('--dry-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cases', type=int, default=99)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--transparent-configs-per-condition', type=int, default=6)
    parser.add_argument('--transparent-frames-per-config', type=int, default=24)
    parser.add_argument('--max-return-depth-m', type=float, default=10.)
    parser.add_argument('--disparity-step-px', type=float, default=1/16)
    parser.add_argument('--disparity-noise-px', type=float, default=.26)
    args = parser.parse_args()
    if min(args.max_return_depth_m, args.disparity_step_px, args.disparity_noise_px) <= 0:
        raise ValueError('positive_sensor_profile_required')
    args.output.mkdir(parents=True, exist_ok=False)
    transparent = transparent_rows(args.transparent_configs_per_condition,
                                   args.transparent_frames_per_config)
    ordinary, partial, exclusions = depth_rows(args)
    rows = transparent + ordinary + partial
    with (args.output / 'records.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    groups = {
        'transparent': [r for r in transparent if r['scene'] == 'homogeneous_transparent'],
        'translucent': [r for r in transparent if r['scene'] == 'homogeneous_translucent'],
        'ordinary': ordinary, 'partial_depth_loss': partial,
    }
    for sensor in sorted({r['sensor'] for r in ordinary}):
        groups['ordinary_' + sensor] = [r for r in ordinary if r['sensor'] == sensor]
        groups['partial_' + sensor] = [r for r in partial if r['sensor'] == sensor]
    for start in range(0, 50, 10):
        groups[f'partial_loss_{start}_{start+10}'] = [
            r for r in partial if start <= r['actual_loss_pct'] < start + 10]
    groups['partial_loss_40_50'] = [
        r for r in partial if 40 <= r['actual_loss_pct'] <= 50]
    summary = {'measurand': 'actual liquid height above interior bottom',
               'tolerance': 'max(20 mm, 2% true liquid height)',
               'field_metrology_qualified': False,
               'sensor_profile': {'max_return_depth_m': args.max_return_depth_m,
                                  'disparity_step_px': args.disparity_step_px,
                                  'disparity_noise_px': args.disparity_noise_px},
               'scenes': {key: metrics(group) for key, group in groups.items()},
               'exclusions': exclusions,
               'protocol': {'wet_root': str(args.wet_root),
                            'dry_root': str(args.dry_root), 'seed': args.seed,
                            'transparent_configs_per_condition': args.transparent_configs_per_condition,
                            'transparent_frames_per_config': args.transparent_frames_per_config,
                            'ordinary_cases': args.cases, 'repeats': args.repeats,
                            'oracle_use': 'controlled deletion and offline scoring only',
                            'no_zero_snap_or_height_clip': True}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('COMPLETE', {k: summary['scenes'][k] for k in
                       ('transparent', 'translucent', 'ordinary', 'partial_depth_loss')}, flush=True)


if __name__ == '__main__':
    main()
