"""Paired 0..50% *actual surface loss* evaluation from native sensor returns.

Select only native frames with <=0.5% initial missing surface returns, then
delete existing returns to reach 10% steps. No depth value is backfilled.
Evaluator-only visible-surface truth controls deletion and scoring, never model
or geometric inference. All installation/selection exclusions are reported.
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

from audit_depth_support_v46 import (diagnosis, finite_valid, point_chain,
                                     route_output, summarize)
from evaluate_liquid_height_v21 import Predictor
from evaluate_other_scenes_v34 import predicted
from evaluate_shared_empty_v45 import install
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics


LEVELS = (0, 10, 20, 30, 40, 50)


def delete_to_actual_loss(raw, visible, percent, seed):
    output = np.array(raw, copy=True)
    visible = np.asarray(visible, bool)
    valid = visible & finite_valid(output)
    target_missing = round(int(visible.sum()) * percent / 100)
    already_missing = int(visible.sum()) - int(valid.sum())
    if target_missing < already_missing:
        raise ValueError('native_loss_exceeds_requested_target')
    yy, xx = np.nonzero(valid)
    phase = (xx + seed % output.shape[1]) % output.shape[1]
    order = np.lexsort((yy, phase))
    chosen = order[:target_missing - already_missing]
    output[yy[chosen], xx[chosen]] = 0
    if np.any((raw <= 0) & (output > 0)):
        raise AssertionError('return_backfill_forbidden')
    return output


def group_stats(rows, denominator):
    base = summarize(rows)
    numeric = [r for r in rows if r['prediction_m'] is not None]
    ae = np.array([abs(r['prediction_m'] - r['true_height_m']) for r in numeric])
    rel = np.array([abs(r['prediction_m'] - r['true_height_m']) / r['true_height_m']
                    for r in numeric if r['true_height_m'] > 0])
    base.update({
        'all_installation_denominator': denominator,
        'all_installation_numeric_coverage_pct': 100 * len(numeric) / denominator,
        'all_installation_pass_pct': 100 * sum(
            abs(r['prediction_m'] - r['true_height_m']) <= max(.005, .02 * r['true_height_m'])
            for r in numeric) / denominator,
        'conditional_p95_mm': float(1000 * np.percentile(ae, 95)) if len(ae) else None,
        'conditional_absrel_pct': float(100 * rel.mean()) if len(rel) else None,
        'absolute_error_bins_all_pct': (100 * np.histogram(
            ae, [0, .005, .01, .02, .05, .1, .2, np.inf])[0] / denominator).tolist(),
        'relative_error_bins_numeric_pct': (100 * np.histogram(
            rel, [0, .005, .01, .02, .03, .05, .1, np.inf])[0] / len(rel)).tolist()
            if len(rel) else None,
        'actual_surface_loss_min_pct': min(r['actual_surface_loss_pct'] for r in rows),
        'actual_surface_loss_max_pct': max(r['actual_surface_loss_pct'] for r in rows),
        'actual_surface_loss_mean_pct': float(np.mean([r['actual_surface_loss_pct'] for r in rows])),
    })
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wet-root', type=Path, required=True)
    parser.add_argument('--dry-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=202610125237)
    parser.add_argument('--cases', type=int, default=99)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = Predictor('v8')
    rows, exclusions = [], []
    for case in range(args.cases):
        folder = args.wet_root / f'{case * 6 + 5:05d}'
        meta = json.loads((folder / 'metadata.json').read_text())
        rgb = cv2.imread(str(folder / 'rgb.png'))
        with np.load(folder / 'geometry.npz') as geom:
            visible = geom['mask'].astype(bool)
        frames = [f for f in json.loads((folder / 'frames.json').read_text())
                  if f['repeat'] < args.repeats]
        try:
            item = install(case, args.wet_root, args.dry_root, args.seed)
        except (ValueError, FileNotFoundError) as exc:
            for frame in frames:
                exclusions.append({'case': case, 'repeat': frame['repeat'],
                                   'reason': 'geometry:' + str(exc)})
            continue
        cal = replace(HeightCalibration.from_simulation(meta),
                      bottom_z_m=float(meta['interior_bottom_z_m']))
        truth = reference_statistics(meta, cal)['mean_m']
        for frame in frames:
            raw_base = np.load(frame['raw_path'], allow_pickle=False)
            baseline_loss = 100 * (1 - int((visible & finite_valid(raw_base)).sum()) /
                                   max(int(visible.sum()), 1))
            if baseline_loss > .5:
                exclusions.append({'case': case, 'repeat': frame['repeat'],
                                   'reason': 'native_surface_loss_above_0.5pct',
                                   'native_loss_pct': baseline_loss})
                continue
            samples = [raw_base if level == 0 else delete_to_actual_loss(
                raw_base, visible, level,
                args.seed + case * 10007 + frame['repeat'] * 509)
                for level in LEVELS]
            predictions = predicted(model, [rgb] * len(samples), samples)
            for level, raw, prediction in zip(LEVELS, samples, predictions):
                result, value = route_output(item['bundle'], rgb, raw, prediction['mask'])
                counts = point_chain(item['bundle'], raw, prediction['mask'], visible)
                actual = 100 * (1 - counts['oracle_surface_valid'] /
                                max(counts['oracle_surface_pixels'], 1))
                if level and abs(actual - level) > .02:
                    raise AssertionError('actual_surface_loss_target_mismatch')
                diag = None if value is not None else diagnosis(
                    item['bundle'], rgb, raw, prediction['mask'], visible,
                    counts, result['reasons'])
                rows.append({
                    'case': case, 'repeat': frame['repeat'], 'sensor': meta['sensor_family'],
                    'band': meta['band'], 'target_loss_pct': level,
                    'actual_surface_loss_pct': actual, 'native_loss_pct': baseline_loss,
                    'true_height_m': truth, 'prediction_m': value,
                    'counts': counts, 'route_reasons': result['reasons'],
                    'diagnosis': diag, 'geometry_source': item['bundle']['source'],
                })
        print(json.dumps({'case_done': case + 1, 'rows': len(rows),
                          'excluded_frames': len(exclusions)}), flush=True)
    with (args.output / 'records.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    (args.output / 'exclusions.json').write_text(json.dumps(exclusions, indent=2))
    groups = defaultdict(list)
    for row in rows:
        key = str(row['target_loss_pct'])
        groups[key].append(row)
        groups[key + '/' + row['sensor']].append(row)
        groups[key + '/band_' + str(row['band'])].append(row)
        groups[key + '/' + row['geometry_source']].append(row)
    summary = {key: group_stats(group, args.cases * args.repeats if '/' not in key else len(group))
               for key, group in sorted(groups.items())}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    sources = [Path(__file__), Path(__file__).with_name('generate_uniform_native_v48.py'),
               Path(__file__).with_name('audit_depth_support_v46.py'),
               Path(__file__).parents[1] / 'src/liquid_depth/simulation.py',
               Path(__file__).parents[1] / 'src/liquid_depth/scene_route_policy_v45.py']
    (args.output / 'protocol.json').write_text(json.dumps({
        'purpose': 'uniform actual surface-missing-return rate on paired native sensor proxy frames',
        'levels_pct': list(LEVELS), 'native_baseline_max_pct': .5,
        'no_backfill': True, 'test_seed': args.seed,
        'oracle_surface_mask_use': 'controlled deletion and offline scoring only',
        'measurand': 'actual liquid height above interior bottom',
        'tolerance': 'max(5mm, 2% actual height)',
        'selected_frames_per_level': len(groups['0']),
        'all_attempted_frames_per_level': args.cases * args.repeats,
        'exclusions_by_reason': dict(Counter(x['reason'] for x in exclusions)),
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, indent=2))
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
