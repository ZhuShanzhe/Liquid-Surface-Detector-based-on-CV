"""Paired 50%-total-loss audit using only existing synthetic sensor returns.

Never backfills or creates a return. Additional contiguous deletion stops at
exactly 50% missing among evaluator-known visible surface pixels. This uses the
oracle mask only for controlled corruption and scoring, never deployment input.
"""
import argparse
import hashlib
import json
from collections import defaultdict
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


def to_total_50(raw, visible, seed):
    output = np.array(raw, copy=True)
    eligible = visible & finite_valid(output)
    total = int(visible.sum())
    loss_now = total - int(eligible.sum())
    if total == 0 or loss_now > round(.5 * total):
        return None
    extra = max(0, round(.5 * total) - loss_now)
    yy, xx = np.nonzero(eligible)
    phase = (xx + seed % raw.shape[1]) % raw.shape[1]
    order = np.lexsort((yy, phase))
    chosen = order[:extra]
    output[yy[chosen], xx[chosen]] = 0
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wet-root', type=Path, required=True)
    parser.add_argument('--dry-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=202610122159)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    model = Predictor('v8')
    rows = []
    for case in range(99):
        item = install(case, args.wet_root, args.dry_root, args.seed)
        folder = args.wet_root / f'{case * 6 + 5:05d}'
        meta = json.loads((folder / 'metadata.json').read_text())
        rgb = cv2.imread(str(folder / 'rgb.png'))
        with np.load(folder / 'geometry.npz') as geom:
            visible = geom['mask'].astype(bool)
        cal = replace(HeightCalibration.from_simulation(meta),
                      bottom_z_m=float(meta['interior_bottom_z_m']))
        truth = reference_statistics(meta, cal)['mean_m']
        frames = [f for f in json.loads((folder / 'frames.json').read_text())
                  if f['variant'] == 'base']
        for frame in frames:
            base = np.load(frame['raw_path'], allow_pickle=False)
            modified = to_total_50(base, visible, args.seed + case * 10007 + frame['repeat'] * 509)
            if modified is None:
                continue
            scenarios = [('original', base), ('to_50_total', modified)]
            predictions = predicted(model, [rgb, rgb], [base, modified])
            for (regime, raw), prediction in zip(scenarios, predictions):
                mask = prediction['mask']
                route, value = route_output(item['bundle'], rgb, raw, mask)
                counts = point_chain(item['bundle'], raw, mask, visible)
                diag = None if value is not None else diagnosis(
                    item['bundle'], rgb, raw, mask, visible, counts, route['reasons'])
                rows.append({
                    'case': case, 'repeat': frame['repeat'], 'sensor': meta['sensor_family'],
                    'band': meta['band'], 'regime': regime, 'true_height_m': truth,
                    'prediction_m': value, 'actual_surface_loss_pct':
                        100 * (1 - counts['oracle_surface_valid'] / max(counts['oracle_surface_pixels'], 1)),
                    'counts': counts, 'route_reasons': route['reasons'], 'diagnosis': diag,
                })
        if case % 9 == 8:
            print(json.dumps({'case_done': case + 1, 'rows': len(rows)}), flush=True)
    with (args.output / 'records.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    groups = defaultdict(list)
    for row in rows:
        groups[row['regime']].append(row)
        groups[row['regime'] + '/' + row['sensor']].append(row)
        groups[row['regime'] + '/band_' + str(row['band'])].append(row)
    (args.output / 'summary.json').write_text(json.dumps(
        {key: summarize(group) for key, group in sorted(groups.items())}, indent=2))
    sources = [Path(__file__), Path(__file__).with_name('audit_depth_support_v46.py'),
               Path(__file__).parents[1] / 'src/liquid_depth/scene_route_policy_v45.py']
    (args.output / 'protocol.json').write_text(json.dumps({
        'no_surface_backfill': True, 'controlled_total_surface_loss_pct': 50,
        'in_scope_original_total_loss_max_pct': 50,
        'oracle_mask_used_only_for_deletion_and_diagnosis': True,
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, indent=2))
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
