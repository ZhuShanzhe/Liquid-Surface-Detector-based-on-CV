"""Paired error distributions and installation-clustered uncertainty, no policy changes."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate_visual_route_v1 import metrics

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/float-tail-v3-20261015')


def main():
    p = OUT/'fresh-evaluation'
    rows = [json.loads(line) for line in (p/'records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        if row['mode'] == 'v3_auto': groups[row['scene']].append(row)
    paired = {}; rng = np.random.default_rng(155927)
    for scene, group in groups.items():
        cases = sorted({r['case'] for r in group})
        matrix = []
        for case in cases:
            values = []
            for row in group:
                if row['case'] != case: continue
                previous = lookup[(case, row['repeat'], 'v2_auto')]
                truth = row['truth']['highest_height_m']; tolerance = max(.05, .05*truth)
                a, b = abs(previous['highest_height_m']-truth), abs(row['highest_height_m']-truth)
                values.append([a, b, a <= tolerance, b <= tolerance])
            matrix.append(values)
        matrix = np.asarray(matrix)
        samples = matrix[rng.integers(0, len(cases), (1200, len(cases)))].reshape(1200, -1, 4)
        deltas = {'mae_change_mm': 1000*(samples[:, :, 1].mean(1)-samples[:, :, 0].mean(1)),
                  'p95_change_mm': 1000*(np.percentile(samples[:, :, 1], 95, axis=1)-np.percentile(samples[:, :, 0], 95, axis=1)),
                  'pass_change_percentage_points': 100*(samples[:, :, 3].mean(1)-samples[:, :, 2].mean(1))}
        paired[scene] = {'installations': len(cases), 'frames': len(group),
            'cluster_bootstrap_95pct': {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in deltas.items()}}
    (p/'paired-bootstrap.json').write_text(json.dumps(paired, indent=2))
    for filename in ['liquid-summary', 'highest-summary']:
        data = json.loads((p/f'{filename}.json').read_text()); print(filename)
        for scene in groups:
            for mode in ['v2_auto', 'v3_off', 'v3_auto']:
                value = data[f'{scene}|all|all_frames|{mode}']
                print(scene, mode, {k: round(value[k], 3) for k in ['frames', 'numeric_coverage_pct', 'mae_mm', 'p95_mm', 'pass_pct']})
    analysis = []
    for row in rows:
        if row['mode'] != 'v3_auto': continue
        old = lookup[(row['case'], row['repeat'], 'v2_auto')]
        truth = row['truth']['highest_height_m']
        analysis.append({'case': row['case'], 'repeat': row['repeat'], 'scene': row['scene'],
            'truth_m': truth, 'old_error_mm': 1000*abs(old['highest_height_m']-truth),
            'new_error_mm': 1000*abs(row['highest_height_m']-truth),
            'new_source': row['diagnostics']['highest_source'], 'visible_float_pixels': row['floating_pixels_offline'],
            'candidate_count': row['diagnostics'].get('candidate_count', 0),
            'accepted_count': len(row['diagnostics']['verified_regions'])})
    (p/'tail-cases.json').write_text(json.dumps(sorted(analysis, key=lambda r: -r['new_error_mm']), indent=2))
    root = Path('/root/autodl-tmp/liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015')
    loss_groups = defaultdict(list); missing = {}
    for row in rows:
        if row['scene'] != 'depth_failure': continue
        key = (row['case'], row['repeat'])
        if key not in missing:
            folder = root/f"{row['case']:05d}"
            label = np.load(folder/'labels_offline.npz')['semantic']
            frames = json.loads((folder/'frames.json').read_text())
            frame = next(frame for frame in frames if frame['repeat'] == row['repeat'])
            raw = np.load(frame['raw_path']); water = label == 2
            missing[key] = 1-float((raw[water] > 0).mean()) if water.any() else None
        value = missing[key]
        subset = 'no_visible_liquid' if value is None else 'within_0_50pct' if value <= .5 else 'above_50pct_boundary'
        loss_groups[subset+'|'+row['mode']].append(row)
    (p/'loss-scope.json').write_text(json.dumps({'definition': 'missing numeric raw depth within visible liquid pixels; offline only',
        'metrics': {k: metrics(v) for k, v in loss_groups.items()}}, indent=2))
    print('DIAGNOSTICS', (p/'diagnostics-summary.json').read_text())


if __name__ == '__main__': main()
