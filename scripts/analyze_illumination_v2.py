"""All-frame development diagnosis; far/shallow is a separate deferred stratum."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def score(rows):
    numeric = [r for r in rows if r['prediction_m'] is not None]
    error = np.array([abs(r['prediction_m'] - r['truth_m']) for r in numeric])
    relative = np.array([abs(r['prediction_m']-r['truth_m'])/r['truth_m']
                         for r in numeric if r['truth_m'] > 0])
    ok = sum(abs(r['prediction_m'] - r['truth_m']) <= max(.02, .02*r['truth_m'])
             for r in numeric)
    return {'n': len(rows), 'coverage': 100*len(numeric)/len(rows),
            'pass': 100*ok/len(rows),
            'mae': float(1000*error.mean()) if len(error) else None,
            'p95': float(1000*np.percentile(error, 95)) if len(error) else None,
            'max': float(1000*error.max()) if len(error) else None,
            'absrel_pct': float(100*relative.mean()) if len(relative) else None,
            'absolute_bins_all_pct': (100*np.histogram(error, [0, .02, .05, .1, np.inf])[0]/len(rows)).tolist(),
            'relative_bins_numeric_wet_pct': (100*np.histogram(relative, [0, .01, .02, .05, .1, np.inf])[0]/len(relative)).tolist() if len(relative) else None}


def temporal(values, window=4):
    valid = [v for v in values[-window:] if v is not None and np.isfinite(v)]
    if len(valid) < 3:
        return values[-1]
    return float(np.median(valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text().splitlines()]
    rows = [r for r in rows if r['route'] == 'merged_dry_geometry_v1']
    groups = defaultdict(list)
    all_rows = []
    for r in sorted(rows, key=lambda r: (r['case'], r['slot'], r['repeat'])):
        key = (r['case'], r['slot'])
        groups[key].append(r['prediction_m'])
        all_rows.append({**r, 'method': 'v1'})
        all_rows.append({**r, 'method': 'causal_median4',
                         'prediction_m': temporal(groups[key])})
    strata = defaultdict(list)
    for r in all_rows:
        far = r['camera_to_bottom_m'] >= 6 and 0 <= r['truth_m'] <= 1
        kind = 'deferred_far_shallow' if far else 'remaining'
        for sub in (kind, kind+('/dry' if r['truth_m'] == 0 else '/wet')):
            strata[f"{r['scene']}|{r['level']}|{sub}|{r['method']}"].append(r)
    summary = {k: score(v) for k, v in sorted(strata.items())}
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False))
    for key, value in summary.items():
        if '|remaining|' in key:
            print(key, value)


if __name__ == '__main__':
    main()
