"""Post-evaluation distributions, paired uncertainty, and scope checks."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-surface-v6-20261016/independent-test')


def error(row): return abs(row['highest_height_m']-row['truth']['highest_height_m'])


def passed(row): return error(row) <= max(.05, .05*row['truth']['highest_height_m'])


def main():
    rows = [json.loads(line) for line in (OUT/'records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    wet = [r for r in rows if r['mode'] == 'surface_reliable' and r['actual_visible_floating_object']]
    comparisons = {}
    for baseline in ['v4', 'v5', 'surface_old_score']:
        deltas = [1000*(error(r)-error(lookup[(r['case'], r['repeat'], baseline)])) for r in wet]
        by_case = {}
        for r in wet:
            old = lookup[(r['case'], r['repeat'], baseline)]
            by_case.setdefault(r['case'], []).append([1000*(error(r)-error(old)), 100*(int(passed(r))-int(passed(old)))])
        grouped = np.asarray([np.mean(v, axis=0) for v in by_case.values()]); rng = np.random.default_rng(61337)
        bootstrap = np.stack([grouped[rng.integers(0, len(grouped), len(grouped))].mean(0) for _ in range(2000)])
        comparisons[baseline] = {'improved_over_5mm': sum(int(d < -5) for d in deltas),
            'degraded_over_5mm': sum(int(d > 5) for d in deltas), 'within_5mm': sum(int(abs(d) <= 5) for d in deltas),
            'mae_delta_mm_95pct_ci': np.percentile(bootstrap[:, 0], [2.5, 97.5]).tolist(),
            'pass_delta_percentage_points_95pct_ci': np.percentile(bootstrap[:, 1], [2.5, 97.5]).tolist()}
    failures = []
    used = []
    for r in wet:
        aux = r['diagnostics']['floating_peak_auxiliary']
        if aux['surface_used']: used.append(r)
        if not passed(r):
            failures.append({'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'band': r['band'],
                'truth_m': r['truth']['highest_height_m'], 'error_mm': 1000*error(r), 'signed_error_mm': 1000*(r['highest_height_m']-r['truth']['highest_height_m']),
                'surface_used': aux['surface_used'], 'old_v4_error_mm': 1000*error(lookup[(r['case'], r['repeat'], 'v4')]),
                'previous_v5_error_mm': 1000*error(lookup[(r['case'], r['repeat'], 'v5')]), 'surfaces': aux['verified_regions']})
    largest = sorted(wet, key=error, reverse=True)[:10]
    diagnosis = {'comparisons': comparisons, 'wet_frames': len(wet), 'surface_used_frames': len(used),
        'surface_fallback_frames': len(wet)-len(used), 'failures': failures,
        'largest_errors': [{'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'error_mm': 1000*error(r)} for r in largest],
        'liquid_changes_all_frames': sum(int(r['liquid_depth_m'] != lookup[(r['case'], r['repeat'], 'v4')]['liquid_depth_m']) for r in rows),
        'source_independence': 'different installation/render/noise seed; same simulator family'}
    strata = {}
    for band in sorted({r['band'] for r in rows}):
        values = [r for r in rows if r['mode'] == 'surface_reliable' and r['band'] == band]
        strata[str(band)] = {'min_liquid_m': min(r['truth']['liquid_height_m'] for r in values), 'max_liquid_m': max(r['truth']['liquid_height_m'] for r in values), 'frames': len(values)}
    (OUT/'post-analysis.json').write_text(json.dumps(diagnosis, indent=2)); (OUT/'depth-strata-ranges.json').write_text(json.dumps(strata, indent=2))
    summary = json.loads((OUT/'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, field, title in [(axes[0], 'absolute_error_bins_pct', 'Highest height absolute error'),
        (axes[1], 'relative_error_bins_pct_nonzero', 'Highest height relative error')]:
        for offset, mode, label in [(-.25, 'v4', 'V4'), (0, 'v5', 'V5'), (.25, 'surface_reliable', 'Surface + reliability')]:
            bins = summary[f'actual_visible_floats|all|{mode}']['highest'][field]
            x = np.arange(len(bins)); ax.bar(x+offset, list(bins.values()), .25, label=label)
        ax.set_xticks(x, list(bins)); ax.set_ylabel('Frames (%)'); ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT/'highest-error-distributions.png', dpi=180)
    print('ANALYSIS_COMPLETE', json.dumps({k: v for k, v in diagnosis.items() if k not in {'failures', 'largest_errors'}}), flush=True)


if __name__ == '__main__': main()
