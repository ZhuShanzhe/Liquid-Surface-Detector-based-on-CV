"""Post-freeze distributions and paired installation-level uncertainty."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-surface-v7-20261016/independent-test')


def error(r): return abs(r['highest_height_m']-r['truth']['highest_height_m'])


def passed(r): return error(r) <= max(.05, .05*r['truth']['highest_height_m'])


def main():
    rows = [json.loads(line) for line in (OUT/'records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    wet = [r for r in rows if r['mode'] == 'v7' and r['actual_visible_floating_object']]; comparisons = {}
    for baseline in ['v4', 'v5', 'v7_guards']:
        deltas = []; by_case = {}
        for r in wet:
            old = lookup[(r['case'], r['repeat'], baseline)]; delta = 1000*(error(r)-error(old)); deltas.append(delta)
            by_case.setdefault(r['case'], []).append([delta, 100*(int(passed(r))-int(passed(old)))])
        grouped = np.asarray([np.mean(v, axis=0) for v in by_case.values()]); rng = np.random.default_rng(72711)
        boot = np.stack([grouped[rng.integers(0, len(grouped), len(grouped))].mean(0) for _ in range(2000)])
        comparisons[baseline] = {'improved_over_5mm': sum(int(d < -5) for d in deltas), 'degraded_over_5mm': sum(int(d > 5) for d in deltas),
            'within_5mm': sum(int(abs(d) <= 5) for d in deltas), 'mae_delta_mm_95pct_ci': np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
            'pass_delta_percentage_points_95pct_ci': np.percentile(boot[:, 1], [2.5, 97.5]).tolist()}
    failures = []; used = []; fallback = []; chosen_kinds = {}; witness_stats = []
    for r in wet:
        aux = r['diagnostics']['floating_peak_auxiliary']
        if aux['v7_used']:
            used.append(r); selected = max(aux['v7_models'], key=lambda m: m['height_m'])
            chosen_kinds[selected['kind']] = chosen_kinds.get(selected['kind'], 0)+1
            witness_stats.append(selected['model']['witness_cells'])
        else: fallback.append(r)
        if not passed(r): failures.append({'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'band': r['band'],
            'truth_m': r['truth']['highest_height_m'], 'error_mm': 1000*error(r), 'signed_error_mm': 1000*(r['highest_height_m']-r['truth']['highest_height_m']),
            'v7_used': aux['v7_used'], 'policy_bypass': aux['policy_bypass'],
            'v4_error_mm': 1000*error(lookup[(r['case'], r['repeat'], 'v4')]), 'v5_error_mm': 1000*error(lookup[(r['case'], r['repeat'], 'v5')]),
            'accepted_models': aux['v7_models']})
    audit = {'wet_frames': len(wet), 'v7_used_frames': len(used), 'fallback_frames': len(fallback), 'chosen_kinds': chosen_kinds,
        'chosen_median_witness_cells': float(np.median(witness_stats)) if witness_stats else None,
        'liquid_changes_all_method_records': sum(int(r['liquid_depth_m'] != lookup[(r['case'], r['repeat'], 'v4')]['liquid_depth_m']) for r in rows),
        'comparisons': comparisons, 'failures': failures,
        'max_error_v7_adopted_mm': 1000*max(map(error, used), default=0), 'max_error_fallback_mm': 1000*max(map(error, fallback), default=0)}
    (OUT/'post-analysis.json').write_text(json.dumps(audit, indent=2))
    summary = json.loads((OUT/'summary.json').read_text()); fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, field, title in [(axes[0], 'absolute_error_bins_pct', 'Highest height absolute error'), (axes[1], 'relative_error_bins_pct_nonzero', 'Highest height relative error')]:
        for offset, mode, label in [(-.27, 'v4', 'V4'), (-.09, 'v5', 'V5'), (.09, 'v7_guards', 'V7 checks'), (.27, 'v7', 'V7 selected facets')]:
            bins = summary[f'actual_visible_floats|all|{mode}']['highest'][field]; x = np.arange(len(bins)); ax.bar(x+offset, list(bins.values()), .18, label=label)
        ax.set_xticks(x, list(bins)); ax.set_ylabel('Frames (%)'); ax.set_title(title); ax.legend(fontsize=7); ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT/'highest-error-distributions.png', dpi=180)
    ranges = {}
    for band in sorted({r['band'] for r in rows}):
        group = [r for r in rows if r['mode'] == 'v7' and r['band'] == band]
        ranges[str(band)] = {'frames': len(group), 'liquid_min_m': min(r['truth']['liquid_height_m'] for r in group), 'liquid_max_m': max(r['truth']['liquid_height_m'] for r in group)}
    (OUT/'depth-strata-ranges.json').write_text(json.dumps(ranges, indent=2))
    print('V7_ANALYSIS_COMPLETE', json.dumps({k: v for k, v in audit.items() if k != 'failures'}), flush=True)


if __name__ == '__main__': main()
