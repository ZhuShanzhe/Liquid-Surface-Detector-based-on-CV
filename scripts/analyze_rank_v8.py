"""Paired errors, learned-risk calibration, instance selection, and distributions."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-rank-v8-20261016/independent-test')
MODES = ['v4', 'v5', 'v8_no_rank', 'v8', 'v8_research']


def error(r):
    return abs(r['highest_height_m']-r['truth']['highest_height_m'])


def passed(r):
    return error(r) <= max(.05, .05*r['truth']['highest_height_m'])


def main():
    rows = [json.loads(x) for x in (OUT/'records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    wet = [r for r in rows if r['mode'] == 'v8' and r['actual_visible_floating_object']]
    paired = {}
    for target, baseline in [(t, b) for t in ['v8', 'v8_research'] for b in ['v4', 'v5', 'v8_no_rank']]:
        groups = {}
        deltas = []
        target_frames = [r for r in rows if r['mode'] == target and r['actual_visible_floating_object']]
        for r in target_frames:
            b = lookup[r['case'], r['repeat'], baseline]
            d = 1000*(error(r)-error(b))
            deltas.append(d)
            groups.setdefault(r['case'], []).append([d, 100*(int(passed(r))-int(passed(b)))])
        values = np.asarray([np.mean(v, axis=0) for v in groups.values()])
        rng = np.random.default_rng(83811)
        boot = np.stack([values[rng.integers(len(values), size=len(values))].mean(0) for _ in range(3000)])
        paired[f'{target}_vs_{baseline}'] = {'improved_over_5mm': sum(d < -5 for d in deltas), 'degraded_over_5mm': sum(d > 5 for d in deltas),
            'within_5mm': sum(abs(d) <= 5 for d in deltas), 'mae_delta_mm_95pct_ci': np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
            'pass_delta_pp_95pct_ci': np.percentile(boot[:, 1], [2.5, 97.5]).tolist()}
    instance = {}
    for mode in MODES[2:]:
        frames = [r for r in rows if r['mode'] == mode and r['actual_visible_floating_object']]
        audits = [r['offline_instance_audit'] for r in frames]
        chosen = [a['chosen'] for a in audits if a['chosen'] is not None]
        physical = [c for c in chosen if c['physical']]
        instance[mode] = {'frames': len(frames), 'learned_adopted_frames': len(chosen),
            'candidate_analysis_frames': sum(a['candidate_count'] > 0 for a in audits),
            'highest_candidate_present_frames': sum(a['highest_instance_candidate_present'] for a in audits),
            'chosen_physical_frames': len(physical),
            'chosen_highest_instance_within_20mm': sum(c['gap_to_visible_top_mm'] <= 20 for c in physical),
            'chosen_other_lower_instance': sum(c['gap_to_visible_top_mm'] > 20 for c in physical),
            'chosen_component_error_mae_mm': float(np.mean([abs(c['height_error_to_component_mm']) for c in physical])) if physical else None,
            'risk_80pct_empirical_coverage_pct': 100*float(np.mean([abs(c['height_error_to_component_mm']) <= c['predicted_risk_mm'] for c in physical])) if physical else None}
    failures = []
    failure_reasons = {}
    for r in wet:
        if passed(r):
            continue
        aux = r['diagnostics']['floating_peak_auxiliary']
        chosen = r['offline_instance_audit']['chosen']
        if not aux['v8_used']:
            reason = 'validation_bypass_'+aux['v8_policy']['bypass'] if aux['v8_policy'].get('bypass') else 'new_candidate_rejected_then_fallback'
        elif not chosen['physical']:
            reason = 'wrong_object_or_echo_accepted'
        elif chosen['gap_to_visible_top_mm'] > 20:
            reason = 'lower_instance_selected'
        else:
            reason = 'highest_instance_height_error'
        failure_reasons[reason] = failure_reasons.get(reason, 0)+1
        failures.append({'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'band': r['band'],
            'reason': reason,
            'signed_error_mm': 1000*(r['highest_height_m']-r['truth']['highest_height_m']), 'audit': r['offline_instance_audit'],
            'v4_error_mm': 1000*error(lookup[r['case'], r['repeat'], 'v4']),
            'v5_error_mm': 1000*error(lookup[r['case'], r['repeat'], 'v5']),
            'v8_used': r['diagnostics']['floating_peak_auxiliary']['v8_used']})
    bypass = [r for r in rows if r['mode'] == 'v8' and r['diagnostics']['floating_peak_auxiliary']['v8_policy'].get('bypass') == 'v5']
    bypass_deltas = [abs(r['highest_height_m']-lookup[r['case'], r['repeat'], 'v5']['highest_height_m']) for r in bypass]
    result = {'paired': paired, 'instance_audit': instance, 'failures': failures, 'failure_reasons': failure_reasons,
        'fast_v5_bypass_frames': len(bypass), 'fast_v5_bypass_differences_over_1nm': sum(d > 1e-9 for d in bypass_deltas),
        'fast_v5_bypass_max_difference_mm': 1000*max(bypass_deltas, default=0),
        'liquid_changes_all_records': sum(r['liquid_depth_m'] != lookup[r['case'], r['repeat'], 'v4']['liquid_depth_m'] for r in rows)}
    (OUT/'post-analysis.json').write_text(json.dumps(result, indent=2))
    summary = json.loads((OUT/'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for ax, field, title in [(axes[0], 'absolute_error_bins_pct', 'Highest height absolute error'), (axes[1], 'relative_error_bins_pct_nonzero', 'Highest height relative error')]:
        for i, mode in enumerate(MODES):
            bins = summary[f'actual_visible_floats|all|{mode}']['highest'][field]
            x = np.arange(len(bins))
            ax.bar(x+(i-2)*.16, list(bins.values()), .16, label=mode)
        ax.set_xticks(x, list(bins))
        ax.set_ylabel('Frames (%)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(OUT/'highest-error-distributions.png', dpi=180)
    print('V8_ANALYSIS_COMPLETE', json.dumps({k: v for k, v in result.items() if k != 'failures'}), flush=True)


if __name__ == '__main__':
    main()
