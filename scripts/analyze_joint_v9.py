"""Distributions, causal cold/warm comparisons, unconditional risk and tails."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-joint-v9-20261016/independent-test')
MODES = ['v4', 'v5', 'v8', 'v9_single', 'v9']


def error(r):
    return abs(r['highest_height_m']-r['truth']['highest_height_m'])


def passed(r):
    return error(r) <= max(.05, .05*r['truth']['highest_height_m'])


def main():
    rows = [json.loads(l) for l in (OUT/'records.jsonl').read_text().splitlines()]
    ix = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    wet = [r for r in rows if r['mode'] == 'v9' and r['actual_visible_floating_object']]
    paired = {}
    for mode in ['v4', 'v5', 'v8', 'v9_single']:
        grouped = {}
        deltas = []
        for r in wet:
            old = ix[r['case'], r['repeat'], mode]
            delta = (error(r)-error(old))*1000
            deltas.append(delta)
            grouped.setdefault(r['case'], []).append([delta, 100*(int(passed(r))-int(passed(old)))])
        v = np.asarray([np.mean(g, axis=0) for g in grouped.values()])
        rng = np.random.default_rng(94913)
        boot = np.stack([v[rng.integers(len(v), size=len(v))].mean(0) for _ in range(3000)])
        paired[mode] = {'improved_over_5mm': sum(d < -5 for d in deltas), 'degraded_over_5mm': sum(d > 5 for d in deltas),
            'within_5mm': sum(abs(d) <= 5 for d in deltas), 'mae_delta_mm_95pct_ci': np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
            'pass_delta_pp_95pct_ci': np.percentile(boot[:, 1], [2.5, 97.5]).tolist()}
    choices = {}
    for mode in ['v8', 'v9_single', 'v9']:
        frames = [r for r in rows if r['mode'] == mode and r['actual_visible_floating_object']]
        chosen = []
        for r in frames:
            audit = r['offline_instance_audit']
            if audit['chosen'] is not None:
                c = audit['chosen']
                chosen.append({**c, 'physical_frame_error_mm': c['height_error_to_frame_mm']+
                    1000*(audit['visible_top_m']-r['truth']['highest_height_m'])})
        choices[mode] = {'adopted_frames': len(chosen), 'physical_frames': sum(c['physical'] for c in chosen),
            'highest_instance_within_20mm': sum(c['physical'] and c['gap_to_visible_top_mm'] <= 20 for c in chosen),
            'risk_empirical_frame_error_coverage_pct_all_adopted': 100*float(np.mean([abs(c['physical_frame_error_mm']) <= c['predicted_risk_mm'] for c in chosen])) if chosen else None}
    stages = {}
    for name in ['visual_features_candidates_ms', 'joint_features_score_group_ms', 'hard_and_temporal_ms', 'fallback_or_output_ms']:
        values = [r['diagnostics']['floating_peak_auxiliary']['stage_ms'][name] for r in wet]
        stages[name] = {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95))}
    weak = [r for r in wet if r['diagnostics']['floating_peak_auxiliary']['chosen_support'] == 'causal_small_object']
    changes = []
    for r in weak:
        old = ix[r['case'], r['repeat'], 'v9_single']
        changes.append({'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'v9_error_mm': 1000*error(r),
            'single_error_mm': 1000*error(old), 'audit': r['offline_instance_audit']})
    failures = []
    for r in wet:
        if passed(r):
            continue
        aux = r['diagnostics']['floating_peak_auxiliary']
        c = r['offline_instance_audit']['chosen']
        reason = ('fallback_after_checks' if not aux['v9_used'] else 'wrong_region_or_echo' if not c['physical'] else
            'lower_instance_selected' if c['gap_to_visible_top_mm'] > 20 else 'highest_instance_metric_error')
        failures.append({'case': r['case'], 'repeat': r['repeat'], 'sensor': r['sensor'], 'band': r['band'], 'reason': reason,
            'signed_error_mm': 1000*(r['highest_height_m']-r['truth']['highest_height_m']), 'audit': r['offline_instance_audit'],
            'v4_error_mm': 1000*error(ix[r['case'], r['repeat'], 'v4']), 'v5_error_mm': 1000*error(ix[r['case'], r['repeat'], 'v5']),
            'v8_error_mm': 1000*error(ix[r['case'], r['repeat'], 'v8']), 'checks': aux['hard_checks'],
            'instances_before_hard': aux['instances_before_hard'], 'identity_pass': aux['joint_identity_pass'],
            'risk_pass': aux['joint_risk_pass']})
    result = {'paired': paired, 'adopted_audit': choices, 'stage_latency': stages, 'small_temporal_adoptions': changes,
        'cold_start_output_differences': sum(abs(r['highest_height_m']-ix[r['case'], r['repeat'], 'v9_single']['highest_height_m']) > 1e-9 for r in wet if r['repeat'] == 0),
        'liquid_changes_all_records': sum(r['liquid_depth_m'] != ix[r['case'], r['repeat'], 'v4']['liquid_depth_m'] for r in rows), 'failures': failures}
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
    print('V9_ANALYSIS_COMPLETE', json.dumps({k: v for k, v in result.items() if k not in ['failures', 'small_temporal_adoptions']}), flush=True)


if __name__ == '__main__':
    main()
