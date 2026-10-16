"""Post-freeze diagnostic analysis, never used for model or threshold selection."""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-apex-v5-20261016/independent-test')


def error(row): return abs(row['highest_height_m']-row['truth']['highest_height_m'])


def tolerance(row): return max(.05, .05*row['truth']['highest_height_m'])


def main():
    rows = [json.loads(line) for line in (OUT/'records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in rows}
    wet = [r for r in rows if r['mode'] == 'v5' and r['actual_visible_floating_object']]
    reasons = Counter(); failures = []; deltas = []
    for row in wet:
        old = lookup[(row['case'], row['repeat'], 'v4')]; deltas.append(1000*(error(row)-error(old)))
        if error(row) <= tolerance(row): continue
        audit = row['offline_candidate_audit']; top = [c for c in audit if c['highest_object_purity'] >= .35 and c['highest_object_overlap_pixels'] >= 2]
        diag = row['diagnostics']['floating_peak_auxiliary']; policy = diag['sensor_policy']
        supported = [c for c in top if c['object_score'] >= policy['threshold']]
        accepted = diag['verified_regions']
        accepted_top = [a for a in accepted if any(c['pixels'] == a['pixels'] and c['variant'] == a['variant'] and
            abs(c['height_raw_m']-a['height_raw_m']) < 1e-9 for c in top)]
        if accepted and row['highest_height_m']-row['truth']['highest_height_m'] > tolerance(row) and max(accepted, key=lambda c: c['height_m']) not in accepted_top:
            reason = 'false_or_other_peak_selected_above_true_highest'
        elif not top: reason = 'highest_object_no_pure_proposal'
        elif not supported: reason = 'highest_object_classifier_rejected'
        elif not accepted_top: reason = 'highest_object_interval_or_grouping_rejected'
        else:
            chosen = max(accepted, key=lambda c: c['height_m'])
            if chosen not in accepted_top: reason = 'other_object_or_echo_selected'
            elif abs(chosen['height_raw_m']-row['truth']['highest_height_m']) <= tolerance(row): reason = 'correction_or_consensus_degraded_good_raw_peak'
            else: reason = 'accepted_apex_raw_depth_or_sampling_bias'
        reasons[reason] += 1
        failures.append({'case': row['case'], 'repeat': row['repeat'], 'sensor': row['sensor'], 'band': row['band'],
            'liquid_truth_m': row['truth']['liquid_height_m'], 'highest_truth_m': row['truth']['highest_height_m'],
            'old_error_mm': 1000*error(old), 'new_error_mm': 1000*error(row),
            'signed_error_mm': 1000*(row['highest_height_m']-row['truth']['highest_height_m']),
            'diagnostic_reason': reason, 'pure_highest_candidates': top, 'accepted_regions': accepted})
    replay = [json.loads(line) for line in (OUT/'old-nine-diagnostic-replay.jsonl').read_text().splitlines()]
    replay_lookup = {(r['case'], r['repeat'], r['mode']): r for r in replay}
    original_failures = [r for r in replay if r['mode'] == 'v4' and error(r) > tolerance(r)]
    old_nine = []
    for old in original_failures:
        new = replay_lookup[(old['case'], old['repeat'], 'v5')]
        old_nine.append({'case': old['case'], 'repeat': old['repeat'], 'sensor': old['sensor'],
            'old_error_mm': 1000*error(old), 'new_error_mm': 1000*error(new), 'new_pass': error(new) <= tolerance(new),
            'new_signed_error_mm': 1000*(new['highest_height_m']-new['truth']['highest_height_m'])})
    analysis = {'independent_wet_frames': len(wet), 'paired_improved_over_5mm': sum(int(x < -5) for x in deltas),
        'paired_degraded_over_5mm': sum(int(x > 5) for x in deltas), 'paired_within_5mm': sum(int(abs(x) <= 5) for x in deltas),
        'failure_reason_counts_offline_only': dict(reasons), 'failures': failures,
        'old_nine_diagnostic_only_not_model_selection': old_nine,
        'old_nine_now_passing': sum(int(r['new_pass']) for r in old_nine)}
    analysis['extreme_error_by_mode'] = {}
    for mode in ['v4', 'v5']:
        values = np.asarray([error(r) for r in rows if r['mode'] == mode and r['actual_visible_floating_object']])
        analysis['extreme_error_by_mode'][mode] = {'maximum_mm': float(1000*values.max()),
            'p99_mm': float(1000*np.percentile(values, 99)), 'errors_at_least_one_metre': int((values >= 1.).sum())}
    (OUT/'post-analysis.json').write_text(json.dumps(analysis, indent=2))
    summary = json.loads((OUT/'summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, field, title in [(axes[0], 'absolute_error_bins_pct', 'Highest height absolute error'),
        (axes[1], 'relative_error_bins_pct_nonzero', 'Highest height relative error')]:
        old = summary['actual_visible_floats|all|v4']['highest'][field]
        new = summary['actual_visible_floats|all|v5']['highest'][field]
        x = np.arange(len(old)); ax.bar(x-.18, list(old.values()), .36, label='V4')
        ax.bar(x+.18, list(new.values()), .36, label='V5')
        ax.set_xticks(x, list(old)); ax.set_ylabel('Frames (%)'); ax.set_title(title); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.savefig(OUT/'highest-error-distributions.png', dpi=180)
    print('POST_ANALYSIS_COMPLETE', json.dumps({k: v for k, v in analysis.items() if k != 'failures'}), flush=True)


if __name__ == '__main__': main()
