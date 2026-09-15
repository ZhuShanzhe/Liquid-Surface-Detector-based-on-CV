"""Validation-only group guard for V13; no scenario labels enter inference."""
from __future__ import annotations
import math

SCENARIOS = ('transparent', 'translucent', 'multilayer', 'compound')
THRESHOLDS = (.3, .4, .5, .6, .7, .8, .9, .95)
METRICS = ('depth_mae_mm', 'depth_abs_rel', 'pixel_tolerance_pass',
           'valid_pixel_coverage', 'non_target_output_fraction', 'within_tolerance_coverage')


def failures(row, reference):
    if any(row.get(k) is None or reference.get(k) is None for k in METRICS):
        return ['missing_metrics']
    if any(not math.isfinite(row[k]) or not math.isfinite(reference[k]) for k in METRICS):
        return ['nonfinite_metrics']
    checks = {
        'mae': row['depth_mae_mm'] <= reference['depth_mae_mm'] * 1.05,
        'absrel': row['depth_abs_rel'] <= max(.03, reference['depth_abs_rel']),
        'precision': row['pixel_tolerance_pass'] >= max(.5, reference['pixel_tolerance_pass'] - .01),
        'coverage': row['valid_pixel_coverage'] >= reference['valid_pixel_coverage'] - .02,
        'non_target': row['non_target_output_fraction'] <= reference['non_target_output_fraction'] + .02,
    }
    return [key for key, passed in checks.items() if not passed]


def operating_point(report, baseline):
    """One global threshold must qualify in ALL predefined scenes and overall."""
    decisions = {}
    for t in THRESHOLDS:
        reasons = {}
        for group in ('all',) + SCENARIOS:
            try:
                row = report['summary'][group]['selected'][str(t)]
                ref = baseline['summary'][group]['selected']['0.5']
                bad = failures(row, ref)
            except KeyError:
                bad = ['missing_group']
            if bad:
                reasons[group] = bad
        decisions[str(t)] = reasons
    eligible = [t for t in THRESHOLDS if not decisions[str(t)]]
    threshold = max(eligible, key=lambda t: (
        sum(report['summary'][s]['selected'][str(t)]['within_tolerance_coverage'] for s in SCENARIOS) / len(SCENARIOS),
        t)) if eligible else None
    return {'threshold': threshold, 'rejected_by_group': decisions,
            'rule': 'one threshold; overall AND every predefined scene must pass; validation only'}


def checkpoint_score(report, reference):
    """Worst-scene delta dominates; each scene contributes independent of area."""
    delta = [report['summary'][s]['top1_recall_all_valid'] -
             reference['summary'][s]['top1_recall_all_valid'] for s in SCENARIOS]
    return min(delta) + .25 * sum(delta) / len(delta)
