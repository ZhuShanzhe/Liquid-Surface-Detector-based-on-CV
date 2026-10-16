"""Trace observed support and candidate decisions without inventing return origins."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(s) for s in (a.evaluation/'records.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    examples = []
    for r in rows:
        if r['camera_to_bottom_m'] >= 6 and r['truth_m'] <= 1:
            continue
        if r['method'] == 'v1':
            continue
        d = r['diagnostics']
        ref = d['illumination_refinement']
        count, total = d['selected_pixels'], d['static_roi_pixels']
        value = r['prediction_m']
        error = None if value is None else abs(value-r['truth_m'])
        reason = ('fewer_than_24_container_returns' if count < 24 else
                  'container_return_fraction_under_1pct' if count/max(total, 1) < .01 else
                  'dry_vote_ambiguous' if (d.get('layer_vote') or {}).get('applied') and
                  'restored_return_quantile_m' not in ref else
                  'returns_present_but_layer_or_noise_unresolved')
        group = f"{r['scene']}|{r['level']}|{r['method']}"
        groups[group].append({**r, 'error': error, 'large_error_reason': reason})
        if r['method'] == 'v2_single' and error is not None:
            examples.append({'case': r['case'], 'slot': r['slot'], 'repeat': r['repeat'],
                'scene': r['scene'], 'level': r['level'], 'sensor': r['sensor'],
                'true_m': r['truth_m'], 'predicted_m': value, 'absolute_error_mm': 1000*error,
                'selected_returns': count, 'static_roi_pixels': total,
                'depth_consistent_fraction_offline_only': r['surface_depth_consistent_fraction'],
                'value_source': ref['value_source'], 'diagnostic_category': reason})
    summary = {}
    for name, group in groups.items():
        bad = [r for r in group if r['error'] is not None and r['error'] > .1]
        history = [r for r in group if r['diagnostics']['illumination_refinement']['temporal_used']]
        corrected = [r for r in group if 'restored_return_quantile_m' in
                     r['diagnostics']['illumination_refinement']]
        summary[name] = {'frames': len(group), 'error_over_100mm': len(bad),
                         'observable_diagnostic_categories': dict(Counter(r['large_error_reason'] for r in bad)),
                         'large_error_sensor_counts': dict(Counter(r['sensor'] for r in bad)),
                         'large_error_depth_bands': dict(Counter(r['band'] for r in bad)),
                         'history_used_frames': len(history), 'dry_vote_guard_frames': len(corrected),
                         'signed_bias_mm': float(1000*np.mean([r['prediction_m']-r['truth_m']
                            for r in group if r['prediction_m'] is not None]))}
    examples.sort(key=lambda r: r['absolute_error_mm'], reverse=True)
    audit = json.loads((a.evaluation/'installation-audit.json').read_text())
    result = {'groups': summary, 'largest_single_frame_errors': examples[:15],
              'calibration': {'installations': len(audit),
                'max_bottom_distance_error_mm': max(abs(r['bottom_distance_error_mm']) for r in audit),
                'max_normal_error_deg': max(r['normal_error_deg'] for r in audit)},
              'return_origin_limit': 'Per-pixel native/wrong/flying source labels were not stored; depth-consistent fraction does not identify physical origin.'}
    (a.evaluation/'input-output-diagnosis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
