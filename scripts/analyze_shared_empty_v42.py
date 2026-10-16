"""Reconcile conditional numeric scores with all-installation denominators."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCENES = ('low_light', 'glare', 'floating_objects', 'uneven_surface',
          'depth_failure/base', 'depth_failure/drop75',
          'depth_failure/drop90', 'depth_failure/drop97')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=99)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    root = args.evaluation
    audit = json.loads((root / 'installation-audit.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if len(audit) != args.cases:
        raise ValueError('installation_audit_count_mismatch')
    success = [a for a in audit if 'initialization_error' not in a]
    bins = ['empty', '0.01-0.1', '0.1-1'] + [f'{x}-{x+1}' for x in range(1, 9)]
    geometry = {
        'attempted_installations': args.cases,
        'initialized_installations': len(success),
        'success_by_depth_band': {name: {
            'success': sum('initialization_error' not in a for a in audit[i * 9:(i + 1) * 9]),
            'attempts': len(audit[i * 9:(i + 1) * 9])
        } for i, name in enumerate(bins)},
        'failure_reasons': dict(Counter(a['initialization_error'] for a in audit
                                        if 'initialization_error' in a)),
    }
    scores = {}
    for scene in SCENES:
        scores[scene] = {}
        for route in ('refined_v8', 'diffuse_v40_naive'):
            key = f'{scene}|{route}|empty_fitted'
            if key not in summary:
                continue
            s = summary[key]
            expected_cases = args.cases if route == 'refined_v8' else 66
            all_frames = expected_cases * args.repeats
            scored = s['frames']
            if scored > all_frames:
                raise ValueError('conditional_frame_count_exceeds_full_population')
            scores[scene][route] = {
                'all_installations_frames': all_frames,
                'initialized_frames': scored,
                'numeric_frames': s['numeric_frames'],
                'full_numeric_coverage_pct': 100 * s['numeric_frames'] / all_frames,
                'full_pass_pct': s['all_frame_pass_pct'] * scored / all_frames,
                'conditional_mae_mm': s['mae_mm'],
                'conditional_p95_mm': s['p95_mm'],
                'conditional_absrel_pct': s['abs_rel_pct'],
                'conditional_absolute_bins_all_pct': s['absolute_bins_all_pct'],
                'conditional_relative_bins_numeric_pct': s['relative_bins_numeric_pct'],
                'failed_initialization_frames': all_frames - scored,
                'measurement_no_output_frames': scored - s['numeric_frames'],
            }
    output = {'geometry': geometry, 'full_pipeline_scores': scores,
              'definition': 'Full denominator includes failed dry-geometry installations; '
                            'conditional MAE/P95 only among numerical outputs after initialization.',
              'pass_tolerance': 'max(5 mm, 2% of actual liquid depth above interior bottom)',
              'production_qualified': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (root / 'full-denominator-analysis.json').write_text(json.dumps(output, indent=2))
    print(json.dumps({'initialized': len(success),
                      'scenes': {k: {r: {
                          'coverage_pct': round(v['full_numeric_coverage_pct'], 2),
                          'mae_mm_conditional': v['conditional_mae_mm'],
                          'pass_pct_all': round(v['full_pass_pct'], 2)}
                          for r, v in routes.items()}
                          for k, routes in scores.items()}}, indent=2))


if __name__ == '__main__':
    main()
