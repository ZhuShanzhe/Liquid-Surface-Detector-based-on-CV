"""All-installation and conditional scores for independent V43 evaluation."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SCENES = ('low_light', 'glare', 'floating_objects', 'uneven_surface',
          'depth_failure/base', 'depth_failure/drop75',
          'depth_failure/drop90', 'depth_failure/drop97')
ROUTES = ('legacy_refined_v8', 'adapted_v44', 'adapted_exposure_v8',
          'adapted_v6', 'diffuse_v40_naive')


def stats(rows, denominator):
    numeric = [r for r in rows if r['prediction_m'] is not None]
    e = np.asarray([abs(r['prediction_m'] - r['true_height_m']) for r in numeric])
    rel = np.asarray([abs(r['prediction_m'] - r['true_height_m']) / r['true_height_m']
                      for r in numeric if r['true_height_m'] > 0])
    passed = sum(abs(r['prediction_m'] - r['true_height_m']) <=
                 max(.005, .02 * r['true_height_m']) for r in numeric)
    return {
        'frames_all': denominator, 'frames_initialized': len(rows),
        'numeric_frames': len(numeric),
        'full_numeric_coverage_pct': 100 * len(numeric) / denominator if denominator else None,
        'full_pass_pct': 100 * passed / denominator if denominator else None,
        'conditional_mae_mm': float(1000 * e.mean()) if len(e) else None,
        'conditional_p95_mm': float(1000 * np.percentile(e, 95)) if len(e) else None,
        'conditional_absrel_pct': float(100 * rel.mean()) if len(rel) else None,
        'abs_bins_all_pct': (100 * np.histogram(e, [0, .005, .01, .02, .05,
            .1, .2, np.inf])[0] / denominator).tolist() if denominator else None,
        'rel_bins_numeric_pct': (100 * np.histogram(rel, [0, .005, .01, .02,
            .03, .05, .1, np.inf])[0] / len(rel)).tolist() if len(rel) else None,
        'no_output_reasons': dict(Counter(r['reason'] for r in rows if r['prediction_m'] is None)),
        'failed_initialization_frames': denominator - len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--installations', type=int, default=99)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    root = args.evaluation
    audit = json.loads((root / 'installation-audit.json').read_text())
    rows = [json.loads(line) for line in (root / 'records.jsonl').open()]
    if len(audit) != args.installations:
        raise ValueError('installation_count_mismatch')
    groups = defaultdict(list)
    for r in rows:
        scene = r['scenario'] + ('/' + r['variant'] if r['scenario'] == 'depth_failure' else '')
        groups[scene, r['route'], r['geometry']].append(r)
        groups[scene + '/' + r['geometry_source'], r['route'], r['geometry']].append(r)
        groups[scene + '/' + str(r['band']), r['route'], r['geometry']].append(r)
    scores = {}
    for scene in SCENES:
        scores[scene] = {}
        for route in ROUTES:
            geometry = 'empty_fitted' if route == 'diffuse_v40_naive' else 'estimated'
            key = (scene, route, geometry)
            if key not in groups:
                continue
            denominator = (66 if route == 'diffuse_v40_naive' else args.installations) * args.repeats
            scores[scene][route] = stats(groups[key], denominator)
            for source in ('empty_rgbd_plus_operator_metric_anchor',
                           'empty_rim_plus_independent_metric_and_gravity'):
                sub = groups.get((scene + '/' + source, route, geometry), [])
                if sub:
                    scores[scene][route][source] = stats(sub, len(sub))
    bands = ['empty', '0.01-0.1', '0.1-1'] + [f'{j}-{j+1}' for j in range(1, 9)]
    band_scores = {}
    for scene in SCENES:
        band_scores[scene] = {}
        for i, band in enumerate(bands):
            band_scores[scene][band] = stats(
                groups.get((scene + '/' + str(i), 'adapted_v44', 'estimated'), []),
                9 * args.repeats)
    output = {
        'measurand': 'actual liquid height above interior bottom',
        'tolerance': 'max(5 mm, 2% actual liquid height)',
        'geometry': {
            'attempted_installations': len(audit),
            'initialized_installations': sum('initialization_error' not in a for a in audit),
            'source_counts': dict(Counter(a.get('geometry_mode', 'failed') for a in audit)),
            'failed_installations': [a for a in audit if 'initialization_error' in a],
            'source_by_band': {b: dict(Counter(a.get('geometry_mode', 'failed')
                for a in audit[i * 9:(i + 1) * 9])) for i, b in enumerate(bands)},
        },
        'scores': scores, 'adapted_by_depth_band': band_scores,
        'qualified_for_production': False,
    }
    (root / 'all-installation-analysis.json').write_text(json.dumps(output, indent=2))
    brief = {scene: {route: (round(s['full_numeric_coverage_pct'], 1),
                             None if s['conditional_mae_mm'] is None else round(s['conditional_mae_mm'], 1),
                             round(s['full_pass_pct'], 1))
                     for route, s in routes.items()} for scene, routes in scores.items()}
    print(json.dumps({'geometry': output['geometry']['source_counts'],
                      'scene_route_coverage_MAE_pass': brief}, indent=2))


if __name__ == '__main__':
    main()
