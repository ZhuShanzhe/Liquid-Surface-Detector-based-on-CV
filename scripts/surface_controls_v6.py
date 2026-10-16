"""Shape/noise controls quantify smoothing benefits and loss of real sharp tops."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from liquid_depth.floating_surface_v6 import fit_observed_surface
from liquid_depth.liquid_height_metrics import HeightCalibration

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-surface-v6-20261016')


def main():
    rows = []; rng = np.random.default_rng(202610161507)
    for shape in ['flat', 'dome', 'ridge', 'narrow_spike']:
        for count in [16, 64, 256]:
            for noise in [.002, .01, .05]:
                for corruption in [0., .1, .3]:
                    for repeat in range(4):
                        uv = rng.uniform(-1, 1, (count, 2)); pixels = uv*20+50
                        if shape == 'flat': heights = np.full(count, .5); truth = .5
                        elif shape == 'dome': heights = .5+.3*(1-(uv*uv).sum(1)/2); truth = .8
                        elif shape == 'ridge': heights = .5+.3*(1-abs(uv[:, 0])); truth = .8
                        else: heights = .5+.3*np.exp(-np.sum(uv*uv, axis=1)/.015); truth = .8
                        raw = 3-heights+rng.normal(0, noise, count)
                        raw[rng.choice(count, int(count*corruption), replace=False)] -= .7
                        pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 3.
                        cal = HeightCalibration(np.array([[200., 0., 50.], [0., 200., 50.], [0., 0., 1.]]), pose, 0., (1., 1.))
                        result = fit_observed_surface(raw, pixels, np.ones(count), cal, 'tof')
                        rows.append({'shape': shape, 'points': count, 'injected_noise_m': noise, 'wrong_fraction': corruption,
                            'repeat': repeat, 'truth_m': truth, 'visible_sample_max_m': float(heights.max()),
                            'raw_q90_height_m': float(np.percentile(3-raw, 90)),
                            'surface_height_m': result.get('height_m'), 'model': result})
    groups = defaultdict(list)
    for r in rows:
        for key in [r['shape']+'|all', r['shape']+f"|wrong={r['wrong_fraction']}", r['shape']+f"|points={r['points']}"]:
            groups[key].append(r)
    summary = {}
    for key, values in groups.items():
        paired = [r for r in values if r['surface_height_m'] is not None]
        raw = [abs(r['raw_q90_height_m']-r['truth_m']) for r in paired]
        fitted = [abs(r['surface_height_m']-r['truth_m']) for r in paired]
        summary[key] = {'cases': len(values), 'valid_models': len(paired), 'paired_raw_q90_mae_mm': float(1000*np.mean(raw)),
            'paired_surface_mae_mm': float(1000*np.mean(fitted)), 'paired_surface_p95_mm': float(1000*np.percentile(fitted, 95)),
            'mean_missing_apex_in_original_sample_mm': float(1000*np.mean([r['truth_m']-r['visible_sample_max_m'] for r in values]))}
    (OUT/'controlled-shape-records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (OUT/'controlled-shape-summary.json').write_text(json.dumps(summary, indent=2))
    print('CONTROLS_COMPLETE', json.dumps({k: v for k, v in summary.items() if k.endswith('|all')}), flush=True)


if __name__ == '__main__': main()
