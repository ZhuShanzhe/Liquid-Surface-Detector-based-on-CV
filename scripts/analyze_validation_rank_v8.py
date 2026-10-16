"""Validation-only head diagnostics, separate from end-to-end frame accuracy."""
import json
from pathlib import Path

import joblib
import numpy as np

OUT = Path('/root/autodl-tmp/liquid-depth-artifacts/floating-rank-v8-20261016')


def main():
    rows = joblib.load(OUT/'validation-scored.joblib')
    summary = {}
    for sensor in ['all', 'tof', 'active_stereo', 'structured_light']:
        selected = [r for r in rows if sensor == 'all' or r['sensor'] == sensor]
        wet = [r for r in selected if r['has_float']]
        pairs = [(c, s) for r in selected for c, s in zip(r['candidates'], r['scored'], strict=True) if c['physical_target']]
        raw_errors = [abs(c['height_raw_m']-c['component_top'])*1000 for c, _ in pairs]
        corrected_errors = [abs(s['height_m']-c['component_top'])*1000 for c, s in pairs]
        summary[sensor] = {'frames': len(selected), 'wet_frames': len(wet), 'physical_candidate_count': len(pairs),
            'original_has_highest_candidate_frames': sum(any(c['physical_target'] and c['top_gap'] <= .02 and not c.get('relaxed', False) for c in r['candidates']) for r in wet),
            'expanded_has_highest_candidate_frames': sum(any(c['physical_target'] and c['top_gap'] <= .02 for c in r['candidates']) for r in wet),
            'raw_component_candidate_mae_mm': float(np.mean(raw_errors)), 'corrected_component_candidate_mae_mm': float(np.mean(corrected_errors)),
            'raw_component_candidate_p95_mm': float(np.percentile(raw_errors, 95)), 'corrected_component_candidate_p95_mm': float(np.percentile(corrected_errors, 95)),
            'risk_empirical_coverage_pct': 100*float(np.mean([e <= s['risk_m']*1000 for e, (_, s) in zip(corrected_errors, pairs, strict=True)]))}
    (OUT/'validation-head-diagnostics.json').write_text(json.dumps(summary, indent=2))
    print('V8_VALIDATION_HEAD_DIAGNOSTICS', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
