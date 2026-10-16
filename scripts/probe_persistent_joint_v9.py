"""Persistent false-region diagnostic on the known V8 six-pixel failure.

Additional noise exposures share one already-noisy base depth: a controlled
correlated probe, not independent hardware data or new benchmark installations.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3, sensor_sigma
from liquid_depth.floating_joint_v9 import FloatingJointMeterV9
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_rank_v8 import FloatingRankMeterV8
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp')
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
V8 = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'
OUT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-rank-v8-test-20261016'


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    item = install(303, ROOT, ROOT/'dry', 202610163811, max_annotation_attempts=6, source_slot_stride=1)
    base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
    inners = {'v4': OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib'),
        'v8_research': FloatingRankMeterV8(base, item['dry'], predictor, V8/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')}
    inners['v8_research'].policy_override = inners['v8_research'].bundle['research_policy_by_sensor'][base.bundle['sensor_family']]
    for mode in ['v9_single', 'v9']:
        inners[mode] = FloatingJointMeterV9(base, item['dry'], predictor, OUT/'model.joblib', V8/'model.joblib',
            OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib', temporal=mode == 'v9')
    routes = {n: FloatingPeakOnlyMeter(legacy, lambda b=inner: b) for n, inner in inners.items()}
    folder = ROOT/'00303'
    frame = json.loads((folder/'frames.json').read_text())[0]
    original = np.load(frame['raw_path'])
    valid = np.isfinite(original) & (original > 0)
    sigma = sensor_sigma(original, base.bundle)
    rgb = cv2.imread(str(folder/'rgb.png'))
    rng = np.random.default_rng(202610169431)
    rows = []
    for repeat in range(4):
        raw = original.copy()
        if repeat:
            raw[valid] += rng.normal(0, .25*sigma[valid])
        inners['v9'].set_frame_context('persistent_false_region_probe', repeat, repeat*.1)
        for name, route in routes.items():
            result = route.measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
            rows.append({'repeat': repeat, 'mode': name, 'highest_height_m': result['highest_height_m'],
                'truth': item['meta']['truth'], 'error_mm': 1000*abs(result['highest_height_m']-item['meta']['truth']['highest_height_m']),
                'liquid_depth_m': result['liquid_depth_m'], 'diagnostics': result['diagnostics']})
        assert len({r['liquid_depth_m'] for r in rows if r['repeat'] == repeat}) == 1
    dest = OUT/'persistent-region-probe'
    dest.mkdir(exist_ok=False)
    (dest/'protocol.json').write_text(json.dumps({'known_case': 303, 'repeat_base': 0, 'exposures': 4,
        'extra_noise_sigma_fraction': .25, 'seed': 202610169431, 'same_RGB': True,
        'correlated_base_depth': True, 'not_part_of_independent_test': True, 'not_used_to_tune_model': True}, indent=2))
    (dest/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    print('V9_PERSISTENT_PROBE_COMPLETE', json.dumps([{k: v for k, v in r.items() if k in ['repeat', 'mode', 'error_mm']} for r in rows]))


if __name__ == '__main__':
    main()
