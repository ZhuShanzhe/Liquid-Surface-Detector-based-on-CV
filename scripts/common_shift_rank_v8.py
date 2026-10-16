"""Fixed common-mode shift probe, not part of primary independent-test accuracy.

Uses the first 33 installations from the fresh test, so it is a paired
perturbation diagnostic rather than 33 additional independent installations.
"""
import json
import time

import cv2
import numpy as np
import torch
from evaluate_rank_v8 import APEX, MODES, OLD, OUT, ROOT, SEED, summarize
from evaluate_shared_empty_v45 import install

from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_apex_v5 import FloatingApexMeterV5
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_rank_v8 import FloatingRankMeterV8
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.supported_scene_meter import FormalSceneMeter


def shift_depth(raw, sensor, bundle, sign):
    raw = raw.copy()
    valid = np.isfinite(raw) & (raw > 0)
    if sensor == 'tof':
        raw[valid] += sign*.01
    else:
        fx = height_calibration(bundle).intrinsics[0, 0]
        fb = fx*bundle['baseline_m']
        raw[valid] = fb/(fb/raw[valid]+sign*.015)
    assert np.isfinite(raw).all() and (raw[valid] > 0).all()
    return raw


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    rows = []
    for slot in range(33):
        case = 3+6*slot
        folder = ROOT/f'{case:05d}'
        item = install(case, ROOT, ROOT/'dry', SEED, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        sensor = base.bundle['sensor_family']
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])
        inners = {'v4': OptionalFloatMeterV3(base, item['dry'], predictor, OLD/'region-model/verifier.joblib'),
            'v5': FloatingApexMeterV5(base, item['dry'], predictor, APEX/'apex-verifier.joblib', OLD/'region-model/verifier.joblib')}
        for mode in MODES[2:]:
            inner = FloatingRankMeterV8(base, item['dry'], predictor, OUT/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
            if mode == 'v8_research':
                inner.policy_override = inner.bundle['research_policy_by_sensor'][sensor]
            elif mode == 'v8_no_rank':
                inner.policy_override = {**inner.bundle['research_policy_by_sensor'][sensor], 'gap_weight': 0.}
            inners[mode] = inner
        routes = {name: FloatingPeakOnlyMeter(legacy, lambda b=inner: b) for name, inner in inners.items()}
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            sign = 1 if (slot+int(frame['repeat'])) % 2 else -1
            raw = shift_depth(np.load(frame['raw_path']), sensor, base.bundle, sign)
            outputs = {}
            for mode in MODES:
                start = time.perf_counter()
                result = routes[mode].measure('floating_objects', depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
                outputs[mode] = result, 1000*(time.perf_counter()-start)
            assert len({v[0]['liquid_depth_m'] for v in outputs.values()}) == 1
            has_float = bool((np.load(folder/'labels_offline.npz')['semantic'] == 3).any())
            for mode, (result, elapsed) in outputs.items():
                rows.append({'case': case, 'repeat': int(frame['repeat']), 'mode': mode, 'sensor': sensor,
                    'band': item['meta']['band'], 'truth': item['meta']['truth'], 'actual_visible_floating_object': has_float,
                    'highest_height_m': result['highest_height_m'], 'liquid_depth_m': result['liquid_depth_m'],
                    'algorithm_ms': elapsed, 'shift_sign': sign, 'diagnostics': result['diagnostics']})
    dest = OUT/'common-shift-probe'
    dest.mkdir(exist_ok=False)
    (dest/'protocol.json').write_text(json.dumps({'installations_reused_from_main': 33, 'frames': 66,
        'tof_shift_m': .01, 'stereo_disparity_shift_px': .015, 'alternating_signs': True,
        'no_range_clipping_or_frame_exclusion': True, 'not_independent_additional_installations': True,
        'training_did_not_include_this_explicit_shift_probe': True}, indent=2))
    (dest/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    (dest/'summary.json').write_text(json.dumps(summarize(rows), indent=2))
    print('V8_COMMON_SHIFT_PROBE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
