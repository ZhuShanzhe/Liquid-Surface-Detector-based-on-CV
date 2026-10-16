"""Locked fresh all-frame paired evaluation: previous auto, new off/auto, true geometric peak."""
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_adaptive_route_v2 import object_counts
from evaluate_shared_empty_v45 import install
from evaluate_visual_route_v1 import metrics

from liquid_depth.adaptive_rgbd_meter import AdaptiveRGBDMeter
from liquid_depth.float_evidence_v3 import (
    MultiScaleEvidencePredictor,
    OptionalFloatMeterV3,
    choose_peak,
    region_candidates,
    score_regions,
)
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
PREVIOUS = BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'
DATA = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'


class LastPrediction:
    def __init__(self, predictor): self.predictor = predictor; self.last = None
    def predict(self, features):
        self.last = self.predictor.predict(features); return self.last


def stratum_labels(row):
    return ['all', f"sensor={row['sensor']}", f"band={row['band']}"]


def used_float_region(mode, value):
    key = 'verified_regions' if mode == 'v3_auto' else 'float_components'
    return bool(value['diagnostics'].get(key))


def main():
    torch.set_num_threads(4); cv2.setNumThreads(1)
    frozen = json.loads((OUT/'frozen-test-protocol.json').read_text())
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest for name, digest in frozen['sha256'].items())
    output = OUT/'fresh-evaluation'; output.mkdir(exist_ok=False)
    oldpred = LastPrediction(RGBDEvidencePredictor(PREVIOUS/'pixel-model-with-real/best.pt', 'cuda'))
    newpred = LastPrediction(MultiScaleEvidencePredictor(OUT/'pixel-model/best.pt', 'cuda'))
    rows = []; objects = defaultdict(lambda: defaultdict(lambda: [0, 0])); objprecision = defaultdict(lambda: [0, 0])
    false_regions = defaultdict(lambda: [0, 0]); changed_liquid = 0; calibration = []
    for case in range(594, 792):
        folder = DATA/f'{case:05d}'; item = install(case, DATA, DATA/'dry', 202610155927,
            max_annotation_attempts=6, source_slot_stride=1)
        calibration.append(item['audit'])
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        oldmeter = AdaptiveRGBDMeter(base, dry_depth_m=item['dry'], predictor=oldpred,
                                    gate_checkpoint=PREVIOUS/'candidate-model-with-real/gate.pt')
        meter = OptionalFloatMeterV3(base, item['dry'], newpred, OUT/'region-model/verifier.joblib')
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); results = {}
            for name in ['v2_auto', 'v3_off', 'v3_auto']:
                start = time.perf_counter()
                if name == 'v2_auto': value = oldmeter.measure(rgb_bgr=rgb, depth_m=raw, mode='auto')
                else: value = meter.measure(depth_m=raw, rgb_bgr=rgb, visual_mode='off' if name == 'v3_off' else 'auto')
                results[name] = (value, (time.perf_counter()-start)*1000)
            changed_liquid += results['v2_auto'][0]['liquid_depth_m'] != results['v3_auto'][0]['liquid_depth_m']
            assert results['v3_off'][0]['liquid_depth_m'] == results['v3_auto'][0]['liquid_depth_m']
            # Reconstruct accepted ORIGINAL pixel masks, before opening test labels.
            candidates = region_candidates(base, item['dry'], rgb, raw, newpred.last, results['v3_auto'][0]['liquid_depth_m'])
            scored = score_regions(candidates, meter.verifier); _, accepted = choose_peak(scored, meter.verifier['policy'])
            final_mask = np.zeros(raw.shape, bool)
            for candidate in accepted: final_mask |= candidate['mask']
            label = np.load(folder/'labels_offline.npz')['semantic']
            for name, pred in [('v2_semantic', oldpred.last['semantic'].argmax(0) == 3),
                               ('v3_semantic', newpred.last['semantic'].argmax(0) == 3), ('v3_accepted', final_mask)]:
                counts, n, matched = object_counts(label == 3, pred)
                objprecision[name][0] += n; objprecision[name][1] += matched
                for key, (total, hit) in counts.items(): objects[name][key][0] += total; objects[name][key][1] += hit
            if not (label == 3).any():
                for name, (value, _) in results.items():
                    if name == 'v3_off': continue
                    key = item['meta']['scenario']+'|'+name; false_regions[key][0] += 1
                    false_regions[key][1] += used_float_region(name, value)
            truth = item['meta']['truth']; far = item['meta']['camera_to_bottom_m'] >= 6 and truth['liquid_height_m'] <= 1
            for mode, (value, elapsed) in results.items():
                rows.append({'case': case, 'repeat': frame['repeat'], 'mode': mode,
                    'scene': item['meta']['scenario'], 'sensor': item['meta']['sensor_family'], 'band': item['meta']['band'],
                    'truth': truth, 'liquid_depth_m': value['liquid_depth_m'], 'highest_height_m': value['highest_height_m'],
                    'numeric_output': value['numeric_output'], 'visual_guidance_used': value['visual_guidance_used'],
                    'liquid_override_used': value.get('liquid_override_used', False), 'algorithm_ms': elapsed,
                    'far_shallow_deferred': far, 'camera_to_bottom_m': item['meta']['camera_to_bottom_m'],
                    'floating_pixels_offline': int((label == 3).sum()), 'diagnostics': value['diagnostics']})
        if case % 20 == 0: print('TESTED', case, flush=True)
    groups = defaultdict(list)
    for row in rows:
        for subset in ['all_frames', 'far_shallow' if row['far_shallow_deferred'] else 'remaining']:
            for stratum in stratum_labels(row):
                groups[f"{row['scene']}|{stratum}|{subset}|{row['mode']}"].append(row)
    (output/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    for filename, highest in [('liquid-summary', False), ('highest-summary', True)]:
        (output/f'{filename}.json').write_text(json.dumps({k: metrics(v, 'highest_height_m', True) if highest else metrics(v)
            for k, v in groups.items()}, indent=2))
    stat = {'changed_liquid_frames': changed_liquid, 'all_frames_retained': True,
        'object_recall': {name: {key: {'objects': n, 'hits': hit, 'recall': hit/max(n, 1)} for key, (n, hit) in table.items()}
                          for name, table in objects.items()},
        'object_precision': {name: {'components': n, 'matched': hit, 'precision': hit/max(n, 1)} for name, (n, hit) in objprecision.items()},
        'false_regions_no_visible_float': {name: {'frames': n, 'false_region_frames': count} for name, (n, count) in false_regions.items()},
        'latency_ms': {name: {key: float(np.percentile([r['algorithm_ms'] for r in rows if r['mode'] == name], q))
                             for key, q in [('median', 50), ('p95', 95), ('max', 100)]} for name in ['v2_auto', 'v3_off', 'v3_auto']}}
    (output/'diagnostics-summary.json').write_text(json.dumps(stat, indent=2))
    (output/'calibration-audit.json').write_text(json.dumps(calibration, indent=2))
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest for name, digest in frozen['sha256'].items())
    print('FRESH_TEST_COMPLETE', json.dumps(stat), flush=True)


if __name__ == '__main__': main()
