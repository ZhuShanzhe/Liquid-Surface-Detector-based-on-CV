"""Reporting-only reconciliation and fair V2/V3 accepted-object comparison."""
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_adaptive_route_v2 import object_counts
from evaluate_float_v3 import used_float_region
from evaluate_shared_empty_v45 import install

from liquid_depth.adaptive_rgbd_meter import build_observed_candidates, supported_float_height
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor, observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
DATA = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'


def main():
    torch.set_num_threads(4); cv2.setNumThreads(1)
    p = OUT/'fresh-evaluation'; protocol_file = OUT/'frozen-test-protocol.json'
    protocol = json.loads(protocol_file.read_text()); changed = str(REPO/'scripts/evaluate_float_v3.py')
    for name, digest in protocol['sha256'].items():
        if name != changed: assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    shutil.copyfile(protocol_file, OUT/'frozen-test-protocol-before-region-report-repair.json')
    protocol['sha256'][changed] = hashlib.sha256(Path(changed).read_bytes()).hexdigest()
    protocol['posthoc_reporting_repair'] = 'V3 verified_regions must not be shadowed by inherited float_components=0; no measured value/model/policy changed'
    protocol_file.write_text(json.dumps(protocol, indent=2))
    rows = [json.loads(line) for line in (p/'records.jsonl').read_text().splitlines()]
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        if row['mode'] == 'v3_off' or row['floating_pixels_offline'] != 0: continue
        key = row['scene']+'|'+row['mode']; counts[key][0] += 1
        counts[key][1] += used_float_region(row['mode'], row)
    path = p/'diagnostics-summary.json'; diagnostics = json.loads(path.read_text())
    shutil.copyfile(path, p/'diagnostics-summary-before-region-report-repair.json')
    diagnostics['false_regions_no_visible_float'] = {key: {'frames': n, 'false_region_frames': count} for key, (n, count) in counts.items()}
    diagnostics['report_reconciliation'] = 'Only false-region counters recomputed from unchanged full per-frame records.'
    path.write_text(json.dumps(diagnostics, indent=2))
    predictor = RGBDEvidencePredictor(BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015/pixel-model-with-real/best.pt', 'cuda')
    objects = defaultdict(lambda: [0, 0]); n_components = 0; n_matched = 0
    for case in range(594, 792):
        folder = DATA/f'{case:05d}'; item = install(case, DATA, DATA/'dry', 202610155927, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); prediction = predictor.predict(observed_features(rgb, raw, item['dry'], base.ordinary_container_roi, item['bundle']))
            _, context = build_observed_candidates(base, raw, prediction); context['calibration'] = height_calibration(item['bundle'])
            _, accepted, mask = supported_float_height(prediction, context)
            boxes = {tuple(c['bbox_xyxy_px']) for c in accepted}
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8)); accepted_mask = np.zeros(mask.shape, bool)
            for label in range(1, n):
                x, y, w, h, _ = stats[label]
                if (int(x), int(y), int(x+w), int(y+h)) in boxes: accepted_mask |= labels == label
            gt = np.load(folder/'labels_offline.npz')['semantic'] == 3
            table, number, matched = object_counts(gt, accepted_mask); n_components += number; n_matched += matched
            for key, (number, hit) in table.items(): objects[key][0] += number; objects[key][1] += hit
    result = {'v2_accepted': {'recall': {k: {'objects': n, 'hits': hit, 'recall': hit/max(n, 1)} for k, (n, hit) in objects.items()},
            'components': n_components, 'matched': n_matched, 'precision': n_matched/max(n_components, 1)},
        'v3_accepted': {'recall': diagnostics['object_recall']['v3_accepted'], **diagnostics['object_precision']['v3_accepted']},
        'metric': 'visible connected components, equivalent-area pixel diameter, matching IoU >= 0.1; both use final accepted original-pixel regions'}
    (p/'accepted-object-comparison.json').write_text(json.dumps(result, indent=2))
    print('ACCEPTED_OBJECT_AUDIT', json.dumps(result), flush=True)


if __name__ == '__main__': main()
