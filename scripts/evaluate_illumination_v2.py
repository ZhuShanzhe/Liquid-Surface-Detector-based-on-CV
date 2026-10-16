"""All-frame V1/V2/causal V2 comparison, with a deferred far/shallow register."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from analyze_illumination_v2 import score
from evaluate_shared_empty_v45 import install

from liquid_depth.illumination_scene_meter import IlluminationSceneMeter


def main():
    p = argparse.ArgumentParser()
    for name in ('wet', 'dry', 'baseline', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    baseline = [json.loads(s) for s in a.baseline.read_text().splitlines()]
    baseline = [r for r in baseline if r['route'] == 'merged_dry_geometry_v1']
    lookup = {(r['case'], r['slot'], r['repeat']): r for r in baseline}
    items, audits = {}, []
    # Complete calibration preflight before any scoring.
    for case in range(99):
        item = install(case, a.wet, a.dry, a.seed, source_slot_stride=18,
                       source_slot_offset=1, max_annotation_attempts=3)
        audits.append(item['audit'])
        profile = ({'disparity_step_px': 1/16, 'disparity_noise_px': .26}
                   if item['meta']['sensor_family'] == 'structured_light' else None)
        item['meter'] = IlluminationSceneMeter.from_empty_geometry(
            item['bundle'], item['dry'], quantized_disparity_profile=profile,
            max_return_depth_m=10., refinement_enabled=True)
        items[case] = item
    (a.output/'installation-audit.json').write_text(json.dumps(audits, indent=2))
    rows, deferred = [], []
    timings = defaultdict(list)
    for case, item in items.items():
        for level in range(3):
            for offset, scene in ((1, 'low_light'), (2, 'glare')):
                slot = (case*3+level)*6+offset
                folder = a.wet/f'{slot:05d}'
                rgb = cv2.imread(str(folder/'rgb.png'))
                frames = json.loads((folder/'frames.json').read_text())
                history = []
                for frame in frames:
                    raw = np.load(frame['raw_path'])
                    old = lookup[(case, slot, frame['repeat'])]
                    rows.append({**old, 'method': 'v1'})
                    for method, previous in (('v2_single', []), ('v2_causal4', history[-3:])):
                        start = time.perf_counter()
                        result = item['meter'].measure(scene, rgb_bgr=rgb, depth_m=raw,
                            recent_depth_frames=previous, stationary_confirmed=True)
                        timings[method].append(time.perf_counter()-start)
                        rows.append({**old, 'method': method,
                                     'prediction_m': result['liquid_depth_m'],
                                     'diagnostics': result['diagnostics']})
                    history.append(raw)
        print('CASE', case+1, flush=True)
    groups = defaultdict(list)
    for r in rows:
        far = r['camera_to_bottom_m'] >= 6 and r['truth_m'] <= 1
        subset = 'deferred_far_shallow' if far else 'remaining'
        if far:
            deferred.append(r)
        for sub in (subset, subset+('/dry' if r['truth_m'] == 0 else '/wet'),
                    subset+'/'+r['band'], subset+'/'+r['sensor']):
            groups[f"{r['scene']}|{r['level']}|{sub}|{r['method']}"].append(r)
    summary = {k: score(v) for k, v in sorted(groups.items())}
    for name, data in (('records.jsonl', rows), ('deferred_far_shallow_records.jsonl', deferred)):
        with (a.output/name).open('w') as f:
            for r in data:
                f.write(json.dumps(r, allow_nan=False)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    sources = [Path(__file__), Path('src/liquid_depth/illumination_scene_meter.py'),
               Path('src/liquid_depth/illumination_surface_evidence.py')]
    (a.output/'protocol.json').write_text(json.dumps({
        'seed': a.seed, 'installation_count': len(items), 'frames_per_method': len(baseline),
        'deferred_definition': 'camera_to_bottom>=6m and true_height<=1m, including dry',
        'far_shallow_used_for_policy_selection': False,
        'causal_history_max_frames': 3, 'startup_frames_scored': True,
        'stationary_surface_and_camera_confirmed_in_simulation': True,
        'no_output_gate_no_zero_snap_no_clipping': True,
        'height_tolerance': 'max(20mm,2% true bottom-relative height)',
        'algorithm_only_time_ms': {k: {'median': float(1000*np.median(v)),
                                      'p95': float(1000*np.percentile(v, 95))}
                                   for k, v in timings.items()},
        'source_sha256': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
    }, indent=2))
    for k, v in summary.items():
        if '|remaining|' in k:
            print(k, v)


if __name__ == '__main__':
    main()
