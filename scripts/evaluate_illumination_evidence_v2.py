"""Paired evidence ablation; simulator labels are loaded only for diagnostics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from analyze_illumination_v2 import score
from evaluate_shared_empty_v45 import install

from liquid_depth.illumination_surface_evidence import dry_changed_evidence
from liquid_depth.supported_scene_meter import FormalSceneMeter


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wet', type=Path, required=True)
    p.add_argument('--dry', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    base = [json.loads(line) for line in a.baseline.read_text().splitlines()]
    base = [r for r in base if r['route'] == 'merged_dry_geometry_v1']
    lookup = {(r['case'], r['slot'], r['repeat']): r for r in base}
    rows, audits = [], []
    for case in range(99):
        item = install(case, a.wet, a.dry, a.seed, source_slot_stride=18,
                       source_slot_offset=1, max_annotation_attempts=3)
        audits.append(item['audit'])
        fixed_roi = FormalSceneMeter.from_empty_geometry(
            item['bundle'], item['dry'], max_return_depth_m=10.).ordinary_container_roi
        for level in range(3):
            for offset in (1, 2):
                slot = (case*3+level)*6+offset
                folder = a.wet/f'{slot:05d}'
                frames = json.loads((folder/'frames.json').read_text())
                with np.load(folder/'geometry.npz') as g:
                    surface, target = g['mask'].astype(bool), g['target']
                for frame in frames:
                    raw = np.load(frame['raw_path'])
                    old = lookup[(case, slot, frame['repeat'])]
                    rows.append({**old, 'method': 'v1'})
                    for method in ('changed_layer',):
                        candidate = dry_changed_evidence(item['bundle'], raw, item['dry'], fixed_roi)
                        value = candidate['height_m']
                        consistent = surface & (raw > 0) & (abs(raw-target) <= np.maximum(.003, .01*target))
                        selected, layer = candidate.pop('selected_mask'), candidate.pop('layer_mask')
                        diagnostics = {
                            'all_nonzero': int((raw > 0).sum()),
                            'container_nonzero': int(selected.sum()),
                            'container_depth_consistent': int((consistent & selected).sum()),
                            'layer_depth_consistent': int((consistent & layer).sum()),
                            **candidate,
                        }
                        rows.append({**old, 'method': method,
                                     'prediction_m': value if value is not None else old['prediction_m'],
                                     'fallback_v1': value is None,
                                     'evidence': diagnostics})
        print('CASE', case+1, flush=True)
    groups = defaultdict(list)
    for r in rows:
        far = r['camera_to_bottom_m'] >= 6 and r['truth_m'] <= 1
        subset = 'deferred_far_shallow' if far else 'remaining'
        for group in (subset, subset+('/dry' if r['truth_m'] == 0 else '/wet')):
            groups[f"{r['scene']}|{r['level']}|{group}|{r['method']}"].append(r)
    summary = {k: score(v) for k, v in sorted(groups.items())}
    with (a.output/'records.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2))
    (a.output/'installation-audit.json').write_text(json.dumps(audits, indent=2))
    for key, value in summary.items():
        if '|remaining|' in key:
            print(key, value)


if __name__ == '__main__':
    main()
