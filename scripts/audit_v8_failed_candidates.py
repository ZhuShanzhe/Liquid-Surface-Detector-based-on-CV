"""Post-measurement truth audit on fixed V8 failures, never used for inference."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor
from liquid_depth.floating_rank_v8 import FloatingRankMeterV8
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp')
OUT = BASE/'liquid-depth-artifacts/floating-v8-restored-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-joint-v9-test-20261016'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
MODEL = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016/model.joblib'


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    analysis = json.loads((OUT/'analysis-and-restoration.json').read_text())
    failures = [f for f in analysis['failures'] if f['sensor'] == 'active_stereo']
    result = []
    for case in sorted({f['case'] for f in failures}):
        item = install(case, ROOT, ROOT/'dry', 202610164913, max_annotation_attempts=6, source_slot_stride=1)
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        meter = FloatingRankMeterV8(base, item['dry'], predictor, MODEL, OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
        folder = ROOT/f'{case:05d}'
        rgb = cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            if not any(f['case'] == case and f['repeat'] == frame['repeat'] for f in failures):
                continue
            # Model measurement completes before loading any truth labels.
            meter.measure(depth_m=np.load(frame['raw_path']), rgb_bgr=rgb)
            labels = np.load(folder/'labels_offline.npz')
            origin = np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
            _, cc = cv2.connectedComponents((labels['semantic'] == 3).astype(np.uint8))
            h = world_points(labels['float_depth_m'], item['true'])[..., 2]-item['true'].bottom_z_m
            tops = {i: float(h[cc == i].max()) for i in range(1, int(cc.max())+1)}
            visible_top = max(tops.values(), default=0.)
            policy = meter.bundle['policy_by_sensor']['active_stereo']
            highest = []
            for c in meter._last_scored:
                ids, counts = np.unique(cc[c['mask']], return_counts=True)
                matched, n = max([(int(i), int(n)) for i, n in zip(ids, counts, strict=True) if i], key=lambda x: x[1], default=(0, 0))
                physical = n/max(c['pixels'], 1) >= .6 and (origin[c['mask']] == 2).mean() >= .6
                if not physical or visible_top-tops[matched] > .020:
                    continue
                reject = []
                for name, flag in [('pixels', c['pixels'] < policy['min_pixels']), ('cells', c['cells'] < policy['min_cells']),
                                   ('identity', c['physical_score'] < policy['physical_threshold']),
                                   ('risk', c['risk_m'] > policy['max_risk_m'])]:
                    if flag:
                        reject.append(name)
                highest.append({'pixels': c['pixels'], 'cells': int(c['cells']), 'identity': c['physical_score'],
                                'risk_mm': 1000*c['risk_m'], 'height_mm': 1000*c['height_m'],
                                'correction_mm': 1000*c['learned_correction_m'],
                                'utility_mm': 1000*(c['height_m']-.5*c['risk_m']), 'rejected_by': reject})
            result.append({'case': case, 'repeat': frame['repeat'], 'all_candidates': len(meter._last_scored),
                           'highest_candidates': highest, 'highest_present': bool(highest),
                           'highest_eligible': any(not c['rejected_by'] for c in highest)})
    (OUT/'failed-candidate-audit.json').write_text(json.dumps(result, indent=2))
    counts = {'active_stereo_failed_frames': len(result),
              'no_physical_highest_candidate': sum(not r['highest_present'] for r in result),
              'highest_present_but_all_rejected': sum(r['highest_present'] and not r['highest_eligible'] for r in result),
              'highest_eligible_but_final_error': sum(r['highest_eligible'] for r in result),
              'labels_used_after_measurement_only': True}
    (OUT/'failed-candidate-summary.json').write_text(json.dumps(counts, indent=2))
    print('HIGHEST_CANDIDATE_AUDIT', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
