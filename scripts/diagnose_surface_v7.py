"""Post-test failure inspection; labels cannot alter the frozen inference."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor
from liquid_depth.floating_surface_v7 import FloatingSurfaceMeterV7, model_allowed
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp')
OUT = BASE/'liquid-depth-artifacts/floating-surface-v7-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-surface-v7-test-20261016'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
APEX = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'


def main():
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    rows = [json.loads(x) for x in (OUT/'independent-test/records.jsonl').read_text().splitlines()]
    wet = [r for r in rows if r['mode'] == 'v7' and r['actual_visible_floating_object']]
    error = lambda r: abs(r['highest_height_m']-r['truth']['highest_height_m'])
    adopted = [r for r in wet if r['diagnostics']['floating_peak_auxiliary']['v7_used']]
    selected = {(r['case'], r['repeat']): r for r in sorted(wet, key=error, reverse=True)[:6]+sorted(adopted, key=error, reverse=True)[:4]}
    predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
    report = []
    for (case, repeat), row in selected.items():
        item = install(case, ROOT, ROOT/'dry', 202610162711, max_annotation_attempts=6, source_slot_stride=1)
        folder = ROOT/f'{case:05d}'
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        meter = FloatingSurfaceMeterV7(base, item['dry'], predictor, OUT/'model.joblib', OLD/'region-model/verifier.joblib', APEX/'apex-verifier.joblib')
        frame = next(f for f in json.loads((folder/'frames.json').read_text()) if int(f['repeat']) == repeat)
        result = meter.measure(depth_m=np.load(frame['raw_path']), rgb_bgr=cv2.imread(str(folder/'rgb.png')))
        # Only now inspect simulator supervision, never for candidate selection.
        labels = np.load(folder/'labels_offline.npz')
        sem = labels['semantic']
        heights = world_points(labels['float_depth_m'], item['true'])[..., 2]-item['true'].bottom_z_m
        _, components = cv2.connectedComponents((sem == 3).astype(np.uint8))
        component_tops = {i: float(heights[components == i].max()) for i in range(1, int(components.max())+1)}
        origin = np.load(folder/f'origin_offline_{repeat:02d}.npy')
        policy = meter.bundle['policy_by_sensor'][base.bundle['sensor_family']]
        models = meter._last_models
        candidate_audit = []
        ranked = sorted(meter._last_candidates, key=lambda c: -c['object_score']*np.sqrt(c['pixels']))
        for rank, candidate in enumerate(ranked):
            use = candidate['mask'] & (sem == 3)
            candidate_audit.append({'rank': rank, 'pixels': candidate['pixels'], 'object_score': candidate['object_score'],
                'scheduled_for_fit': rank < 28 and candidate['pixels'] >= 12 and candidate['object_score'] >= .3,
                'offline_float_echo_fraction': float((origin[candidate['mask']] == 2).mean()),
                'offline_float_pixel_fraction': float((sem[candidate['mask']] == 3).mean()),
                'offline_candidate_visible_top_m': float(heights[use].max()) if use.any() else None})
        scores = meter.bundle['classifier'].predict_proba(np.stack([m['quality_features'] for m in models]))[:, 1] if models else []
        details = []
        for m, score in zip(models, scores, strict=True):
            mask = m['mask']
            component_ids, counts = np.unique(components[mask], return_counts=True)
            matched = max([(int(i), int(n)) for i, n in zip(component_ids, counts, strict=True) if i], key=lambda p: p[1], default=(0, 0))[0]
            float_pixels = mask & (sem == 3)
            details.append({'kind': m['kind'], 'height_m': m['height_m'], 'pixels': m['pixels'],
                'object_score': m['object_score'], 'quality_score': float(score), 'model': m['model'],
                'passes_model_guard': model_allowed(m['model'], max_cv=policy['max_cv'], min_cells=policy['min_cells']),
                'offline_float_pixel_fraction': float((sem[mask] == 3).mean()),
                'offline_float_echo_fraction': float((origin[mask] == 2).mean()),
                'offline_matched_component_visible_top_m': component_tops.get(matched),
                'offline_candidate_visible_top_m': float(heights[float_pixels].max()) if float_pixels.any() else None})
        report.append({'case': case, 'repeat': repeat, 'sensor': row['sensor'], 'band': row['band'],
            'truth': row['truth'], 'output_m': row['highest_height_m'], 'error_mm': 1000*error(row),
            'candidate_count': len(meter._last_candidates), 'fitted_model_count': len(models),
            'v7_used': result['diagnostics']['v7_used'],
            'offline_all_visible_top_m': float(heights[sem == 3].max()),
            'candidate_audit': candidate_audit, 'models': details})
    (OUT/'independent-test/worst-frame-diagnosis.json').write_text(json.dumps(report, indent=2))
    print('V7_WORST_FRAME_DIAGNOSIS_COMPLETE', len(report))


if __name__ == '__main__':
    main()
