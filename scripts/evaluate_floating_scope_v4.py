"""Replay unchanged held-out data through the application gate; labels only stratify reports."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from evaluate_visual_route_v1 import metrics

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.supported_scene_meter import FormalSceneMeter

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
OUT = BASE/'liquid-depth-artifacts/floating-peak-only-v4-20261015'
DATA = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'
MAPPING = {'ordinary': 'ordinary_diffuse', 'depth_failure': 'partial_depth_loss'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False, parents=True); torch.set_num_threads(4); cv2.setNumThreads(1)
    sources = [REPO/'src/liquid_depth/floating_peak_only_v4.py', REPO/'scripts/evaluate_floating_scope_v4.py',
        OLD/'pixel-model/best.pt', OLD/'region-model/verifier.joblib']
    protocol = {'sha256': {str(p): sha(p) for p in sources}, 'source_test_seed': 202610155927,
        'new_training': False, 'replay_of_previous_independent_dataset_not_new_independent_capture': True,
        'truth_used_only_after_measurement_for_reporting': True, 'visual_scope': 'floating_objects_highest_only'}
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    old_rows = [json.loads(line) for line in (OLD/'fresh-evaluation/records.jsonl').read_text().splitlines()]
    lookup = {(r['case'], r['repeat'], r['mode']): r for r in old_rows}
    predictor = None; model_calls = Counter(); factory_calls = Counter(); rows = []; audits = []
    changes = {'liquid_vs_geometry': 0, 'floating_peak_vs_previous_v3': 0, 'nonfloating_peak_vs_geometry': 0,
               'nonfloating_visual_executions': 0}
    for case in range(594, 792):
        folder = DATA/f'{case:05d}'; item = install(case, DATA, DATA/'dry', 202610155927,
            max_annotation_attempts=6, source_slot_stride=1); audits.append(item['audit'])
        base = FormalSceneMeter.from_empty_geometry(item['bundle'], item['dry'], max_return_depth_m=10.5)
        scene = MAPPING.get(item['meta']['scenario'], item['meta']['scenario'])
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=item['dry'])

        def factory(route=scene, calibrated=base, dry=item['dry']):
            nonlocal predictor
            factory_calls[route] += 1
            if predictor is None: predictor = MultiScaleEvidencePredictor(OLD/'pixel-model/best.pt', 'cuda')
            inner = OptionalFloatMeterV3(calibrated, dry, predictor, OLD/'region-model/verifier.joblib')
            class Counted:
                def measure(self, **kwargs):
                    model_calls[route] += 1; return inner.measure(**kwargs)
            return Counted()

        meter = FloatingPeakOnlyMeter(legacy, factory)
        rgb = cv2.imread(str(folder/'rgb.png')) if scene == 'floating_objects' else None
        for frame in json.loads((folder/'frames.json').read_text()):
            raw = np.load(frame['raw_path']); start = time.perf_counter()
            result = meter.measure(scene, depth_m=raw, rgb_bgr=rgb, visual_enabled=True)
            elapsed = 1000*(time.perf_counter()-start)
            control = lookup[(case, frame['repeat'], 'v3_off')]
            assert result['liquid_depth_m'] == control['liquid_depth_m']
            changes['liquid_vs_geometry'] += int(result['liquid_depth_m'] != control['liquid_depth_m'])
            if scene == 'floating_objects':
                previous = lookup[(case, frame['repeat'], 'v3_auto')]['highest_height_m']
                changes['floating_peak_vs_previous_v3'] += int(abs(result['highest_height_m']-previous) > 1e-6)
            else:
                changes['nonfloating_visual_executions'] += int(result['visual_module_executed'])
                changes['nonfloating_peak_vs_geometry'] += int(result['highest_height_m'] != control['highest_height_m'])
            # Neither this label nor the previous results participate in inference.
            labels = np.load(folder/'labels_offline.npz')['semantic']
            if scene == 'floating_objects':
                row = {**control, 'mode': 'scoped_float_peak_v4', 'liquid_depth_m': result['liquid_depth_m'],
                    'highest_height_m': result['highest_height_m'], 'algorithm_ms': elapsed,
                    'visual_module_executed': result['visual_module_executed'],
                    'visual_guidance_used': result['visual_guidance_used'],
                    'liquid_override_used': result['liquid_override_used'],
                    'actual_visible_floating_object': bool((labels == 3).any()),
                    'diagnostics': result['diagnostics']}
                rows.append(row)
        if case % 30 == 0: print('SCOPED_REPLAY', case, flush=True)
    assert all(value == 0 for value in changes.values()), changes
    summary = {}
    for subset in ['all_category_frames', 'actual_visible_floats', 'empty_control']:
        selected = [r for r in rows if subset == 'all_category_frames' or
            (r['actual_visible_floating_object'] if subset == 'actual_visible_floats' else not r['actual_visible_floating_object'])]
        summary[subset] = {}
        for mode in ['v3_off', 'v2_auto', 'scoped_float_peak_v4']:
            branch = selected if mode == 'scoped_float_peak_v4' else [lookup[(r['case'], r['repeat'], mode)] for r in selected]
            summary[subset][mode] = {'highest': metrics(branch, 'highest_height_m', True), 'liquid': metrics(branch)}
    bands = {}
    for band in sorted({r['band'] for r in rows}):
        group = [r for r in rows if r['band'] == band]
        bands[str(band)] = {'frames': len(group), 'true_liquid_min_m': min(r['truth']['liquid_height_m'] for r in group),
            'true_liquid_max_m': max(r['truth']['liquid_height_m'] for r in group),
            'highest': metrics(group, 'highest_height_m', True), 'liquid': metrics(group)}
    scope = {'changes': changes, 'model_calls_by_scene': dict(model_calls), 'model_factory_calls_by_scene': dict(factory_calls),
        'nonfloating_test_frames': 330, 'floating_category_test_frames': len(rows),
        'actual_visible_float_frames': sum(r['actual_visible_floating_object'] for r in rows),
        'opaque_float_only_not_transparent_compound_qualification': True}
    (OUT/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2))
    (OUT/'depth-strata.json').write_text(json.dumps(bands, indent=2))
    (OUT/'scope-audit.json').write_text(json.dumps(scope, indent=2))
    (OUT/'calibration-audit.json').write_text(json.dumps(audits, indent=2))
    assert all(sha(Path(name)) == digest for name, digest in protocol['sha256'].items())
    print('SCOPED_APPLICATION_COMPLETE', json.dumps(scope), flush=True)


if __name__ == '__main__': main()
