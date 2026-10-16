"""Existing depth/loss grid replay through the unchanged determined facade."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from evaluate_determined_v1 import summarize
from evaluate_shared_empty_v45 import install
from evaluate_uniform_loss_v48 import LEVELS, delete_to_actual_loss

from liquid_depth.determined_scene_meter import DeterminedSceneMeter
from liquid_depth.liquid_height_metrics import HeightCalibration, reference_statistics

BASE = Path('/root/autodl-tmp')
SOURCE = BASE/'liquid-depth-artifacts/evaluation/supported-three-scenes-v3-independent-20261014'
OUT = BASE/'liquid-depth-artifacts/determined-all-scenes-v1-20261016/depth-grid'


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True, exist_ok=False)
    source = json.loads((SOURCE/'summary.json').read_text())['protocol']
    wet, dry = Path(source['wet_root']), Path(source['dry_root'])
    ordinary, partial, exclusions, audits = [], [], [], []
    for case in range(source['ordinary_cases']):
        folder = wet/f'{case*6+5:05d}'
        meta = json.loads((folder/'metadata.json').read_text())
        cal = replace(HeightCalibration.from_simulation(meta), bottom_z_m=float(meta['interior_bottom_z_m']))
        truth = float(reference_statistics(meta, cal)['mean_m'])
        distance = float(abs(cal.camera_to_world_cv[2, 3]-cal.bottom_z_m))
        common = {'case': case, 'sensor': meta['sensor_family'], 'band': meta['band'],
                  'camera_to_bottom_m': distance, 'truth': {'liquid_height_m': truth},
                  'cohort': 'native_and_controlled_loss_202610141101'}
        try:
            item = install(case, wet, dry, source['seed'])
            audits.append(item['audit'])
            meter = DeterminedSceneMeter.from_empty_geometry(item['bundle'], item['dry'])
        except (ValueError, FileNotFoundError) as exc:
            exclusions.append({'case': case, 'reason': 'input_geometry:'+str(exc), 'frames': source['repeats']})
            ordinary.extend({**common, 'scene': 'ordinary_diffuse', 'repeat': rep, 'liquid_depth_m': None,
                             'diagnostics': {'input_error': str(exc)}} for rep in range(source['repeats']))
            continue
        with np.load(folder/'geometry.npz') as geometry:
            surface = geometry['mask'].astype(bool)
        for rep in range(source['repeats']):
            raw = np.load(folder/f'raw_native_{rep:02d}.npy', allow_pickle=False)
            result = meter.measure('ordinary_diffuse', depth_m=raw)
            ordinary.append({**common, 'scene': 'ordinary_diffuse', 'repeat': rep,
                             'liquid_depth_m': result['liquid_depth_m'], 'diagnostics': result['diagnostics']})
            loss = 100*(1-int((surface & (raw > 0)).sum())/max(int(surface.sum()), 1))
            if loss > 50:
                exclusions.append({'case': case, 'repeat': rep, 'reason': 'native_loss_above_50pct', 'actual_loss_pct': loss})
                continue
            samples = [('native', loss, raw)]
            for target in LEVELS[1:]:
                if target+.001 >= loss:
                    samples.append(('controlled', float(target), delete_to_actual_loss(
                        raw, surface, target, source['seed']+case*10007+rep*509)))
            for kind, actual, frame in samples:
                result = meter.measure('partial_depth_loss', depth_m=frame)
                partial.append({**common, 'scene': 'partial_depth_loss', 'repeat': rep, 'kind': kind,
                                'actual_loss_pct': actual, 'liquid_depth_m': result['liquid_depth_m'],
                                'diagnostics': result['diagnostics']})
        if case % 15 == 0:
            print('DEPTH_GRID', case+1, 99, flush=True)
    rows = ordinary+partial
    (OUT/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    (OUT/'summary.json').write_text(json.dumps(summarize(rows), indent=2))
    loss_groups = {}
    for start in [0, 10, 20, 30, 40]:
        group = [r for r in partial if start <= r['actual_loss_pct'] < start+10 or
                 start == 40 and r['actual_loss_pct'] == 50]
        loss_groups[str(start)+'-'+str(start+10)] = summarize(group)
    (OUT/'loss-strata.json').write_text(json.dumps(loss_groups, indent=2))
    (OUT/'input-audit.json').write_text(json.dumps({'source': str(SOURCE), 'source_protocol': source,
        'exclusions': exclusions, 'calibration_audits': audits,
        'no_input_repair_per_current_scope': True, 'ordinary_initialization_errors_in_all_attempted_denominator': True,
        'partial_precision_is_conditional_on_valid_calibration_and_0_50pct_scope': True,
        'current_wrapper_raw_profile_m': 10.5, 'old_source_profile_m': 10.0,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
    print('DETERMINED_DEPTH_GRID_COMPLETE', len(ordinary), len(partial), len(exclusions), flush=True)


if __name__ == '__main__':
    main()
