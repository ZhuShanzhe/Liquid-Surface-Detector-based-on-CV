"""Evaluator-only cause analysis; oracle surface pixels never enter deployment."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from evaluate_shared_empty_v45 import install
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.scene_route_policy_v45 import surface_candidate
from liquid_depth.surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--wet-root',type=Path,required=True)
    p.add_argument('--dry-root',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    old={(r['case'],r['repeat']):r for r in map(json.loads,a.records.open())
         if r['target_loss_pct']==0 and r['sensor']=='structured_light'}
    rows=[]
    for case in sorted({key[0] for key in old}):
        item=install(case,a.wet_root,a.dry_root,a.seed)
        folder=a.wet_root/f'{case*6+5:05d}'
        meta=json.loads((folder/'metadata.json').read_text())
        rgb=cv2.imread(str(folder/'rgb.png'))
        with np.load(folder/'geometry.npz') as z:
            visible=z['mask'].astype(bool)
        cal=replace(HeightCalibration.from_simulation(meta),
                    bottom_z_m=float(meta['interior_bottom_z_m']))
        estimator=RefinedSurfaceEstimator(mode='sensor',
                    stereo_noise=StereoNoiseModel.simulation_proxy('structured_light'))
        balanced=RefinedSurfaceEstimator(mode='balanced')
        for frame in json.loads((folder/'frames.json').read_text()):
            rep=frame['repeat']
            if (case,rep) not in old:
                continue
            raw=np.load(frame['raw_path'])
            fitted=surface_candidate(item['bundle'],rgb,raw,visible,scene='depth_failure')
            truth_geometry=estimator.estimate(rgb,raw,{'mask':visible},
                       cal.intrinsics,cal.camera_to_world_cv,cal.bottom_z_m,
                       cal.area_grid(),cal.radii_m)
            median_geometry=balanced.estimate(rgb,raw,{'mask':visible},
                       cal.intrinsics,cal.camera_to_world_cv,cal.bottom_z_m,
                       cal.area_grid(),cal.radii_m)
            previous=old[case,rep]
            rows.append({'case':case,'repeat':rep,'band':previous['band'],
                         'truth':previous['true_height_m'],
                         'baseline':previous['prediction_m'],
                         'oracle_mask_installed':fitted['numeric_candidate_m'],
                         'oracle_mask_true_geometry':truth_geometry['level_m'],
                         'oracle_mask_true_balanced':median_geometry['level_m']})
    with a.output.open('w') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    main()
