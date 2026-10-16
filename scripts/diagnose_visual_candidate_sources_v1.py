"""Offline provenance of selected candidates; never supplies origin labels to inference."""
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_shared_empty_v45 import install
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.surface_refinement import StereoNoiseModel
from liquid_depth.symmetric_visual_meter import symmetric_candidates
from liquid_depth.visual_region_model import RegionPredictor

BASE=Path('/root/autodl-tmp');OUT=BASE/'liquid-depth-artifacts/visual-route-v1-20261015'
ROOT=BASE/'liquid-depth-data/synthetic/visual-evidence-final-confirmation-20261015';SEED=202610153229


def main():
    records=[json.loads(s) for s in (OUT/'final-model-v1/records.jsonl').read_text().splitlines()]
    rows=[r for r in records if r['method']=='visual_guided']
    rows.sort(key=lambda r:abs(r['liquid_depth_m']-r['truth']['liquid_height_m']) if r['liquid_depth_m'] is not None else -1,reverse=True)
    predictor=RegionPredictor(OUT/'model-v1/best.pt','cpu')
    result=[]
    for r in rows[:8]:
        case=r['case'];folder=ROOT/f'{case:05d}'
        item=install(case,ROOT,ROOT/'dry',SEED,max_annotation_attempts=6,source_slot_stride=1)
        base=FormalSceneMeter.from_empty_geometry(item['bundle'],item['dry'],max_return_depth_m=10.)
        frame=json.loads((folder/'frames.json').read_text())[r['repeat']]
        raw=np.load(frame['raw_path']);roi=base.ordinary_container_roi
        valid=roi&np.isfinite(raw)&(raw>0);y,x=np.nonzero(valid)
        cal=height_calibration(item['bundle']);h=world_points(raw,cal)[...,2][valid]-cal.bottom_z_m;z=raw[valid]
        probs=predictor(cv2.imread(str(folder/'rgb.png')),roi)[:,valid]
        if item['bundle']['sensor_family']=='tof':noise=.0015+.0007*z
        else:
            m=StereoNoiseModel.simulation_proxy(item['bundle']['sensor_family'])
            noise=np.sqrt((m.depth_noise_floor_m+m.depth_noise_quadratic*z*z)**2+(m.disparity_sigma_px*z*z/(cal.intrinsics[0,0]*m.baseline_m))**2)
        candidates=symmetric_candidates(h,np.column_stack([x,y]),noise,probs)
        # Origin labels enter here, after blind candidate selection has finished.
        origins=np.load(folder/f"origin_offline_{r['repeat']:02d}.npy")[valid]
        detail=[]
        for c in candidates[:4]:
            source={str(i):int((origins[c['mask']]==i).sum()) for i in range(7)}
            detail.append({k:v for k,v in c.items() if k not in {'mask','weights'}}|{'physical_origin_counts_offline':source})
        result.append({'case':case,'repeat':r['repeat'],'scene':r['scene'],'sensor':r['sensor'],
           'true_liquid_m':r['truth']['liquid_height_m'],'prediction_m':r['liquid_depth_m'],
           'mean_semantic_probabilities_on_returns':probs.mean(1).tolist(),'candidates':detail})
    payload={'source_legend':{'0':'no return','1':'native liquid source, may still be noisy','2':'floating solid','3':'bottom','4':'wall/background context','5':'simulated wrong far layer','6':'flying mixed return'},
             'limitations':'Exact generator provenance, not real optical-path labels or proof every native liquid return is metrically accurate.',
             'frames':result}
    (OUT/'candidate-source-diagnosis.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
