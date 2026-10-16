"""Secondary replay on existing V38 raw frames; not a new independent test set."""
import json,time
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from liquid_depth.device_profiles_v38 import PROFILES,noisy_depth
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.surface_refinement import StereoNoiseModel
from liquid_depth.ordinary_surface_v36 import estimate_v36
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference,estimate_v39
from experiment_ordinary_v39 import summarize


def run(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['scenario']!='ordinary' or meta['profile']!='gemini2xl_dense':return []
    cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
    profile=PROFILES[1];model=StereoNoiseModel(.1,.18,1e-9,1e-9,.4,10.)
    yy,xx=np.indices((profile.height,profile.width));k=cal.intrinsics;t=cal.camera_to_world_cv
    rays=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones(xx.shape)),-1)@t[:3,:3].T
    bottom=t[2,3]-cal.bottom_z_m;dry=bottom/(-rays[...,2])
    seed=meta['seed']-meta['case']*100003+int(folder.name)*137
    refs={regime:calibrate_dry_reference(noisy_depth(dry,profile,seed,99,regime),cal,meta['container_rim_z_m'],model) for regime in ('nominal','adverse')}
    rows=[]
    for frame in json.loads((folder/'frames.json').read_text()):
        raw=np.load(frame['raw_path']);regime=frame['regime']
        for name,result in [('baseline_v36',estimate_v36(None,raw,cal,meta['container_rim_z_m'],profile.sensor,uniform=True,noise_model=model)),
                            ('v39_bias_noisy_echo',estimate_v39(raw,cal,meta['container_rim_z_m'],model,refs[regime]))]:
            h=meta['nominal_liquid_height_m'];pred=result['statistics']
            rows.append(dict(cluster=meta['cluster'],case=meta['case'],group='empty' if meta['empty'] else ('near_zero' if meta['population']=='near_zero_enrichment' else 'uniform_'+str(int(h))),
                band='empty' if h==0 else str(int(h)),population=meta['population'],regime=regime,repeat=frame['repeat'],method=name,true_height_m=h,
                empty=meta['empty'],predicted_height_m=pred['mean_m'] if pred else None,camera_to_surface_m=meta['camera_position_m'][2],reasons=result['reasons']))
    return rows


if __name__=='__main__':
    root=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')
    out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v39-replay-20261012');out.mkdir(exist_ok=False)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for records in pool.map(run,sorted(x.parent for x in root.glob('*/metadata.json'))):rows+=records
    (out/'records.json').write_text(json.dumps(rows,allow_nan=False))
    (out/'summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    (out/'complete.json').write_text(json.dumps(dict(records=len(rows),purpose='Existing rendered full-frame raw-depth regression; added synthetic empty reference, not independent test')))
    print('COMPLETE',len(rows),flush=True)
