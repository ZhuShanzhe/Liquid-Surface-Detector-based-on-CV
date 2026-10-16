"""Post-evaluation audit only; known simulator offsets never enter inference."""
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
from liquid_depth.device_profiles_v38 import PROFILES,noisy_depth,spatial_field
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.surface_refinement import StereoNoiseModel
from liquid_depth.ordinary_surface_v35 import interior_ray_roi
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference,estimate_v39

root=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')
records=[]
for case in (8,11,14,20,29,32):
    slot=(case*2+1)*10;folder=root/f'{slot:05d}';meta=json.loads((folder/'metadata.json').read_text())
    cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
    profile=PROFILES[1];m=StereoNoiseModel(.1,.18,1e-9,1e-9,.4,10.)
    yy,xx=np.indices((profile.height,profile.width));k=cal.intrinsics;t=cal.camera_to_world_cv
    rays=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones(xx.shape)),-1)@t[:3,:3].T
    dry=(t[2,3]-cal.bottom_z_m)/(-rays[...,2]);rim=meta['container_rim_z_m']
    roi=interior_ray_roi(cal,rim,dry.shape,.65)
    seed=meta['seed']-meta['case']*100003+slot*137
    global_rng=np.random.default_rng(np.random.SeedSequence([seed,382]));field=spatial_field(global_rng,dry.shape)
    global_bias=float(global_rng.uniform(-.025,.025)*2)
    rawdry=noisy_depth(dry,profile,seed,99,'adverse');ref=calibrate_dry_reference(rawdry,cal,rim,m)
    valid=roi&(rawdry>0)
    raw=np.load(folder/'raw_00_adverse_base.npy');p=estimate_v39(raw,cal,rim,m,ref)
    records.append(dict(case=case,true_height_m=meta['nominal_liquid_height_m'],predicted=p['statistics']['mean_m'],
        reference=ref.__dict__,dry_valid_roi=float(valid.sum()/max(roi.sum(),1)),
        dry_geometric_axial_range=[float(dry[roi].min()),float(dry[roi].max())],
        simulator_global_bias=global_bias,simulator_effective_roi_bias=global_bias+.36*np.sqrt(.5)*float(field[roi].mean()),
        simulator_local_sigma=float(np.sqrt(.36**2*.5*(1+np.var(field[roi]))))))
out=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v39-replay-20261012/calibration-diagnostic.json')
out.write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
