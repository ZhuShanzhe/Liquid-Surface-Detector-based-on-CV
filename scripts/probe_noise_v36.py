"""Controlled one-ray-distribution ablation; not a substitute for rendered tests."""
import json
from pathlib import Path
import numpy as np
from liquid_depth.surface_refinement import StereoNoiseModel,censored_clearance
from liquid_depth.ordinary_surface_v36 import discrete_clearance
rows=[]
for sensor in ('active_stereo','structured_light'):
    model=StereoNoiseModel.simulation_proxy(sensor);fx=320/(2*np.tan(np.deg2rad(27)));fb=fx*model.baseline_m
    for z in (8.2,8.9,9.6):
        for condition in ('continuous','quantized','quantized_bottom_echo'):
            for seed in range(12):
                rng=np.random.default_rng(36001+seed);d=z+rng.normal(0,model.depth_noise_floor_m+model.depth_noise_quadratic*z*z,4096)
                disp=fb/d+rng.normal(0,model.disparity_sigma_px,len(d))
                if condition!='continuous':disp=np.round(disp*16)/16
                d=fb/np.maximum(disp,1e-3)
                if condition=='quantized_bottom_echo':d[rng.random(len(d))<.05]=z+.2
                d=d[(d>=model.min_depth_m)&(d<=model.max_depth_m)]
                d=np.round(d/.001)*.001;c=np.ones(len(d));w=c/len(c)
                old=censored_clearance(d,c,w,fx,model)
                methods={'continuous_likelihood':old}
                if condition!='continuous':
                    methods['quantized_likelihood']=discrete_clearance(d,c,w,fx,model)
                    methods['quantized_echo_likelihood']=discrete_clearance(d,c,w,fx,model,bottom_clearance=z+.2,echo=True)
                rows.append(dict(sensor=sensor,distance=z,condition=condition,seed=seed,
                    error_mm={k:(z-v)*1000 if v is not None else None for k,v in methods.items()}))
out={}
for sensor in ('active_stereo','structured_light'):
    for condition in ('continuous','quantized','quantized_bottom_echo'):
        q=[r for r in rows if r['sensor']==sensor and r['condition']==condition]
        out[sensor+'/'+condition]={m:dict(bias_mm=float(np.mean([r['error_mm'][m] for r in q if r['error_mm'][m] is not None])),
            mae_mm=float(np.mean([abs(r['error_mm'][m]) for r in q if r['error_mm'][m] is not None]))) for m in q[0]['error_mm']}
Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-v36-dev/noise-probe.json').write_text(json.dumps(dict(rows=rows,summary=out),indent=2))
print(json.dumps(out,indent=2))
