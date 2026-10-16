"""Independent native-pixel geometric sensor experiment, not RGB rendering.

Ordinary estimator consumes calibrated ROI and raw depth only. Test truth is
kept in this generator/evaluator, never passed to an estimator. One empty frame
is the only added prior. V38 noise is applied to BOTH surface and bottom echoes.
"""
import argparse, json, os, time, hashlib
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from liquid_depth.device_profiles_v38 import PROFILES, noisy_depth, metric_summary
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.surface_refinement import StereoNoiseModel
from liquid_depth.ordinary_surface_v36 import estimate_v36
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference, estimate_v39

PROFILE=PROFILES[1]
MODEL=StereoNoiseModel(PROFILE.baseline_m,.18,1e-9,1e-9,PROFILE.min_z,PROFILE.max_z)


def configuration(seed, index, group):
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,3901]))
    if group=='far_shallow':
        h=float(rng.uniform(.01,1.)); distance=float(rng.uniform(5.5,min(9.7,9.98-h)))
    elif group=='empty':
        h=0.;distance=float(rng.uniform(5.5,9.95))
    elif group=='near_zero':
        h=float(rng.uniform(.005,.1));distance=float(rng.uniform(5.5,9.8))
    else:
        band=int(group.split('_')[1]);h=float(rng.uniform(band,band+1))
        distance=float(rng.uniform(.65,9.98-h))
    bottom=h+distance
    pos=np.array([rng.uniform(-.035,.035)*bottom,rng.uniform(-.035,.035)*bottom,bottom])
    target=np.array([0.,0.,h/2])
    forward=(target-pos)/np.linalg.norm(target-pos)
    right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    pose=np.eye(4);pose[:3,:3]=np.column_stack((right,down,forward));pose[:3,3]=pos
    # Keep Euclidean bottom-centre distance below 10 m too.
    assert np.linalg.norm(pos)<=10
    rx=bottom*float(rng.uniform(.14,.27));ry=rx*float(rng.uniform(.75,1.2))
    k=np.array([[PROFILE.fx,0,(PROFILE.width-1)/2],[0,PROFILE.fx,(PROFILE.height-1)/2],[0,0,1.]])
    cal=HeightCalibration(k,pose,0.,(rx,ry),float(rng.choice([2.,4.,10.])))
    rim=h+min(float(rng.uniform(.08,.5)),distance*.3)
    return rng,cal,rim,h,distance,float(rng.uniform(.02,.16)),float(rng.uniform(.03,.2))


def clean_maps(cal,rim,h):
    yy,xx=np.indices((PROFILE.height,PROFILE.width))
    k=cal.intrinsics;t=cal.camera_to_world_cv
    rays=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones(xx.shape)),-1)@t[:3,:3].T
    c=-rays[...,2]
    bottom=t[2,3]/c;surface=(t[2,3]-h)/c
    def inside(z):
        xy=t[:2,3]+z[...,None]*rays[...,:2]
        return (abs(xy/np.array(cal.radii_m))**cal.footprint_exponent).sum(-1)<=1
    bottommask=inside(bottom);surfacemask=inside(surface)
    # Actual finite vessel plus outer background; wall details stay outside
    # the conservative bottom-and-opening ROI used by both compared methods.
    background=(t[2,3]+.4)/c
    dry=np.where(bottommask,bottom,background)
    wet=np.where(surfacemask,surface,background)
    return dry.astype(np.float32),wet.astype(np.float32),bottom.astype(np.float32),surfacemask


def run_case(task):
    seed,index,group,ablation,drift=task
    rng,cal,rim,h,distance,echo,drop=configuration(seed,index,group)
    dry,wet,floor,inside=clean_maps(cal,rim,h)
    cluster=f'{seed}_{index}_{group}'
    rows=[];audit=[]
    for regime in ('nominal','adverse'):
        # Same installation bias/spatial field, independent dry-frame innovation.
        # Drift is an extra explicit sensitivity run, not hidden in nominal.
        ns=seed+index*907
        empty=noisy_depth(dry,PROFILE,ns,99,regime)
        empty[np.random.default_rng(ns+3907).random(empty.shape)<drop]=0
        reference=calibrate_dry_reference(empty,cal,rim,MODEL)
        for repeat in range(4):
            rr=np.random.default_rng(np.random.SeedSequence([seed,index,repeat,3903]))
            clean=wet.copy()
            # Select first, then apply range/disparity noise to every return.
            hit=inside&(rr.random(clean.shape)<echo);clean[hit]=floor[hit]
            clean[rr.random(clean.shape)<drop]=0
            if drift:
                offset=np.random.default_rng(ns+3913).uniform(-drift,drift)
                clean=np.where(clean>0,PROFILE.fx*PROFILE.baseline_m/(PROFILE.fx*PROFILE.baseline_m/np.maximum(clean,1e-6)+offset),0)
            raw=noisy_depth(clean,PROFILE,ns,repeat,regime)
            methods={'baseline_v36':lambda:estimate_v36(None,raw,cal,rim,PROFILE.sensor,uniform=True,noise_model=MODEL),
                     'v39_bias_noisy_echo':lambda:estimate_v39(raw,cal,rim,MODEL,reference)}
            if ablation:
                methods.update(v39_no_reference=lambda:estimate_v39(raw,cal,rim,MODEL),
                               v39_noise_cal_only=lambda:estimate_v39(raw,cal,rim,MODEL,replace(reference,disparity_bias_px=0.)),
                               v39_bias_atom_echo=lambda:estimate_v39(raw,cal,rim,MODEL,reference,noisy_echo=False))
            for name,fn in methods.items():
                started=time.perf_counter();result=fn();elapsed=time.perf_counter()-started
                stats=result['statistics'];pred=float(stats['mean_m']) if stats else None
                rows.append(dict(cluster=cluster,case=index,group=group,band='empty' if h==0 else str(int(h)),
                    population='uniform' if group.startswith('uniform') else group,regime=regime,repeat=repeat,
                    method=name,true_height_m=h,empty=h==0,predicted_height_m=pred,camera_to_surface_m=distance,
                    camera_to_bottom_m=float(np.linalg.norm(cal.camera_to_world_cv[:3,3])),echo_probability=echo,
                    reasons=result['reasons'],elapsed_s=elapsed,calibrated_bias_px=result.get('calibrated_bias_px'),
                    calibrated_sigma_px=result.get('calibrated_sigma_px'),fitted_echo_fraction=result.get('fitted_echo_fraction')))
        audit.append(dict(cluster=cluster,regime=regime,reference=reference.__dict__))
    return rows,audit


def summarize(rows):
    result={}
    for regime in sorted({r['regime'] for r in rows}):
      for method in sorted({r['method'] for r in rows}):
        rr=[r for r in rows if r['regime']==regime and r['method']==method]
        key=regime+'|'+method
        groups={'uniform':[r for r in rr if r['population']=='uniform'],
                'far_shallow':[r for r in rr if r['group']=='far_shallow'],
                'empty':[r for r in rr if r['empty']],
                'near_zero':[r for r in rr if r['group']=='near_zero']}
        groups.update({'band_'+str(i):[r for r in rr if r['population']=='uniform' and r['band']==str(i)] for i in range(9)})
        result[key]={name:metric_summary(records) for name,records in groups.items()}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--far',type=int,default=96);p.add_argument('--per-band',type=int,default=12);p.add_argument('--empty',type=int,default=24)
    p.add_argument('--near-zero',type=int,default=24);p.add_argument('--workers',type=int,default=8);p.add_argument('--ablation',action='store_true');p.add_argument('--drift',type=float,default=0.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    groups=['far_shallow']*a.far+['empty']*a.empty+['near_zero']*a.near_zero
    groups += [f'uniform_{i}' for i in range(9) for j in range(a.per_band)]
    protocol=dict(seed=a.seed,independent_configurations=len(groups),four_repeats=True,one_empty_prior=True,
                  native_shape=[PROFILE.height,PROFILE.width],geometric_sensor_simulation_not_rgb_render=True,
                  noise='V38 nominal/adverse; shared installation field, independent dry/wet innovations',drift_limit_px=a.drift,
                  oracle_inputs=False,ideal_intrinsics_pose_bottom=True,no_height_clipping=True,relaxed_gate=True,
                  source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),Path(__file__).parents[1]/'src/liquid_depth/ordinary_surface_v39.py']})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];audit=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_case,(a.seed,i,g,a.ablation,a.drift)) for i,g in enumerate(groups)]
        for i,f in enumerate(futures):
            r,d=f.result();rows+=r;audit+=d
            with (a.output/'records.jsonl').open('a') as file:
                for row in r:file.write(json.dumps(row,allow_nan=False)+'\n')
            print('DONE',i+1,len(groups),round(time.time()-start,1),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    (a.output/'calibration-audit.json').write_text(json.dumps(audit,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(configurations=len(groups),records=len(rows),elapsed_s=time.time()-start)))
    print('COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
