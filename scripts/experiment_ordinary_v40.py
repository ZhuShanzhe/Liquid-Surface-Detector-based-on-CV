"""Independent diffuse-route experiment; ground truth is evaluator-only."""
import argparse,json,time,hashlib,sys
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment_ordinary_v39 import configuration,clean_maps,PROFILE,MODEL,summarize
from liquid_depth.device_profiles_v38 import noisy_depth,spatial_field
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference
from liquid_depth.ordinary_surface_v39b import estimate_v39
from liquid_depth.ordinary_surface_v40 import prepare,calibrate,estimate,disparity,smooth_residual,sigma_from_neighbors


def observation(clean,seed,frame,regime,rho,kind,drop):
    # max_z changes numeric simulation acceptance only, NOT vendor qualification.
    profile=replace(PROFILE,max_z=1e6)
    raw=noisy_depth(clean,profile,seed,frame,regime)
    rng=np.random.default_rng(np.random.SeedSequence([seed,frame,4001]))
    valid=raw>0;fb=PROFILE.fx*PROFILE.baseline_m
    if regime=='adverse' and frame!=99 and rho<1:
        shared=spatial_field(np.random.default_rng(np.random.SeedSequence([seed,382])),raw.shape)
        novel=spatial_field(np.random.default_rng(np.random.SeedSequence([seed,frame,4002])),raw.shape)
        delta=.36*np.sqrt(.5)*((rho-1)*shared+np.sqrt(1-rho*rho)*novel)
        d=fb/np.maximum(raw,1e-6)+delta
        raw=np.where(valid,fb/np.maximum(np.round(d*16)/16,1e-6),0)
        raw=np.round(raw/.001)*.001
    corrupted=np.zeros(raw.shape,bool)
    if kind=='isolated_outliers' and frame!=99:
        corrupted=valid&(rng.random(raw.shape)<.02)
        d=fb/np.maximum(raw,1e-6)+rng.uniform(-2.,2.,raw.shape)*corrupted
        raw=np.where(valid,fb/np.maximum(np.round(d*16)/16,1e-6),0)
    raw[rng.random(raw.shape)<drop]=0
    return raw.astype(np.float32),corrupted


def run(task):
    seed,index,group,kind,rho=task
    _,cal,rim,h,distance,_,drop=configuration(seed,index,group)
    dry,wet,floor,inside=clean_maps(cal,rim,h)
    ctx=prepare(cal,rim,wet.shape,PROFILE.baseline_m)
    ns=seed+index*907;model=replace(MODEL,max_depth_m=1e6)
    rows=[];audit=[]
    for regime in ('nominal','adverse'):
        empty,_=observation(dry,ns,99,regime,rho,kind,drop)
        newref=calibrate(empty,ctx)
        try:oldref=calibrate_dry_reference(empty,cal,rim,model)
        except ValueError:oldref=None
        for repeat in range(4):
            raw,bad=observation(wet,ns,repeat,regime,rho,kind,drop)
            d,valid=disparity(raw,ctx)
            smoothed,_,_=smooth_residual(d,valid,ctx,sigma_from_neighbors(d,valid))
            genuine=valid&inside&~bad
            true_error=np.zeros(raw.shape,np.float32)
            true_error[genuine]=abs(raw[genuine]*ctx.cosine[genuine]-(ctx.bottom-h))
            y,x=np.nonzero(genuine);tiles=np.unique((x//16)+(raw.shape[1]//16+1)*(y//16))
            diag=dict(valid_points=int(valid.sum()),true_surface_points=int(genuine.sum()),
                true_surface_fraction=float(genuine.sum()/max(valid.sum(),1)),true_surface_blocks16=len(tiles),
                points_within_5mm_true_plane=int(np.count_nonzero(genuine&(true_error<=.005))),
                points_within_20mm_true_plane=int(np.count_nonzero(genuine&(true_error<=.02))),
                raw_over10_points=int(np.count_nonzero(valid&(raw>10))),
                genuine_raw_plane_residual_p50_mm=float(np.median(true_error[genuine])*1000),
                genuine_denoised_plane_residual_p50_mm=float(np.median(abs(ctx.fb/smoothed[genuine]*ctx.cosine[genuine]-(ctx.bottom-h)))*1000))
            methods={'v39b_relaxed_range':lambda:estimate_v39(raw,cal,rim,model,oldref)}
            methods.update({mode:(lambda m=mode:estimate(raw,ctx,newref,m)) for mode in ('dense','denoise','scalar','paired')})
            for method,fn in methods.items():
                started=time.perf_counter();out=fn();elapsed=time.perf_counter()-started
                pred=None if out['statistics'] is None else out['statistics']['mean_m']
                rows.append(dict(cluster=f'{seed}_{index}',case=index,group=group,
                    band='empty' if h==0 else str(int(h)),population='uniform' if group.startswith('uniform') else group,
                    method=method,regime=regime,repeat=repeat,true_height_m=h,empty=h==0,predicted_height_m=pred,
                    camera_to_surface_m=distance,camera_to_bottom_m=float(np.linalg.norm(cal.camera_to_world_cv[:3,3])),
                    kind=kind,spatial_retention=rho,elapsed_s=elapsed,reasons=out['reasons'],
                    spatial_transfer=out.get('spatial_transfer'),free_plane_angle_deg=out.get('free_plane_angle_deg'),
                    free_plane_height_m=out.get('free_plane_height_m'),**diag))
        audit.append(dict(case=index,regime=regime,new_reference_points=newref.observed_pixels,
            estimated_independent_sigma_px=newref.sigma,old_reference=None if oldref is None else oldref.__dict__))
    return rows,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--far',type=int,default=96);p.add_argument('--per-band',type=int,default=16)
    p.add_argument('--empty',type=int,default=24);p.add_argument('--near-zero',type=int,default=24)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--kind',choices=['diffuse_clean','isolated_outliers'],default='diffuse_clean')
    p.add_argument('--rho',type=float,default=1.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    groups=['far_shallow']*a.far+['empty']*a.empty+['near_zero']*a.near_zero+[f'uniform_{i}' for i in range(9) for _ in range(a.per_band)]
    paths=[Path(__file__),Path(__file__).parents[1]/'src/liquid_depth/ordinary_surface_v40.py',Path(__file__).with_name('experiment_ordinary_v39.py'),Path(__file__).parents[1]/'src/liquid_depth/device_profiles_v38.py']
    protocol=dict(seed=a.seed,configurations=len(groups),kind=a.kind,spatial_retention=a.rho,
        native_shape=[800,1280],one_empty_prior=True,four_test_frames=True,ideal_installed_geometry=True,
        application_camera_to_bottom_limit_m=10.,numeric_depth_has_no_10m_cutoff=True,hardware_over10_qualified=False,
        ordinary_bottom_echo_probability=0.,truth_in_estimator=False,height_clipping=False,forced_zero=False,
        geometric_sensor_simulation_not_rgb_or_ir_renderer=True,
        source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];audit=[];started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(r,d) in enumerate(pool.map(run,[(a.seed,i,g,a.kind,a.rho) for i,g in enumerate(groups)])):
            rows+=r;audit+=d
            with (a.output/'records.jsonl').open('a') as f:
                for row in r:f.write(json.dumps(row,allow_nan=False)+'\n')
            print('DONE',i+1,len(groups),round(time.time()-started,1),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    (a.output/'calibration-audit.json').write_text(json.dumps(audit,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(configurations=len(groups),records=len(rows),elapsed_s=time.time()-started)))

if __name__=='__main__':main()
