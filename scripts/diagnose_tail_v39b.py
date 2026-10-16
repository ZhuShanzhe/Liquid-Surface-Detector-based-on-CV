"""Post-hoc tail attribution. Oracle values are diagnostic ONLY, never deployment.
Does not edit models, production routes, or prior evaluation artifacts.
"""
import json
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from experiment_ordinary_v39 import configuration,clean_maps,PROFILE,MODEL
from liquid_depth.device_profiles_v38 import noisy_depth,spatial_field
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.ordinary_surface_v35 import interior_ray_roi
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference,_observations,_cells,_prob
from liquid_depth.ordinary_surface_v39b import estimate_v39
from diagnose_over10_v39b import ROOT,OUT,load


def population(rows):
    result={}
    for reg in ('nominal','adverse'):
        rr=[r for r in rows if r['regime']==reg]
        e=np.array([abs(r['predicted_height_m']-r['true_height_m'])*1000 for r in rr])
        cases=defaultdict(list)
        for r in rr:cases[r['cluster']].append((r['predicted_height_m']-r['true_height_m'])*1000)
        tail=[np.array(x) for x in cases.values() if np.mean(abs(np.array(x)))>=10]
        groups={}
        for name,lo,hi in [('under_5.5',0,5.5),('5.5_to_8',5.5,8),('8_to_9',8,9),('9_plus',9,11)]:
            ss=[r for r in rr if lo<=r['camera_to_surface_m']<hi]
            ee=np.array([abs(r['predicted_height_m']-r['true_height_m'])*1000 for r in ss])
            groups[name]=dict(frames=len(ss),mae_mm=float(ee.mean()) if len(ee) else None,
                over10_pct=float((ee>=10).mean()*100) if len(ee) else None,over20_pct=float((ee>=20).mean()*100) if len(ee) else None)
        result[reg]=dict(frames=len(rr),quantiles_mm=dict(zip(['p50','p90','p95','p99','max'],np.percentile(e,[50,90,95,99,100]).tolist())),
            tail_counts={str(t):int((e>=t).sum()) for t in (5,10,20,50,100)},
            cases_mae_over10=len(tail),cases_over10_all_four_same_sign=int(sum(np.all(x>0) or np.all(x<0) for x in tail)),
            tail_median_within_case_std_mm=float(np.median([np.std(x) for x in tail])) if tail else None,
            tail_median_case_mae_mm=float(np.median([np.mean(abs(x)) for x in tail])) if tail else None,
            distance_groups=groups)
    return result


def roi(cal,rim,shape,wide=False):
    r=interior_ray_roi(cal,rim,shape,.9 if wide else .65)
    if r.sum()<256:r=interior_ray_roi(replace(cal,bottom_z_m=rim-1e-5),rim,shape,.65)
    return r


def fixed_alpha(raw,cal,rim,ref,alpha):
    z,c=_observations(raw,cal,rim)[0];fb=ref.fb;cells=_cells(z,fb,MODEL);sd=ref.disparity_sigma_px;b=ref.disparity_bias_px
    fm,fn=_prob(cells,fb*c/ref.bottom_clearance_m+b,sd);floor=fm/np.maximum(fn,1e-300)
    start=float(np.median((fb/z-b)/c))
    def objective(p):
        m,n=_prob(cells,p[0]*c+b,sd);prob=(1-alpha)*m/np.maximum(n,1e-300)+alpha*floor
        return float(-np.mean(np.log(np.maximum(.995*prob+.005/4096,1e-300))))
    fit=minimize(objective,[start],method='L-BFGS-B',bounds=[(max(.01,start-max(3*sd,start*.15)),start+max(3*sd,start*.15))],options={'ftol':1e-12,'maxiter':100})
    return ref.bottom_clearance_m-fb/fit.x[0]


def inputs(source,index,group,regime):
    if source=='test':
        seed=202610124037
        _,cal,rim,h,distance,echo,drop=configuration(seed,index,group)
        dry,wet,floor,inside=clean_maps(cal,rim,h);ns=seed+index*907
        empty=noisy_depth(dry,PROFILE,ns,99,regime)
        empty[np.random.default_rng(ns+3907).random(empty.shape)<drop]=0
        frames=[]
        for repeat in range(4):
            rr=np.random.default_rng(np.random.SeedSequence([seed,index,repeat,3903]));clean=wet.copy()
            hit=inside&(rr.random(clean.shape)<echo);clean[hit]=floor[hit]
            clean[rr.random(clean.shape)<drop]=0
            frames.append((noisy_depth(clean,PROFILE,ns,repeat,regime),hit))
    else:
        folder=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012')/f'{(index*2+1)*10:05d}'
        meta=json.loads((folder/'metadata.json').read_text());cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
        rim=meta['container_rim_z_m'];h=meta['nominal_liquid_height_m'];distance=meta['camera_position_m'][2]
        yy,xx=np.indices((PROFILE.height,PROFILE.width));k=cal.intrinsics;t=cal.camera_to_world_cv
        rays=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones(xx.shape)),-1)@t[:3,:3].T
        dry=(t[2,3]-cal.bottom_z_m)/(-rays[...,2]);ns=meta['seed']-index*100003+int(folder.name)*137
        empty=noisy_depth(dry,PROFILE,ns,99,regime)
        frames=[(np.load(folder/f'raw_{j:02d}_{regime}_base.npy'),None) for j in range(4)]
        echo=None
    return cal,rim,h,distance,ns,dry,empty,frames,echo


def probe(source,index,group,regime):
    cal,rim,h,distance,ns,dry,empty,frames,echo=inputs(source,index,group,regime)
    reference=calibrate_dry_reference(empty,cal,rim,MODEL);ro=roi(cal,rim,empty.shape)
    rng=np.random.default_rng(np.random.SeedSequence([ns,382]));field=spatial_field(rng,empty.shape)
    bias=float(rng.uniform(-.025,.025)*(2 if regime=='adverse' else 1))
    sigma=.18 if regime=='nominal' else .36;fieldscale=0 if regime=='nominal' else sigma*np.sqrt(.5)
    wetroi=roi(cal,rim,empty.shape,wide=np.median(frames[0][0][frames[0][0]>0])<3)
    # Pure simulator diagnostic: scalar mean of persistent field BEFORE censoring.
    actual_bias=bias+fieldscale*float(field[wetroi].mean())
    actual_sigma=sigma if regime=='nominal' else float(np.sqrt(sigma**2*.5*(1+field[wetroi].var())))
    variants=defaultdict(list);fitted_alpha=[];observed_alpha=[];errors=[]
    for raw,hit in frames:
        p=estimate_v39(raw,cal,rim,MODEL,reference);variants['current'].append(p['statistics']['mean_m']);fitted_alpha.append(p['fitted_echo_fraction'])
        for key,ref in [('oracle_scalar_bias',replace(reference,disparity_bias_px=actual_bias)),
                        ('oracle_scalar_sigma',replace(reference,disparity_sigma_px=actual_sigma)),
                        ('oracle_scalar_bias_sigma',replace(reference,disparity_bias_px=actual_bias,disparity_sigma_px=actual_sigma))]:
            variants[key].append(estimate_v39(raw,cal,rim,MODEL,ref)['statistics']['mean_m'])
        if hit is not None:
            valid=wetroi&(raw>0)
            alpha=float((hit&valid).sum()/max(valid.sum(),1));observed_alpha.append(alpha)
            variants['oracle_echo_fraction'].append(float(fixed_alpha(raw,cal,rim,reference,alpha)))
    original=[r['predicted_height_m'] for r in load(source) if r['case']==index and r['regime']==regime]
    assert np.max(abs(np.array(original)-variants['current']))<1e-6,(source,index,original,variants['current'])
    return dict(source=source,case=index,group=group,regime=regime,true_height_m=h,camera_to_surface_m=distance,
        camera_to_bottom_m=reference.bottom_clearance_m,dry_valid_roi_pct=float((ro&(empty>0)).sum()/ro.sum()*100),
        reference_points=reference.samples,wet_points=int(estimate_v39(frames[0][0],cal,rim,MODEL,reference)['selected_points']),
        estimated_bias_px=reference.disparity_bias_px,simulator_effective_wet_bias_px=actual_bias,
        estimated_sigma_px=reference.disparity_sigma_px,simulator_effective_wet_sigma_px=actual_sigma,
        mean_bias_predicted_height_shift_mm=float(-distance**2/reference.fb*(reference.disparity_bias_px-actual_bias)*1000),
        fitted_echo_fraction=fitted_alpha,actual_valid_echo_fraction=observed_alpha,
        separation_in_noise_sigma=float(reference.fb*(1/distance-1/reference.bottom_clearance_m)/reference.disparity_sigma_px),
        variants={key:dict(predictions_m=value,mae_mm=float(np.mean(abs(np.array(value)-h))*1000),signed_bias_mm=float(np.mean(np.array(value)-h)*1000)) for key,value in variants.items()})


if __name__=='__main__':
    stats={k:population(load(k)) for k in ('test','replay')}
    specs=[('test',301,'uniform_1','adverse'),('test',275,'uniform_0','adverse'),('test',43,'far_shallow','adverse'),
           ('test',91,'far_shallow','nominal'),('test',242,'near_zero','nominal'),('test',360,'uniform_3','adverse'),
           ('replay',14,'uniform_2','adverse'),('replay',31,'uniform_8','adverse'),('replay',5,'near_zero','nominal')]
    probes=[]
    for spec in specs:
        cache=OUT/f'probe-{spec[0]}-{spec[1]}-{spec[3]}.json'
        p=json.loads(cache.read_text()) if cache.exists() else probe(*spec)
        p['mean_bias_predicted_height_shift_mm']=float(-p['camera_to_surface_m']**2/(PROFILE.fx*PROFILE.baseline_m)*(p['estimated_bias_px']-p['simulator_effective_wet_bias_px'])*1000)
        probes.append(p);cache.write_text(json.dumps(p,indent=2));print('PROBE',spec,'current',p['variants']['current']['mae_mm'],flush=True)
    result=dict(populations=stats,probes=probes,oracle_scope='Post-hoc diagnostics only; scalar bias/sigma probes do not remove non-Gaussian spatial structure and are not achievable measurements')
    (OUT/'tail-causal-diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
