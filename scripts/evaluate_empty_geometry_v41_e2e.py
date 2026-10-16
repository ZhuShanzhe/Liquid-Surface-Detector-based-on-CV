"""Evaluate saved V41 profiles via the actual product factory and frame I/O.

RGB is a neutral placeholder, NOT an optical render. Simulated operator clicks
and measured lengths have explicit noise. No model tuning or test-frame memory.
"""
import argparse,json,time,hashlib,csv
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import cv2
import numpy as np
from experiment_ordinary_v39 import configuration,clean_maps,PROFILE,summarize
from experiment_ordinary_v40 import observation
from experiment_empty_geometry_v41 import annotation
from liquid_depth.empty_geometry_runtime_v41 import create_empty_geometry_profile
from liquid_depth.rail_runtime import make_product_system
from liquid_depth.ordinary_surface_v40 import prepare,calibrate,estimate
from liquid_depth.device_profiles_v38 import metric_summary


def write_frame(path,k,depth):
    path.mkdir(parents=True,exist_ok=False)
    # Millimetre storage deliberately exercises profile unit conversion.
    np.save(path/'depth.npy',depth.astype(np.float32)*1000,allow_pickle=False)
    if not cv2.imwrite(str(path/'rgb.png'),np.full((*depth.shape,3),100,np.uint8)):
        raise OSError('synthetic_frame_image_write_failed')
    (path/'depth_info.json').write_text(json.dumps(dict(K=k.ravel().tolist(),width=depth.shape[1],height=depth.shape[0])))
    return path


def run(task):
    seed,index,group,outroot,rho=task
    basegroup='uniform_'+group.rsplit('_',1)[1] if group.startswith('fixed_sweep_') else group
    rng,cal,rim,h,_,_,drop=configuration(seed,index,basegroup)
    shape=['circle','ellipse','rectangle'][index%3]
    cal=replace(cal,radii_m=(cal.radii_m[0],cal.radii_m[0]) if shape=='circle' else cal.radii_m,
        footprint_exponent=16. if shape=='rectangle' else 2.)
    if group.startswith('fixed_sweep_'):
        heights=sorted(set([0.,.005,.02,.05,.1,.25,.5]+[i+.5 for i in range(9)]+[.9*rim]))
        heights=[x for x in heights if x<=rim-.01]
    else:heights=[h]
    dry,_,_,_=clean_maps(cal,rim,0.)
    spec=annotation(cal,rim,shape,rng)
    distance_spec={**spec,'metric_anchor':dict(kind='perpendicular_distance',
        distance_m=float(cal.camera_to_world_cv[2,3]+rng.normal(0,.001+.001*cal.camera_to_world_cv[2,3])))}
    specs={'empty_span':spec,'empty_distance':distance_spec}
    ns=seed+index*907;root=Path(outroot)/'cases'/f'{index:04d}';rows=[];audit=[]
    for regime in ('nominal','adverse'):
        case=root/regime;empty,_=observation(dry,ns,99,regime,rho,'diffuse_clean',drop)
        emptydir=write_frame(case/'empty',cal.intrinsics,empty)
        truectx=prepare(cal,rim,dry.shape,PROFILE.baseline_m)
        idealref=calibrate(empty,truectx)
        meters={};failures={}
        for method,s in specs.items():
            start=time.perf_counter()
            try:
                profile,bundle=create_empty_geometry_profile(dict(frame_dir=str(emptydir),geometry=s,
                    depth_scale_to_m=.001,profile_path=str(case/(method+'.yaml'))))
                meters[method]=make_product_system(profile)
                n=np.asarray(bundle['bottom_plane_camera']['up']);trueup=cal.camera_to_world_cv[2,:3]
                audit.append(dict(case=index,group=group,regime=regime,method=method,shape=shape,
                    initialized=True,initialization_s=time.perf_counter()-start,
                    bottom_distance_error_mm=(bundle['bottom_plane_camera']['perpendicular_distance_m']-cal.camera_to_world_cv[2,3])*1000,
                    normal_error_deg=float(np.degrees(np.arccos(np.clip(n@trueup,-1,1)))),
                    geometry_quality=bundle['quality'],profile_path=str(profile)))
            except ValueError as ex:
                meters.pop(method,None)
                failures[method]=str(ex)
                audit.append(dict(case=index,group=group,regime=regime,method=method,shape=shape,
                    initialized=False,failure_stage='profile_initialization',error=str(ex)))
        for hi,truth in enumerate(heights):
            _,wet,_,_=clean_maps(cal,rim,truth)
            for repeat in range(4):
                # Independent innovations at every level; one empty frame remains fixed.
                raw,_=observation(wet,ns,hi*4+repeat,regime,rho,'diffuse_clean',drop)
                frame=write_frame(case/f'level_{hi:02d}_frame_{repeat}',cal.intrinsics,raw)
                for method in ('ideal_geometry_v40','empty_span','empty_distance'):
                    elapsed=None;accepted=False;failure_stage=None
                    if method=='ideal_geometry_v40':
                        start=time.perf_counter();result=estimate(raw,truectx,idealref)
                        elapsed=time.perf_counter()-start;pred=None if result['statistics'] is None else result['statistics']['mean_m']
                        reasons=result['reasons']
                    elif method in meters:
                        start=time.perf_counter();result=meters[method].measure(frame)
                        elapsed=time.perf_counter()-start;pred=result['liquid_depth_candidate_m'];accepted=result['accepted']
                        reasons=result['rejection_reasons']
                        if pred is None:failure_stage='measurement'
                    else:pred=None;reasons=[failures[method]];failure_stage='profile_initialization'
                    if pred is not None and not np.isfinite(pred):raise AssertionError('nonfinite_prediction')
                    if accepted:raise AssertionError('research_route_must_not_be_promoted_to_production')
                    row=dict(cluster=f'{seed}_{index}',case=index,group=group,
                        population='fixed_sweep' if group.startswith('fixed_sweep_') else ('uniform' if group.startswith('uniform') else group),
                        band='empty' if truth==0 else str(int(truth)),method=method,regime=regime,repeat=repeat,
                        level_index=hi,shape=shape,true_height_m=truth,empty=truth==0,predicted_height_m=pred,
                        camera_to_surface_m=float(cal.camera_to_world_cv[2,3]-truth),
                        camera_to_bottom_m=float(np.linalg.norm(cal.camera_to_world_cv[:3,3])),
                        container_capacity_m=rim,elapsed_s=elapsed,accepted=accepted,reasons=reasons,
                        failure_stage=failure_stage,frame_dir=str(frame))
                    rows.append(row)
    return rows,audit


def metrics(rows):
    s=metric_summary(rows)
    times=[x['elapsed_s']*1000 for x in rows if x['elapsed_s'] is not None]
    s['latency_ms']=dict(p50=float(np.percentile(times,50)),p95=float(np.percentile(times,95)),
        over_500ms_pct=float(np.mean(np.asarray(times)>500)*100),scope='frame_load_and_cpu_meter_no_capture_no_network_no_serialization_parallel_workers') if times else None
    s['no_output_reasons']=dict(Counter(x['reasons'][0] for x in rows if x['predicted_height_m'] is None))
    s['production_accepted_outputs']=sum(x['accepted'] for x in rows)
    good=[x for x in rows if x['predicted_height_m'] is not None]
    if good:
        e=np.asarray([abs(x['predicted_height_m']-x['true_height_m'])*1000 for x in good])
        s['max_error_mm']=float(e.max());s['median_error_mm']=float(np.median(e))
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--far',type=int,default=96);p.add_argument('--empty',type=int,default=24);p.add_argument('--near-zero',type=int,default=24)
    p.add_argument('--per-band',type=int,default=24);p.add_argument('--fixed-per-band',type=int,default=2)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--rho',type=float,default=1.)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    groups=['far_shallow']*a.far+['empty']*a.empty+['near_zero']*a.near_zero
    groups += [f'uniform_{i}' for i in range(9) for _ in range(a.per_band)]
    groups += [f'fixed_sweep_{i}' for i in range(9) for _ in range(a.fixed_per_band)]
    repo=Path(__file__).parents[1]
    sources=[Path(__file__),Path(__file__).with_name('experiment_empty_geometry_v41.py'),
        Path(__file__).with_name('experiment_ordinary_v40.py'),Path(__file__).with_name('experiment_ordinary_v39.py')]
    sources += [repo/'src/liquid_depth'/x for x in ('empty_geometry_v41.py','empty_geometry_runtime_v41.py','ordinary_surface_v40.py','rail_runtime.py','io.py','device_profiles_v38.py')]
    protocol=dict(seed=a.seed,configurations=len(groups),main_configurations=len(groups)-9*a.fixed_per_band,
        fixed_container_sweep_configurations=9*a.fixed_per_band,one_empty_frame_per_installation=True,
        four_independent_test_frames_per_level=True,temporal_joint_estimation=False,
        actual_profile_save_factory_and_measure=True,stored_depth_units='mm',profile_conversion_to_m=.001,
        camera=PROFILE.manifest(),click_sigma_px=.3,manual_length_sigma_m='0.001+0.001*length',
        spatial_retention=a.rho,independent_dimension_check=True,independent_gravity_provided=False,
        application_camera_to_bottom_limit_m=10.,numeric_range_clipping=False,forced_zero=False,
        ordinary_bottom_echo_probability=0.,runtime_gate_relaxed=True,production_qualified=False,
        synthetic_rgb_is_placeholder_not_photometric_simulation=True,
        uncertainty='noise sensitivity assumptions, not vendor-calibrated probabilities; repeats share geometry and prior',
        source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    allrows=[];audit=[];started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(rows,aa) in enumerate(pool.map(run,[(a.seed,i,g,str(a.output),a.rho) for i,g in enumerate(groups)])):
            allrows+=rows;audit+=aa
            with (a.output/'records.jsonl').open('a') as f:
                for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
            print('DONE',i+1,len(groups),round(time.time()-started,1),flush=True)
    summary={}
    for method in sorted({r['method'] for r in allrows}):
      for regime in ('nominal','adverse'):
        rr=[r for r in allrows if r['method']==method and r['regime']==regime]
        slices={g:[r for r in rr if r['population']==g] for g in ('uniform','far_shallow','empty','near_zero','fixed_sweep')}
        slices['main_all']=[r for r in rr if r['population']!='fixed_sweep']
        slices.update({f'band_{i}':[r for r in rr if r['population']=='uniform' and r['band']==str(i)] for i in range(9)})
        summary[regime+'|'+method]={g:metrics(v) for g,v in slices.items()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (a.output/'calibration-audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    with (a.output/'records.csv').open('w',newline='') as f:
        fields=[k for k in allrows[0] if k!='reasons'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in allrows:w.writerow({k:r[k] for k in fields})
    (a.output/'complete.json').write_text(json.dumps(dict(configurations=len(groups),records=len(allrows),elapsed_s=time.time()-started)))
    print('COMPLETE',len(allrows),flush=True)

if __name__=='__main__':main()
