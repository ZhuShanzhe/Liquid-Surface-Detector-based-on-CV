"""No oracle geometry passed to the fitted meter; annotations have explicit noise."""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment_ordinary_v39 import configuration,clean_maps,PROFILE,summarize
from experiment_ordinary_v40 import observation
from liquid_depth.empty_geometry_v41 import build_geometry,height_calibration
from liquid_depth.ordinary_surface_v40 import prepare,calibrate,estimate


def annotation(cal,rim,kind,rng,noise=True):
    rx,ry=cal.radii_m;pose=cal.camera_to_world_cv;k=cal.intrinsics
    def project(xy):
        points=np.column_stack((xy,np.zeros(len(xy))))
        camera=(points-pose[:3,3])@pose[:3,:3];q=camera@k.T
        return q[:,:2]/q[:,2:]
    if kind=='rectangle':outline=project(np.array([[-rx,-ry],[rx,-ry],[rx,ry],[-rx,ry]]))
    else:
        aa=np.linspace(0,2*np.pi,12,endpoint=False);outline=project(np.column_stack((rx*np.cos(aa),ry*np.sin(aa))))
    anchor=project(np.array([[-.6*rx,0],[.6*rx,0]]))
    checker=project(np.array([[0,-.5*ry],[0,.5*ry]]))
    level=lambda x:float(x+rng.normal(0,.001+.001*x)) if noise else float(x)
    points=lambda p:(p+rng.normal(0,.3,p.shape)).tolist() if noise else p.tolist()
    return dict(shape=kind,bottom_outline_px=points(outline),wall_height_m=level(rim),baseline_m=PROFILE.baseline_m,
        metric_anchor=dict(kind='span',pixels=points(anchor),length_m=level(1.2*rx)),
        check_spans=[dict(pixels=points(checker),length_m=level(ry))],
        confirmed_empty=True,horizontal_flat_bottom=True,fixed_camera=True,rectified_aligned=True)


def run(task):
    seed,index,group=task;rng,cal,rim,h,distance,_,drop=configuration(seed,index,group)
    kind=['circle','ellipse','rectangle'][index%3]
    if kind=='circle':cal=replace(cal,radii_m=(cal.radii_m[0],cal.radii_m[0]),footprint_exponent=2.)
    else:cal=replace(cal,footprint_exponent=16. if kind=='rectangle' else 2.)
    dry,wet,_,_=clean_maps(cal,rim,h);ns=seed+index*907
    spec=annotation(cal,rim,kind,rng)
    specs={'empty_span':spec,'empty_distance':{**spec,'metric_anchor':dict(kind='perpendicular_distance',distance_m=float(cal.camera_to_world_cv[2,3]+rng.normal(0,.001+.001*cal.camera_to_world_cv[2,3])))}}
    rows=[];audit=[]
    for regime in ['nominal','adverse']:
        empty,_=observation(dry,ns,99,regime,1.,'diffuse_clean',drop)
        contexts={};errors={};bundles={}
        truectx=prepare(cal,rim,wet.shape,PROFILE.baseline_m);contexts['ideal_geometry_v40']=(truectx,calibrate(empty,truectx))
        for method,s in specs.items():
            try:
                b=build_geometry(empty,cal.intrinsics,s);derived=height_calibration(b)
                ctx=prepare(derived,b['rim_z_m'],wet.shape,b['baseline_m']);contexts[method]=(ctx,calibrate(empty,ctx));bundles[method]=b
                nw=derived.camera_to_world_cv[:3,:3]@cal.camera_to_world_cv[:3,:3].T@np.array([0.,0.,1.])
                audit.append(dict(case=index,shape=kind,regime=regime,method=method,
                    plane_angle_error_deg=float(np.degrees(np.arccos(np.clip(nw[2],-1,1)))),
                    bottom_distance_error_mm=(b['bottom_plane_camera']['perpendicular_distance_m']-cal.camera_to_world_cv[2,3])*1000,
                    width_relative_error_pct=(b['internal_dimensions_m']['width']/(2*cal.radii_m[0])-1)*100,
                    quality=b['quality'],warnings=b['warnings']))
            except ValueError as ex:errors[method]=str(ex);audit.append(dict(case=index,regime=regime,method=method,error=str(ex)))
        for repeat in range(4):
            raw,_=observation(wet,ns,repeat,regime,1.,'diffuse_clean',drop)
            for method in ['ideal_geometry_v40','empty_span','empty_distance']:
                if method in contexts:
                    ctx,ref=contexts[method];start=time.perf_counter();out=estimate(raw,ctx,ref);elapsed=time.perf_counter()-start
                    pred=None if out['statistics'] is None else out['statistics']['mean_m'];reasons=out['reasons']
                else:pred=None;elapsed=0.;reasons=[errors[method]]
                rows.append(dict(cluster=f'{seed}_{index}',case=index,shape=kind,group=group,
                    band='empty' if h==0 else str(int(h)),population='uniform' if group.startswith('uniform') else group,
                    method=method,regime=regime,repeat=repeat,true_height_m=h,empty=h==0,predicted_height_m=pred,
                    camera_to_surface_m=distance,reasons=reasons,elapsed_s=elapsed))
    return rows,audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--far',type=int,default=32);p.add_argument('--empty',type=int,default=12);p.add_argument('--near-zero',type=int,default=12);p.add_argument('--per-band',type=int,default=8);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    groups=['far_shallow']*a.far+['empty']*a.empty+['near_zero']*a.near_zero+[f'uniform_{i}' for i in range(9) for _ in range(a.per_band)]
    sources=[Path(__file__),Path(__file__).parents[1]/'src/liquid_depth/empty_geometry_v41.py',Path(__file__).parents[1]/'src/liquid_depth/ordinary_surface_v40.py',Path(__file__).with_name('experiment_ordinary_v40.py'),Path(__file__).with_name('experiment_ordinary_v39.py')]
    protocol=dict(seed=a.seed,configurations=len(groups),native_shape=[800,1280],shapes=['circle','ellipse','rectangle'],
        click_noise_sigma_px=.3,measured_length_noise_sigma_m='0.001+0.001*length',one_empty_frame=True,
        independent_dimension_check=True,normal_estimated_from_empty_depth=True,ground_truth_in_fitted_meter=False,
        simulation='geometric flat planes and assumed V38 sensor noise, NOT ray-traced wall RGB/IR or measured hardware',
        source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[];audits=[];start=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,(rr,aa) in enumerate(pool.map(run,[(a.seed,i,g) for i,g in enumerate(groups)])):
            rows+=rr;audits+=aa
            with (a.output/'records.jsonl').open('a') as f:
                for row in rr:f.write(json.dumps(row,allow_nan=False)+'\n')
            print('DONE',i+1,len(groups),round(time.time()-start,1),flush=True)
    (a.output/'summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    (a.output/'geometry-audit.json').write_text(json.dumps(audits,indent=2,allow_nan=False))
    (a.output/'complete.json').write_text(json.dumps(dict(configurations=len(groups),records=len(rows),elapsed_s=time.time()-start)))

if __name__=='__main__':main()
