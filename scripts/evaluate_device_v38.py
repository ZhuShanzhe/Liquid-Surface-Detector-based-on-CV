"""Native-depth evaluation. Predicted masks only; GT opened after inference."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from collections import deque,Counter
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from evaluate_liquid_height_v21 import Predictor
from liquid_depth.device_profiles_v38 import PROFILES,metric_summary
from liquid_depth.liquid_height_metrics import HeightCalibration,reference_statistics
from liquid_depth.surface_refinement import RefinedSurfaceEstimator,StereoNoiseModel
from liquid_depth.ordinary_surface_v36 import estimate_v36
from liquid_depth.ordinary_temporal_v37 import estimate_guarded4
from liquid_depth.illumination import adaptive_exposure_correction
from evaluate_other_scenes_v34 import grid

def infer(model,rgb,raw):
    # Only segmentation/network input is reduced. Metric fitting uses native raw.
    w=320;h=256;small=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    z=cv2.resize(raw,(w,h),interpolation=cv2.INTER_NEAREST)
    c=torch.from_numpy(cv2.cvtColor(small,cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)[None]/255).cuda()
    d=torch.from_numpy(z[None,None]).cuda()
    t=time.perf_counter();depth,mask,_=model(c,d);torch.cuda.synchronize()
    elapsed=time.perf_counter()-t
    mask=cv2.resize(mask[0,0].cpu().numpy(),(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_LINEAR)>=.5
    depth=cv2.resize(depth[0,0].cpu().numpy(),(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_LINEAR)
    return dict(mask=mask,depth=depth,neural_seconds=elapsed)

def fitted(rgb,raw,mask,cal,noise,partition=False):
    r=RefinedSurfaceEstimator(mode='partition' if partition else 'sensor',stereo_noise=noise).estimate(
        rgb,raw,{'mask':mask},cal.intrinsics,cal.camera_to_world_cv,cal.bottom_z_m,cal.area_grid(),cal.radii_m)
    s=r['statistics']
    if s:s={k:s[k.replace('_m','_depth_m')] for k in ('mean_m','min_m','max_m','p05_m','p95_m')}
    return dict(statistics=s,reasons=r['quality_flags'])

def measure(frame,meta,rgb,raw,p,enhanced,history):
    t=time.perf_counter();prof=next(x for x in PROFILES if x.name==meta['profile'])
    cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
    noise=StereoNoiseModel(prof.baseline_m,.18,1e-9,1e-9,prof.min_z,prof.max_z) if prof.sensor!='tof' else None
    methods={'v8_neural':grid(p['depth'],p['mask'],cal),
             'native_refined':fitted(rgb,raw,p['mask'],cal,noise,meta['scenario']=='uneven_surface')}
    if enhanced is not None:methods['lowlight_enhanced']=fitted(rgb,raw,enhanced['mask'],cal,noise)
    if meta['scenario']=='ordinary':
        methods['ordinary_profile']=estimate_v36(rgb,raw,cal,meta['container_rim_z_m'],prof.sensor,uniform=True,noise_model=noise)
        if frame['repeat']==3:
            methods['ordinary_guarded4']=estimate_guarded4(history,cal,meta['container_rim_z_m'],prof.sensor,
                noise_model=noise,timestamps=np.arange(4)/prof.fps)
    # Ground truth belongs to evaluation, never to masks, fit, or fusion.
    truth=dict(np.load(Path(frame['folder'])/'geometry.npz'));tm=truth['mask'].astype(bool)
    reference=reference_statistics(meta,cal)
    methods['oracle_mask_diagnostic']=fitted(rgb,raw,tm,cal,noise,meta['scenario']=='uneven_surface')
    common=dict(profile=prof.name,scenario=meta['scenario'],regime=frame['regime'],variant=frame['variant'],
        cluster=meta['cluster'],slot=frame['slot'],repeat=frame['repeat'],population=meta['population'],empty=meta['empty'],
        band=meta['band'],true_height_m=reference['mean_m'],nominal_height_m=meta['nominal_liquid_height_m'],
        camera_to_bottom_m=meta['camera_to_bottom_m'],camera_to_surface_m=meta['camera_position_m'][2],
        mask_iou=float(np.count_nonzero(p['mask']&tm)/max(np.count_nonzero(p['mask']|tm),1)),
        raw_valid_fraction=float(np.count_nonzero(tm&(raw>0))/max(tm.sum(),1)),reference=reference,
        computation_s=time.perf_counter()-t+p['neural_seconds'],hardware_qualified=False)
    records=[]
    for method,result in methods.items():
        s=result['statistics'];value=float(s['mean_m']) if s is not None and np.isfinite(s['mean_m']) else None
        row=dict(common,method=method,predicted_height_m=value,reasons=result.get('reasons',[]))
        if value is not None:
            row.update(absolute_error_m=abs(value-reference['mean_m']),
                relative_error_pct=None if meta['empty'] else abs(value-reference['mean_m'])/reference['mean_m']*100,
                extrema_error_m={k:abs(s[k]-reference[k]) for k in ('min_m','max_m','p05_m','p95_m')})
        records.append(row)
    return records

def summarize(records):
    result={}
    for profile in sorted({r['profile'] for r in records}):
      for scene in sorted({r['scenario'] for r in records}):
       for regime in ('nominal','adverse'):
        for variant in sorted({r['variant'] for r in records if r['scenario']==scene}):
         for method in sorted({r['method'] for r in records if r['scenario']==scene}):
            rows=[r for r in records if (r['profile'],r['scenario'],r['regime'],r['variant'],r['method'])==(profile,scene,regime,variant,method)]
            if not rows:continue
            key='|'.join((profile,scene,regime,variant,method));result[key]={'all':metric_summary(rows)}
            for pop in ('uniform','empty','near_zero_enrichment'):
                result[key][pop]=metric_summary([r for r in rows if r['population']==pop])
            for band in range(11):result[key]['band_'+str(band)]=metric_summary([r for r in rows if r['band']==band])
            for name,lo,hi in [('near_sensor',0,3),('middle_sensor',3,5.5),('far_sensor',5.5,10.01)]:
                result[key][name]=metric_summary([r for r in rows if lo<=r['camera_to_surface_m']<hi])
            result[key]['no_output_reasons']=dict(Counter(x for r in rows if r['predicted_height_m'] is None for x in r['reasons']))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cv2.setNumThreads(1);torch.set_num_threads(2);model=Predictor('v8');rows=[];pending=deque();start=time.time();count=0
    folders=sorted(a.data.glob('*/frames.json'))
    with ThreadPoolExecutor(max_workers=6) as pool:
      for f in folders:
        frames=json.loads(f.read_text());meta=json.loads((f.parent/'metadata.json').read_text());rgb=cv2.imread(str(f.parent/'rgb.png'))
        history={}
        for frame in frames:
            raw=np.load(frame['raw_path']);p=infer(model,rgb,raw)
            pe=None
            if meta['scenario']=='low_light':
                bright=adaptive_exposure_correction(rgb,dict(enabled=True,clahe_clip_limit=1.5)).image_bgr;pe=infer(model,bright,raw)
            key=(frame['regime'],frame['variant']);history.setdefault(key,[]).append(raw)
            pending.append(pool.submit(measure,frame,meta,rgb,raw,p,pe,list(history[key])))
            while len(pending)>12:rows.extend(pending.popleft().result())
            count+=1
        print('EVALUATE',f.parent.name,count,round(time.time()-start,1),flush=True)
      for future in pending:rows.extend(future.result())
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'test-raw.json').write_text(json.dumps(dict(records=rows,elapsed_s=time.time()-start),allow_nan=False))
    (a.output/'summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False))
    print('COMPLETE',count,len(rows),time.time()-start,flush=True)
if __name__=='__main__':main()
