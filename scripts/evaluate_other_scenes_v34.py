"""All-frame diagnostics of frozen surface routes; GT is evaluator-only."""
import argparse,hashlib,json,time
from collections import Counter,defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np,torch
from evaluate_liquid_height_v21 import Predictor,WEIGHTS
from liquid_depth.liquid_height_metrics import HeightCalibration,estimate_height,reference_statistics
from liquid_depth.surface_refinement import RefinedSurfaceEstimator,StereoNoiseModel
from liquid_depth.illumination import adaptive_exposure_correction

def grid(z,mask,cal):
    p=estimate_height(z,mask,cal)
    return dict(statistics=p['statistics'],reasons=p['reasons'],qualified=False,
        legacy_geometry_gate=p['accepted'],support=p['footprint_support_fraction'])

def refined(rgb,z,mask,cal,sensor,mode):
    noise=None if sensor=='tof' else StereoNoiseModel.simulation_proxy(sensor)
    p=RefinedSurfaceEstimator(mode=mode,stereo_noise=noise).estimate(rgb,z,{'mask':mask},cal.intrinsics,
        cal.camera_to_world_cv,cal.bottom_z_m,cal.area_grid(),cal.radii_m)
    s=p['statistics']
    if s:s={k:s[k.replace('_m','_depth_m')] for k in ('mean_m','min_m','max_m','p05_m','p95_m')}
    return dict(statistics=s,reasons=p['quality_flags'],qualified=False)

def measure(args):
    frame,meta,rgb,raw,p8,p6,pe=args
    cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
    mode='partition' if meta['scenario']=='uneven_surface' else 'sensor'
    # Inference finishes before loading truth arrays. Only declared installation calibration enters.
    methods={'v8_neural':grid(p8['depth'],p8['mask'],cal),'raw_grid':grid(raw,p8['mask'],cal),
        'refined':refined(rgb,raw,p8['mask'],cal,meta['sensor_family'],mode)}
    if p6 is not None:methods['v6_optional']=grid(p6['depth'],p6['mask'],cal)
    if pe is not None:methods['exposure_refined']=refined(rgb,raw,pe['mask'],cal,meta['sensor_family'],mode)
    # Offline oracle masks are NEVER routed into the deployment methods above.
    geom=dict(np.load(Path(frame['folder'])/'geometry.npz'));truth_mask=geom['mask'].astype(bool)
    methods['oracle_mask_refined']=refined(rgb,raw,truth_mask,cal,meta['sensor_family'],mode)
    oracle=grid(geom['target'],truth_mask,cal)
    ref=reference_statistics(meta,cal)
    union=np.count_nonzero(p8['mask']|truth_mask);inside=truth_mask&(raw>0)
    r=dict(frame=frame,cluster=meta['cluster'],scenario=meta['scenario'],sensor=meta['sensor_family'],
        nominal_height_m=meta['nominal_liquid_height_m'],true_height_m=ref['mean_m'],reference=ref,
        camera_to_bottom_m=meta['camera_to_bottom_m'],camera_to_surface_m=meta['camera_position_m'][2],
        predicted_mask_pixels=int(p8['mask'].sum()),true_visible_pixels=int(truth_mask.sum()),
        mask_iou=float(np.count_nonzero(p8['mask']&truth_mask)/max(union,1)),
        predicted_mask_precision=float(np.count_nonzero(p8['mask']&truth_mask)/max(p8['mask'].sum(),1)),
        raw_surface_valid_fraction=float(inside.sum()/max(truth_mask.sum(),1)),
        oracle_geometry=oracle,methods=methods)
    for p in methods.values():
        s=p['statistics'];p['numeric']=bool(s is not None and np.isfinite(s['mean_m']))
        if p['numeric']:
            e=s['mean_m']-ref['mean_m'];p.update(signed_error_m=e,absolute_error_m=abs(e),
                abs_rel=abs(e)/ref['mean_m'] if meta['nominal_liquid_height_m']>0 else None,
                within_tolerance=bool(abs(e)<=max(.005,.02*ref['mean_m'])),
                extrema_error_m={k:abs(s[k]-ref[k]) for k in ('min_m','max_m','p05_m','p95_m')})
    return r

def predicted(model,rgb,raw):
    colors=torch.from_numpy(np.stack([cv2.cvtColor(v,cv2.COLOR_BGR2RGB).astype(np.float32)/255 for v in rgb]).transpose(0,3,1,2)).cuda()
    depths=torch.from_numpy(np.stack(raw)[:,None]).cuda()
    d,m,_=model(colors,depths)
    d=d[:,0].cpu().numpy();m=m[:,0].cpu().numpy()
    return [{'depth':x,'mask':y>=.5} for x,y in zip(d,m)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cv2.setNumThreads(1);torch.set_num_threads(4)
    frames=[r for f in sorted(a.data.glob('*/frames.json')) for r in json.loads(f.read_text())]
    model=Predictor('v8');optional=Predictor('v6');records=[];pending=deque();started=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(frames),8):
            chunk=frames[start:start+8]
            metas=[json.loads((Path(r['folder'])/'metadata.json').read_text()) for r in chunk]
            rgbs=[cv2.imread(str(Path(r['folder'])/'rgb.png')) for r in chunk]
            raws=[np.load(r['raw_path']) for r in chunk]
            p8=predicted(model,rgbs,raws)
            opt_ids=[i for i,m in enumerate(metas) if m['scenario'] in ('glare','low_light','depth_failure')]
            p6={i:v for i,v in zip(opt_ids,predicted(optional,[rgbs[i] for i in opt_ids],[raws[i] for i in opt_ids]))} if opt_ids else {}
            low_ids=[i for i,m in enumerate(metas) if m['scenario']=='low_light']
            brighter=[adaptive_exposure_correction(rgbs[i],dict(enabled=True,clahe_clip_limit=1.5)).image_bgr for i in low_ids]
            pe={i:v for i,v in zip(low_ids,predicted(model,brighter,[raws[i] for i in low_ids]))} if low_ids else {}
            for i in range(len(chunk)):
                pending.append(pool.submit(measure,(chunk[i],metas[i],rgbs[i],raws[i],p8[i],p6.get(i),pe.get(i))))
            while len(pending)>32:records.append(pending.popleft().result())
            if start%80==0:print(json.dumps(dict(done=start,total=len(frames),elapsed_s=time.time()-started)),flush=True)
        records.extend(f.result() for f in pending)
    (a.output/'test-raw.json').write_text(json.dumps(dict(records=records,elapsed_s=time.time()-started),allow_nan=False))
    print('COMPLETE',len(records),time.time()-started,flush=True)
if __name__=='__main__':main()
