"""Post-hoc label-free geometry-fence ablation; NOT an independent promotion test."""
import argparse,hashlib,json,time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import cv2,numpy as np,torch
from evaluate_other_scenes_v34 import predicted,refined
from evaluate_liquid_height_v21 import Predictor
from liquid_depth.liquid_height_metrics import HeightCalibration,world_points
from summarize_other_scenes_v34 import summarize,band

def installation_roi(meta,cal,shape):
    # Declared container opening geometry; never a liquid height or GT surface mask.
    t=np.linspace(0,2*np.pi,256,endpoint=False);e=cal.footprint_exponent
    xy=np.column_stack((np.sign(np.cos(t))*abs(np.cos(t))**(2/e)*cal.radii_m[0],
        np.sign(np.sin(t))*abs(np.sin(t))**(2/e)*cal.radii_m[1]))
    xyz=np.column_stack((xy,np.full(len(xy),meta['container_rim_z_m'])))
    pose=cal.camera_to_world_cv;cam=(xyz-pose[:3,3])@pose[:3,:3]
    uv=cam[:,:2]/cam[:,2,None]@cal.intrinsics[:2,:2].T+cal.intrinsics[:2,2]
    roi=np.zeros(shape,np.uint8);cv2.fillPoly(roi,[np.rint(uv).astype(np.int32)],1)
    return roi.astype(bool)

def one(args):
    old,meta,rgb,raw,pred=args
    cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
    points=world_points(raw,cal)
    inside=((abs(points[...,:2]/cal.radii_m)**cal.footprint_exponent).sum(-1)<=1)
    roi=installation_roi(meta,cal,raw.shape)
    mode='partition' if meta['scenario']=='uneven_surface' else 'sensor'
    methods={name:refined(rgb,raw,mask,cal,meta['sensor_family'],mode) for name,mask in (
        ('predicted_plus_footprint',pred['mask']&inside),
        ('predicted_plus_opening',pred['mask']&roi),
        ('opening_only',roi))}
    # Compare with the same already evaluated frames; truths are accessed only here.
    reference=old['reference']
    for p in methods.values():
        s=p['statistics'];p['numeric']=bool(s is not None and np.isfinite(s['mean_m']))
        if p['numeric']:
            err=s['mean_m']-reference['mean_m']
            p.update(signed_error_m=err,absolute_error_m=abs(err),
                abs_rel=abs(err)/reference['mean_m'] if old['nominal_height_m']>0 else None,
                within_tolerance=bool(abs(err)<=max(.005,.02*reference['mean_m'])),
                extrema_error_m={k:abs(s[k]-reference[k]) for k in ('min_m','max_m','p05_m','p95_m')})
    r={**old,'methods':{**old['methods'],**methods}}
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (a.output/'roi-diagnostic-frozen.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='post-hoc diagnostic on existing fresh test, not independent confirmation',
        methods=['predicted_plus_footprint','predicted_plus_opening','opening_only'],
        added_calibration='known container opening height and outline, no liquid depth labels',
        modifies_production=False),indent=2))
    old=json.loads((a.output/'test-raw.json').read_text())['records']
    cv2.setNumThreads(1);torch.set_num_threads(4);model=Predictor('v8');records=[];start=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(old),16):
            batch=old[offset:offset+16]
            metas=[json.loads((Path(r['frame']['folder'])/'metadata.json').read_text()) for r in batch]
            rgb=[cv2.imread(str(Path(r['frame']['folder'])/'rgb.png')) for r in batch]
            raw=[np.load(r['frame']['raw_path']) for r in batch]
            predictions=predicted(model,rgb,raw)
            records.extend(pool.map(one,zip(batch,metas,rgb,raw,predictions)))
            if offset%160==0:print('DONE',offset,len(old),time.time()-start,flush=True)
    (a.output/'roi-diagnostic-raw.json').write_text(json.dumps(dict(records=records),allow_nan=False))
    groups=defaultdict(list)
    for r in records:
        scene=r['scenario']+('/'+r['frame']['variant'] if r['scenario']=='depth_failure' else '')
        b=band(r['true_height_m'],r['nominal_height_m']);z=r['camera_to_surface_m']
        zb='0-1' if z<1 else '1-3' if z<3 else '3-6' if z<6 else '6-10'
        for key in (scene,scene+'/'+b,scene+'/Z='+zb):groups[key].append(r)
    summary={key:{m:summarize(rows,m) for m in ('raw_grid','refined','predicted_plus_footprint','predicted_plus_opening','opening_only','oracle_mask_refined')} for key,rows in groups.items()}
    (a.output/'roi-diagnostic-analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('COMPLETE',len(records),time.time()-start,flush=True)
if __name__=='__main__':main()
