"""V35 versus V36, same observations, labels used only after inference."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import cv2,numpy as np
from liquid_depth.ordinary_surface_v35 import estimate_ordinary
from liquid_depth.ordinary_surface_v36 import estimate_v36
from liquid_depth.liquid_height_metrics import HeightCalibration,reference_statistics
from summarize_other_scenes_v34 import summarize,band

def one(args):
    frame,ablate=args;m=json.loads((Path(frame['folder'])/'metadata.json').read_text())
    raw=np.load(frame['raw_path']);rgb=None
    cal=replace(HeightCalibration.from_simulation(m),bottom_z_m=m['interior_bottom_z_m'])
    methods={};options=[('v35',None),('v36',dict(mode='quant_echo',wide_near=True,uniform=True))]
    if ablate:options += [('quant_only',dict(mode='quant',wide_near=False)),('quant_echo',dict(mode='quant_echo',wide_near=False)),('weighted',dict(mode='quant_echo',wide_near=True,uniform=False))]
    for name,opt in options:
        start=time.perf_counter()
        p=estimate_ordinary(rgb,raw,cal,m['container_rim_z_m'],m['sensor_family'],0) if opt is None else estimate_v36(rgb,raw,cal,m['container_rim_z_m'],m['sensor_family'],**opt)
        p['compute_ms']=(time.perf_counter()-start)*1000;methods[name]=p
    ref=reference_statistics(m,cal);h=ref['mean_m']
    for p in methods.values():
        s=p['statistics'];p['numeric']=bool(s is not None and np.isfinite(s['mean_m']))
        if p['numeric']:
            e=s['mean_m']-h;p.update(signed_error_m=e,absolute_error_m=abs(e),abs_rel=abs(e)/h if h>0 else None,
                within_tolerance=bool(abs(e)<=max(.005,.02*h)),extrema_error_m={k:abs(s[k]-ref[k]) for k in ('min_m','max_m','p05_m','p95_m')})
    return dict(frame=frame,cluster=m['cluster'],sensor=m['sensor_family'],scenario='ordinary',nominal_height_m=m['nominal_liquid_height_m'],
        true_height_m=h,reference=ref,camera_to_surface_m=m['camera_position_m'][2],camera_to_bottom_m=m['camera_to_bottom_m'],
        mask_iou=0.,predicted_mask_precision=0.,raw_surface_valid_fraction=0.,methods=methods)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ablate',action='store_true');a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True);cv2.setNumThreads(1)
    frames=[r for f in sorted(a.data.glob('*/frames.json')) for r in json.loads(f.read_text())]
    rows=[];start=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,r in enumerate(pool.map(one,[(f,a.ablate) for f in frames])):
            rows.append(r)
            if i%160==0:print('DONE',i,len(frames),time.time()-start,flush=True)
    (a.output/'raw.json').write_text(json.dumps(dict(records=rows),allow_nan=False));groups=defaultdict(list)
    for r in rows:
        b=band(r['true_height_m'],r['nominal_height_m']);sensor=r['sensor']
        keys=['all',b,'sensor/'+sensor,b+'/'+sensor,'near' if r['camera_to_surface_m']<3 else 'far']
        if r['camera_to_surface_m']>=3 and r['true_height_m']<1:keys+=['far_shallow','far_shallow/'+sensor]
        for key in keys:groups[key].append(r)
    summary={k:{m:summarize(q,m) for m in rows[0]['methods']} for k,q in groups.items()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print('COMPLETE',len(rows),time.time()-start,flush=True)
if __name__=='__main__':main()
