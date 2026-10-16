"""Causal paired endpoints; disjoint 4-frame windows, no truth in inference."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import cv2,numpy as np
from liquid_depth.liquid_height_metrics import HeightCalibration,reference_statistics
from liquid_depth.ordinary_temporal_v37 import estimate_window,estimate_guarded4
from summarize_other_scenes_v34 import summarize,band

def score(p,ref):
    s=p['statistics'];p['numeric']=bool(s is not None and np.isfinite(s['mean_m']))
    if p['numeric']:
        h=ref['mean_m'];e=s['mean_m']-h
        p.update(signed_error_m=e,absolute_error_m=abs(e),abs_rel=abs(e)/h if h>0 else None,
            within_tolerance=bool(abs(e)<=max(.005,.02*h)),extrema_error_m={k:abs(s[k]-ref[k]) for k in ('min_m','max_m','p05_m','p95_m')})

def one(args):
    folder,mode=args;m=json.loads((folder/'metadata.json').read_text())
    cal=replace(HeightCalibration.from_simulation(m),bottom_z_m=m['interior_bottom_z_m'])
    frames=sorted([r for r in json.loads((folder/'frames.json').read_text()) if r['variant']==mode],key=lambda r:r['repeat'])
    raw=[np.load(f['raw_path']) for f in frames];single=[];outputs=[]
    for i,d in enumerate(raw):
        start=time.perf_counter();p=estimate_window([d],cal,m['container_rim_z_m'],m['sensor_family'])
        p['compute_ms']=(time.perf_counter()-start)*1000;single.append(p)
    for endpoint in range(3,len(frames),4):
        methods={'single':single[endpoint]}
        for n in (2,4):
            start=time.perf_counter()
            p=estimate_window(raw[endpoint-n+1:endpoint+1],cal,m['container_rim_z_m'],m['sensor_family'],
                timestamps=[f['timestamp'] for f in frames[endpoint-n+1:endpoint+1]])
            p['compute_ms']=(time.perf_counter()-start)*1000;methods[f'joint{n}']=p
        # Reference baseline: arithmetic mean of independently estimated heights.
        previous=single[endpoint-3:endpoint+1]
        s=None
        if all(p['statistics'] is not None for p in previous):s={k:float(np.mean([p['statistics'][k] for p in previous])) for k in previous[0]['statistics']}
        methods['mean4']=dict(statistics=s,qualified=False,reasons=[] if s else ['some_single_frames_unobservable'])
        start=time.perf_counter()
        methods['guarded4']=estimate_guarded4(raw[endpoint-3:endpoint+1],cal,m['container_rim_z_m'],m['sensor_family'],
            timestamps=[f['timestamp'] for f in frames[endpoint-3:endpoint+1]])
        methods['guarded4']['compute_ms']=(time.perf_counter()-start)*1000
        # All inference above precedes ground-truth reference access.
        ref=reference_statistics(m,cal)
        for p in methods.values():score(p,ref)
        outputs.append(dict(frame=frames[endpoint],cluster=m['cluster'],sensor=m['sensor_family'],scenario='ordinary',noise=mode,
            nominal_height_m=m['nominal_liquid_height_m'],true_height_m=ref['mean_m'],reference=ref,
            camera_to_surface_m=m['camera_position_m'][2],camera_to_bottom_m=m['camera_to_bottom_m'],
            mask_iou=0.,predicted_mask_precision=0.,raw_surface_valid_fraction=0.,methods=methods))
    return outputs

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cv2.setNumThreads(1);rows=[];start=time.time();jobs=[(f.parent,mode) for f in sorted(a.data.glob('*/frames.json')) for mode in ('iid','spatial','persistent')]
    if not jobs:raise RuntimeError('No generated sequences')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,q in enumerate(pool.map(one,jobs)):
            rows.extend(q)
            if i%30==0:print('DONE',i,len(jobs),time.time()-start,flush=True)
    (a.output/'raw.json').write_text(json.dumps(dict(records=rows),allow_nan=False));groups=defaultdict(list)
    for r in rows:
        mode=r['noise'];b=band(r['true_height_m'],r['nominal_height_m']);sensor=r['sensor'];z=r['camera_to_surface_m']
        keys=['all',b,'sensor/'+sensor,b+'/'+sensor,'near' if z<3 else 'far']
        if z>=3 and r['true_height_m']<1:keys+=['far_shallow','far_shallow/'+sensor]
        for key in keys:groups[mode+'/'+key].append(r)
    summary={k:{m:summarize(q,m) for m in rows[0]['methods']} for k,q in groups.items()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print('COMPLETE',len(rows),time.time()-start,flush=True)
if __name__=='__main__':main()
