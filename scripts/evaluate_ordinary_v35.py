"""Paired all-frame baseline and geometry-only point-selection ablation."""
import argparse,json,time
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import cv2,numpy as np,torch
from evaluate_other_scenes_v34 import predicted,refined,grid
from evaluate_liquid_height_v21 import Predictor
from liquid_depth.ordinary_surface_v35 import estimate_ordinary
from liquid_depth.liquid_height_metrics import HeightCalibration,reference_statistics
from summarize_other_scenes_v34 import summarize,band


def one(args):
    frame,meta,rgb,raw,pred,fractions=args
    cal=replace(HeightCalibration.from_simulation(meta),bottom_z_m=meta['interior_bottom_z_m'])
    methods={'baseline':refined(rgb,raw,pred['mask'],cal,meta['sensor_family'],'sensor'),
             'raw_grid':grid(raw,pred['mask'],cal)}
    for fraction in fractions:
        start=time.perf_counter()
        p=estimate_ordinary(rgb,raw,cal,meta['container_rim_z_m'],meta['sensor_family'],fraction)
        p['compute_ms']=(time.perf_counter()-start)*1000;methods[f'core_{fraction:g}']=p
    # Labels first accessed after all operational predictions are complete.
    geom=np.load(Path(frame['folder'])/'geometry.npz');truth=geom['mask'].astype(bool)
    ref=reference_statistics(meta,cal);h=ref['mean_m']
    for p in methods.values():
        s=p['statistics'];p['numeric']=bool(s is not None and np.isfinite(s['mean_m']))
        if p['numeric']:
            e=s['mean_m']-h;p.update(signed_error_m=e,absolute_error_m=abs(e),abs_rel=abs(e)/h if h>0 else None,
                within_tolerance=bool(abs(e)<=max(.005,.02*h)),
                extrema_error_m={k:abs(s[k]-ref[k]) for k in ('min_m','max_m','p05_m','p95_m')})
    return dict(frame=frame,cluster=meta['cluster'],sensor=meta['sensor_family'],scenario='ordinary',
        nominal_height_m=meta['nominal_liquid_height_m'],true_height_m=h,reference=ref,
        camera_to_surface_m=meta['camera_position_m'][2],camera_to_bottom_m=meta['camera_to_bottom_m'],
        mask_iou=float((pred['mask']&truth).sum()/max((pred['mask']|truth).sum(),1)),
        predicted_mask_precision=float((pred['mask']&truth).sum()/max(pred['mask'].sum(),1)),
        raw_surface_valid_fraction=float(((raw>0)&truth).sum()/max(truth.sum(),1)),methods=methods)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fractions',default='.65,.85');a=p.parse_args();a.output.mkdir(exist_ok=True)
    fractions=[float(v) for v in a.fractions.split(',')]
    cv2.setNumThreads(1);torch.set_num_threads(4);model=Predictor('v8');records=[]
    frames=[r for f in sorted(a.data.glob('*/frames.json')) for r in json.loads(f.read_text())]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for offset in range(0,len(frames),16):
            chunk=frames[offset:offset+16];metas=[json.loads((Path(f['folder'])/'metadata.json').read_text()) for f in chunk]
            rgb=[cv2.imread(str(Path(f['folder'])/'rgb.png')) for f in chunk];raw=[np.load(f['raw_path']) for f in chunk]
            pred=predicted(model,rgb,raw)
            records.extend(pool.map(one,[(f,m,c,d,q,fractions) for f,m,c,d,q in zip(chunk,metas,rgb,raw,pred)]))
            if offset%160==0:print('DONE',offset,len(frames),flush=True)
    (a.output/'raw.json').write_text(json.dumps(dict(records=records),allow_nan=False))
    groups=defaultdict(list)
    for r in records:
        b=band(r['true_height_m'],r['nominal_height_m']);z=r['camera_to_surface_m'];zb='near' if z<3 else 'far'
        keys=['all',b,'sensor/'+r['sensor'],b+'/'+r['sensor'],zb]
        if z>=3 and r['true_height_m']<1:keys+=['far_shallow','far_shallow/'+r['sensor']]
        for key in keys:groups[key].append(r)
    summary={k:{m:summarize(rows,m) for m in records[0]['methods']} for k,rows in groups.items()}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
