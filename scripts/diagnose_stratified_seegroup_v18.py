#!/usr/bin/env python3
"""Postprocessing and GT top/bottom affine diagnostic; no checkpoint changes."""
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_stratified_seegroup_v18 import report_frame
def suppress_components(depth, active, separation=.02):
    order=np.argsort(depth,axis=0)
    d=np.take_along_axis(depth,order,axis=0)
    valid=np.take_along_axis(active,order,axis=0)&np.isfinite(d)&(d>separation)
    keep=np.zeros_like(valid);last=np.full(d.shape[1:],-np.inf)
    for k in range(len(d)):
        good=valid[k]&((d[k]-last)>separation)
        keep[k]=good;last=np.where(good,d[k],last)
    return d,keep
def align_top_bottom(depth,active,roi,top,bottom):
    valid=active&np.isfinite(depth)&(depth>0)
    support=roi&(valid.sum(0)>=2)
    if not support.any():return depth*0,active&False,None
    lo=np.median(np.where(valid,depth,np.inf).min(0)[support])
    hi=np.median(np.where(valid,depth,-np.inf).max(0)[support])
    if not hi>lo:return depth*0,active&False,None
    scale=(bottom-top)/(hi-lo);bias=top-scale*lo
    return depth*scale+bias,active,dict(scale=float(scale),bias=float(bias),support=int(support.sum()))
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records={k:[] for k in ['official_dedup_sensor_scale','official_oracle_top_bottom','official_dedup_oracle_top_bottom']}
    for i in range(64):
        meta=json.loads((a.data/f'{i:03d}/scene.json').read_text());arrays=dict(np.load(a.data/f'{i:03d}/arrays.npz'))
        z=np.load(a.output/f'predictions/official_native-{i:03d}.npz');d,active=z['depth'],z['active']
        dd,aa=suppress_components(d,active)
        first=np.where(aa,dd,np.inf).min(0);support=np.isfinite(first)&(arrays['raw']>0)
        scale=float(np.median(arrays['raw'][support])/np.median(first[support])) if support.any() else 0.
        r=report_frame(dd*scale,aa,arrays,meta);r['sensor_scale']=scale;records['official_dedup_sensor_scale'].append(r)
        for name,x,v in [('official_oracle_top_bottom',d,active),('official_dedup_oracle_top_bottom',dd,aa)]:
            y,v,fit=align_top_bottom(x,v,arrays['roi'],meta['camera_distance_m'],meta['camera_distance_m']+.1001)
            r=report_frame(y,v,arrays,meta);r['oracle_fit']=fit;records[name].append(r)
    for k,v in records.items():(a.output/(k+'.json')).write_text(json.dumps(v))
    print('DIAGNOSTICS_FINISHED',flush=True)
if __name__=='__main__':main()
