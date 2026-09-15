#!/usr/bin/env python3
"""Evaluate fixed-camera inverse refraction without passing labels to inference."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from liquid_depth.refraction_inverse_v19 import estimate_layers,RefractionConfig

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--burst-mode',choices=['persistent','resampled']);p.add_argument('--frames',type=int,default=16);p.add_argument('--burst-tag',default='');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);rows=[];manifest=json.loads((a.data/'manifest.json').read_text())
 for meta in manifest:
  folder=a.data/f"{meta['id']:03d}"
  raw=dict(np.load(folder/'observation.npz'))
  if a.burst_mode:raw['raw']=np.load(folder/('burst_'+a.burst_mode+a.burst_tag+'.npz'))['raw'][:a.frames]
  rgb=cv2.cvtColor(cv2.imread(str(folder/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  empty=cv2.cvtColor(cv2.imread(str(folder/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  camera=json.loads((folder/'camera.json').read_text())
  for method,use_rgb in [('refraction_raw',False),('refraction_rgbd',True)]:
   start=time.perf_counter()
   try:r=estimate_layers(rgb,raw['raw'],empty,raw['empty_raw'],camera,use_rgb=use_rgb)
   except (ValueError,FloatingPointError) as e:r=dict(status='invalid_observation',accepted=False,thickness_m=None,error=str(e))
   r['elapsed_ms']=(time.perf_counter()-start)*1000
   # Ground truth is accessed only after inference and is never used for acceptance.
   r.update(id=meta['id'],pair=meta['pair'],group=meta['group'],method=method,truth_thickness_m=meta['thickness_m'],truth_top_m=meta['camera_distance_m'],truth_total_m=meta['total_depth_m'],pattern=meta['pattern'])
   r['top_error_mm']=abs(r['top_depth_m']-meta['camera_distance_m'])*1000 if 'top_depth_m' in r else None
   if meta['group']!='single' and r['accepted']:
    r['thickness_error_mm']=abs(r['thickness_m']-meta['thickness_m'])*1000
    r['internal_error_mm']=abs(r['internal_depth_m']-meta['camera_distance_m']-meta['thickness_m'])*1000
    r['both5mm']=max(r['top_error_mm'],r['internal_error_mm'])<=5
    r['truth_in_profile_range']=r['profile_range_m'][0]<=meta['thickness_m']<=r['profile_range_m'][1]
   rows.append(r)
  print('EVAL',meta['id'],meta['group'],[(r['method'],r['status'],r.get('thickness_error_mm')) for r in rows[-2:]],flush=True)
 (a.output/'results.json').write_text(json.dumps(rows,indent=2))
 summary=[]
 for method in ['refraction_raw','refraction_rgbd']:
  for group in ['single','index_matched','weak','strong']:
   rs=[r for r in rows if r['group']==group and r['method']==method];accepted=[r for r in rs if r['accepted']]
   errs=[r['thickness_error_mm'] for r in accepted if 'thickness_error_mm' in r]
   record=dict(method=method,group=group,scenes=len(rs),accepted=len(accepted),coverage_pct=100*len(accepted)/len(rs),thickness_mae_mm=float(np.mean(errs)) if errs else None,thickness_p95_mm=float(np.percentile(errs,95)) if errs else None,both5mm_all_scenes_pct=100*sum(r.get('both5mm',False) for r in rs)/len(rs),statuses=dict(Counter(r['status'] for r in rs)),median_ms=float(np.median([r['elapsed_ms'] for r in rs])))
   summary.append(record);print('SUMMARY',record,flush=True)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
