#!/usr/bin/env python3
"""Frozen neural baselines with the same observable empty-container calibration."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from evaluate_stratified_seegroup_v18 import load_original,load_v9
from liquid_depth.models.observable_v16 import encode_rgbd
from liquid_depth.refraction_inverse_v19 import calibrate_empty

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--physics-results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--burst-tag',default='64');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 measured={r['id']:{k:r.get(k) for k in ['top_depth_m','bottom_depth_m']} for r in json.loads(a.physics_results.read_text()) if r['method']=='refraction_raw'}
 manifest=json.loads((a.data/'manifest.json').read_text());rows=[]
 for method in ['seegroup_observed_calibration','v9_1_temporal']:
  model=load_original() if method.startswith('seegroup') else load_v9()
  for meta in manifest:
   folder=a.data/f"{meta['id']:03d}";data=dict(np.load(folder/'observation.npz'));cal=calibrate_empty(data['empty_raw']);roi=cal['roi']
   rgb=cv2.cvtColor(cv2.imread(str(folder/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
   x=torch.from_numpy(rgb.transpose(2,0,1))[None].cuda()
   with torch.inference_mode():
    torch.cuda.synchronize();start=time.perf_counter()
    if method.startswith('seegroup'):
     x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
     output=model.forward_unscale({'image':x});depth=output['depth'][0].cpu().numpy();active=output['valid_mask'][0].cpu().numpy()>0
    else:
     frames=np.load(folder/('burst_persistent'+a.burst_tag+'.npz'))['raw']
     raw=np.nanmedian(np.where(frames>0,frames,np.nan),axis=0);raw=np.nan_to_num(raw)
     output=model(encode_rgbd(x,torch.from_numpy(raw[None,None]).cuda()))
     depth=output['layer_depths_m'][0].cpu().numpy();active=output['layer_presence_logits'][0].sigmoid().cpu().numpy()>=.5
    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
   ordered=np.sort(np.where(active&np.isfinite(depth)&(depth>0),depth,np.inf),axis=0)
   support=roi&np.isfinite(ordered[1])
   r=dict(id=meta['id'],pair=meta['pair'],group=meta['group'],method=method,accepted=False,elapsed_ms=elapsed,support_fraction=float(support.sum()/roi.sum()))
   if support.sum()>=roi.sum()*.5:
    first=float(np.median(ordered[0][support]));second=float(np.median(ordered[1][support]))
    if method.startswith('seegroup'):
     last=np.max(np.where(active,depth,-np.inf),axis=0);far=float(np.median(last[support]));m=measured[meta['id']]
     if far>first and m['top_depth_m'] is not None:
      scale=(m['bottom_depth_m']-m['top_depth_m'])/(far-first)
      second=m['top_depth_m']+(second-first)*scale;first=m['top_depth_m']
     else:first=second=np.nan
    if np.isfinite(first) and np.isfinite(second):
     r.update(accepted=True,top_depth_m=first,internal_depth_m=second,thickness_m=second-first,top_error_mm=abs(first-meta['camera_distance_m'])*1000)
     if meta['group']!='single':
      r['thickness_error_mm']=abs(second-first-meta['thickness_m'])*1000
      r['internal_error_mm']=abs(second-meta['camera_distance_m']-meta['thickness_m'])*1000
      r['both5mm']=max(r['top_error_mm'],r['internal_error_mm'])<=5
   rows.append(r)
   if meta['id']%16==15:print('BASELINE',method,meta['id']+1,flush=True)
  del model;torch.cuda.empty_cache()
 (a.output/'results.json').write_text(json.dumps(rows,indent=2))
 summary=[]
 for method in ['seegroup_observed_calibration','v9_1_temporal']:
  for group in ['single','index_matched','weak','strong']:
   rs=[r for r in rows if r['method']==method and r['group']==group];acc=[r for r in rs if r['accepted']];errors=[r['thickness_error_mm'] for r in acc if 'thickness_error_mm' in r]
   rec=dict(method=method,group=group,scenes=len(rs),accepted=len(acc),coverage_pct=100*len(acc)/len(rs),thickness_mae_mm=float(np.mean(errors)) if errors else None,thickness_p95_mm=float(np.percentile(errors,95)) if errors else None,both5mm_all_scenes_pct=100*sum(r.get('both5mm',False) for r in rs)/len(rs))
   summary.append(rec);print('SUMMARY',rec,flush=True)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
