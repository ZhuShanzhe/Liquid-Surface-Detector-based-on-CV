#!/usr/bin/env python3
"""Replay development-frozen chromatic evidence on untouched geometry profiles."""
import argparse,json,hashlib,time
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from liquid_depth.refraction_evidence_v19 import add_chromatic_evidence,qualify_profile

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stress',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False);inputs=json.loads(a.results.read_text());rows=[]
 if a.stress:inputs=[dict(s['result'],id=s['id'],case=s['case']) for s in inputs if 'result' in s]
 else:inputs=[r for r in inputs if r['method']=='refraction_rgbd']
 for r in inputs:
  folder=a.data/f"{r['id']:03d}";camera=json.loads((folder/'camera.json').read_text())
  rgb=cv2.cvtColor(cv2.imread(str(folder/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  empty=cv2.cvtColor(cv2.imread(str(folder/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  start=time.perf_counter();q=qualify_profile(add_chromatic_evidence(r,rgb,empty,np.load(folder/'observation.npz')['empty_raw'],camera),weight=30000.,contrast_floor=.02)
  q['evidence_ms']=(time.perf_counter()-start)*1000
  for key in ['thickness_error_mm','internal_error_mm','both5mm','truth_in_profile_range']:q.pop(key,None)
  if not a.stress and q['accepted'] and q['group']!='single':
   q['thickness_error_mm']=abs(q['thickness_m']-q['truth_thickness_m'])*1000
   q['internal_error_mm']=abs(q['internal_depth_m']-q['truth_top_m']-q['truth_thickness_m'])*1000
   q['both5mm']=max(q['top_error_mm'],q['internal_error_mm'])<=5
  rows.append(q)
 (a.output/'results.json').write_text(json.dumps(rows,indent=2))
 summary=[]
 for group in (['stable_ghost_in_single_liquid'] if a.stress else ['single','index_matched','weak','strong']):
  rs=[r for r in rows if r.get('group',r.get('case'))==group];acc=[r for r in rs if r['accepted']];errs=[r['thickness_error_mm'] for r in acc if 'thickness_error_mm' in r]
  row=dict(method='chromatic_refraction_evidence',group=group,scenes=len(rs),accepted=len(acc),coverage_pct=100*len(acc)/len(rs),thickness_mae_mm=float(np.mean(errs)) if errs else None,thickness_p95_mm=float(np.percentile(errs,95)) if errs else None,both5mm_all_scenes_pct=100*sum(r.get('both5mm',False) for r in rs)/len(rs),statuses=dict(Counter(r['status'] for r in rs)))
  summary.append(row);print(row,flush=True)
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
