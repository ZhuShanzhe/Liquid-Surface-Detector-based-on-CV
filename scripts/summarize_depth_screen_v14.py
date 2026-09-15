#!/usr/bin/env python3
"""Paired independent-scene bootstrap; no method fitting or threshold selection."""
import argparse,json,csv
from pathlib import Path
import numpy as np

def compare(a,b,scenario,view='all'):
 def aggregate(d):
  result={}
  for f in d['frames']:
   if scenario!='all' and f['scenario']!=scenario:continue
   k=f['sequence_id'];r=result.setdefault(k,{'good':0,'possible':0})
   r['good']+=f[view]['good'];r['possible']+=f[view]['possible']
  return result
 ar=aggregate(a);br=aggregate(b)
 assert ar.keys()==br.keys()
 keys=sorted(ar);n=len(keys)
 ag=np.array([ar[k]['good'] for k in keys]);bg=np.array([br[k]['good'] for k in keys]);p=np.array([ar[k]['possible'] for k in keys])
 assert np.array_equal(p,np.array([br[k]['possible'] for k in keys]))
 rng=np.random.default_rng(20260915109);ix=rng.integers(0,n,(2000,n))
 delta=100*(ag[ix].sum(1)-bg[ix].sum(1))/np.maximum(1,p[ix].sum(1))
 return dict(delta_pp=float(100*(ag.sum()-bg.sum())/max(1,p.sum())),ci95_pp=np.percentile(delta,[2.5,97.5]).tolist(),independent_scenes=n)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();root=Path(a.run_dir)
 data={}
 for split in ('dev','fresh','stress'):
  data[split]={p.stem[len(split)+1:]:json.loads(p.read_text()) for p in root.glob(split+'-*.json')}
 rows=[];comparison={}
 for split,methods in data.items():
  if not methods:continue
  comparison[split]={}
  for name,d in methods.items():
   comparison[split][name]={}
   for scene in ('all','transparent','translucent','multilayer','compound'):
    v=d['summary'][scene]['all'].copy()
    chosen=[f['all'] for f in d['frames'] if scene=='all' or f['scenario']==scene]
    frame_rel=[f['rel_sum']/f['n'] for f in chosen if f['n']]
    v['frame_absrel_p95']=float(np.percentile(frame_rel,95)) if frame_rel else None
    v['empty_frames']=sum(f['n']==0 for f in chosen)
    row=dict(split=split,method=name,scene=scene,**v,latency_median_ms=d['latency_ms']['median'],latency_p95_ms=d['latency_ms']['p95'])
    if name!='v12' and 'v12' in methods:comparison[split][name][scene]=compare(d,methods['v12'],scene)
    rows.append(row)
 with (root/'screening_table.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (root/'comparison.json').write_text(json.dumps(comparison,indent=2))
 brief={s:{m:dict(summary=d['summary'],latency_ms=d['latency_ms']) for m,d in methods.items()} for s,methods in data.items()}
 (root/'summary.json').write_text(json.dumps(brief,indent=2))
 for split,methods in data.items():
  print(split)
  for name,d in methods.items():
   print(name,' | '.join(f"{sc}: MAE {d['summary'][sc]['all']['mae_mm'] if d['summary'][sc]['all']['mae_mm'] is not None else -1:.2f}, good {100*(d['summary'][sc]['all']['success'] or 0):.2f}%, cov {100*(d['summary'][sc]['all']['coverage'] or 0):.1f}%" for sc in ('transparent','translucent')))
if __name__=='__main__':main()
