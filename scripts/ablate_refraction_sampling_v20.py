"""Controlled sampling ablation: round coordinates only; keep all other V20 settings."""
import json,argparse
from pathlib import Path
from collections import Counter
import cv2,numpy as np
import liquid_depth.refraction_evidence_v20 as ev
from liquid_depth.refraction_calibration_v19 import apply_reference_quality
from liquid_depth.refraction_evidence_v19 import add_chromatic_evidence as old_photo,qualify_profile as old_qualify
exact=ev.map_coordinates
def quantized(input,coordinates,**kwargs):
 return exact(input,np.round(np.asarray(coordinates)*32)/32,**kwargs)
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--stress',action='store_true');a=p.parse_args()
 rows=json.loads(a.results.read_text())
 if a.stress:rows=[dict(r['result'],id=r['id'],group='ghost') for r in rows if 'result' in r]
 out=[]
 for r in rows:
  f=a.data/f"{r['id']:03d}";camera=json.loads((f/'camera.json').read_text());er=np.load(f/'observation.npz')['empty_raw']
  rgb=cv2.cvtColor(cv2.imread(str(f/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  empty=cv2.cvtColor(cv2.imread(str(f/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  for mode in ['v20_continuous','v20_coordinate_quantized','v20_depth_with_v19_photo']:
   if mode=='v20_depth_with_v19_photo':q=old_qualify(old_photo(r,rgb,empty,er,camera))
   else:
    ev.map_coordinates=exact if mode=='v20_continuous' else quantized
    q=ev.qualify_profile(ev.add_chromatic_evidence(r,rgb,empty,er,camera))
   q=apply_reference_quality(q,empty,er,camera)
   q['ablation']=mode
   for k in ['thickness_error_mm','internal_error_mm','both5mm']:q.pop(k,None)
   if q['accepted'] and q.get('truth_thickness_m') is not None:
    q['thickness_error_mm']=abs(q['thickness_m']-q['truth_thickness_m'])*1000
   q.pop('profile',None);out.append(q)
 summary=[]
 for mode in sorted({r['ablation'] for r in out}):
  for group in sorted({r['group'] for r in out}):
   rs=[r for r in out if r['group']==group and r['ablation']==mode];acc=[r for r in rs if r['accepted']]
   errs=[r['thickness_error_mm'] for r in acc if 'thickness_error_mm' in r]
   summary.append(dict(ablation=mode,group=group,scenes=len(rs),accepted=len(acc),mae_mm=float(np.mean(errs)) if errs else None,statuses=dict(Counter(r['status'] for r in rs))))
 a.output.mkdir(exist_ok=False);(a.output/'summary.json').write_text(json.dumps(summary,indent=2));(a.output/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
