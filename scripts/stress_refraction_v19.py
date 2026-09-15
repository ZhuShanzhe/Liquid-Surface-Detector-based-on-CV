#!/usr/bin/env python3
"""Deliberate sensor-model violations and fixed-camera acquisition failures."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from liquid_depth.refraction_inverse_v19 import estimate_layers,calibrate_empty,ray_factors
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=12);p.add_argument('--seed',type=int,default=77100);p.add_argument('--random-ghost-thickness',action='store_true');a=p.parse_args()
 rows=[]
 for i in range(0,a.count*4,4):
  f=a.data/f'{i:03d}';m=json.loads((f/'scene.json').read_text());c=json.loads((f/'camera.json').read_text());o=dict(np.load(f/'observation.npz'))
  rgb=cv2.cvtColor(cv2.imread(str(f/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  empty=cv2.cvtColor(cv2.imread(str(f/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  raw=np.load(f/'burst_persistent64.npz')['raw'];cal=calibrate_empty(o['empty_raw'])
  rng=np.random.default_rng(a.seed+i);yy,xx=np.indices(raw.shape[1:]);q2=((xx+.5-c['cx'])/c['fx'])**2+((yy+.5-c['cy'])/c['fy'])**2
  affected=cal['roi']&(rng.random(q2.shape)<.015)
  ghost=raw.copy()
  # There is no physical internal interface; an extra stable delayed return mimics one.
  fake_thickness=float(rng.uniform(.006,.045)) if a.random_ghost_thickness else (.010 if i%8==0 else .030)
  fake=m['camera_distance_m']+fake_thickness*ray_factors(m['n_upper'],q2)
  for k in range(len(ghost)):ghost[k][affected]=fake[affected]+rng.normal(0,.001,affected.sum())
  r=estimate_layers(rgb,ghost,empty,o['empty_raw'],c)
  rows.append(dict(id=i,case='stable_ghost_in_single_liquid',physical_internal_interface=False,accepted=r['accepted'],status=r['status'],estimated_thickness_m=r.get('thickness_m'),profile_width_m=r.get('profile_width_m'),result=r))
 for i in [3,11,35,47]:
  f=a.data/f'{i:03d}';c=json.loads((f/'camera.json').read_text());o=dict(np.load(f/'observation.npz'))
  rgb=cv2.cvtColor(cv2.imread(str(f/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255;empty=cv2.cvtColor(cv2.imread(str(f/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  raw=np.load(f/'burst_persistent64.npz')['raw']
  for case in ['moving_surface','missing_depth','unsupported_sensor']:
   z=raw.copy();camera=dict(c)
   if case=='moving_surface':z=np.where(z>0,z+np.linspace(0,.01,len(z))[:,None,None],0)
   elif case=='missing_depth':z[:]=0
   else:camera['sensor_model']='active_stereo'
   r=estimate_layers(rgb,z,empty,o['empty_raw'],camera)
   rows.append(dict(id=i,case=case,accepted=r['accepted'],status=r['status']))
 a.output.write_text(json.dumps(rows,indent=2))
 for case in sorted({r['case'] for r in rows}):
  rs=[r for r in rows if r['case']==case];print(case,len(rs),sum(r['accepted'] for r in rs),'accepted',flush=True)
if __name__=='__main__':main()
