#!/usr/bin/env python3
"""Generate sensor-proxy sequences for fixed-camera static scenes, not measured sensor data."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('--frames',type=int,default=16);p.add_argument('--tag',default='');p.add_argument('--modes',nargs='+',choices=['persistent','resampled'],default=['persistent','resampled']);a=p.parse_args()
 for meta in json.loads((a.data/'manifest.json').read_text()):
  folder=a.data/f"{meta['id']:03d}";c=meta['camera'];f=c['fx']
  vv,uu=np.indices((180,320));x=(uu+.5-c['cx'])/f;y=(vv+.5-c['cy'])/f
  q2=x*x+y*y;cos0=1/np.sqrt(1+q2);s2=q2/(1+q2)
  n1=meta['n_upper'];n2=meta['n_lower'];d=meta['camera_distance_m'];H=meta['total_depth_m'];t=meta['thickness_m'] or H
  cos1=np.sqrt(1-s2/n1**2);cos2=np.sqrt(1-s2/n2**2)
  surfaces=np.stack([np.full_like(x,d),d+n1*t*cos0/cos1,d+n1*t*cos0/cos1+n2*(H-t)*cos0/cos2])
  mask=(abs(x*d)<.137)&(abs(y*d)<.087)
  for mode in a.modes:
   rng=np.random.default_rng(meta['seed']*331+meta['pair'])
   stable_draw=rng.random(x.shape)
   # A fixed per-pixel offset remains across the sequence; averaging cannot remove it.
   fixed_bias=rng.normal(0,.00025,x.shape)
   frames=[]
   for k in range(a.frames):
    draw=stable_draw if mode=='persistent' else rng.random(x.shape)
    event=np.searchsorted(np.cumsum(meta['raw_event_weights']),draw)
    raw=np.take_along_axis(surfaces,event[None],axis=0)[0]
    raw=np.where(mask,raw,d+H+.015)
    raw=raw+fixed_bias+rng.normal(0,meta['sensor_sigma']+.001*raw**2,x.shape)
    raw[rng.random(x.shape)<meta['drop']]=0
    frames.append(np.maximum(np.round(raw*1000)/1000,0).astype(np.float32))
   dest=folder/('burst_'+mode+a.tag+'.npz')
   if dest.exists():raise FileExistsError(dest)
   np.savez_compressed(dest,raw=np.stack(frames))
 print('BURSTS_FINISHED',len(json.loads((a.data/'manifest.json').read_text())),a.frames,flush=True)
if __name__=='__main__':main()
