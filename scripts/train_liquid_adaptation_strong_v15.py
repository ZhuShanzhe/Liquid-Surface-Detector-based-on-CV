#!/usr/bin/env python3
"""Predeclared harder-input ablation; reuse the V15 trainer without copying it."""
import json,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
import train_liquid_adaptation_v15 as t

original_corrupt=t.corrupt

def stronger_corrupt(raw,mode,rng):
 if mode=='random':mode=rng.choice(['clean','scale','scale','scale','local','drop'])
 return original_corrupt(raw,mode,rng)

def huber_relative_loss(depth,truth,valid):
 error=(depth-truth)/(truth*.02).clamp_min(.005)
 loss=F.smooth_l1_loss(error,torch.zeros_like(error),reduction='none',beta=1.)
 count=valid.sum((1,2,3));value=(loss*valid).sum((1,2,3))/count.clamp_min(1)
 return value[count>0].mean()

if __name__=='__main__':
 t.OUT=t.OUT/'strong';t.corrupt=stronger_corrupt;t.depth_loss=huber_relative_loss
 t.main()
 p=t.OUT/'robust/frozen.json'
 if p.exists():
  d=json.loads(p.read_text());d['variant']='robust_strong';d['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();d['variant_definition']='50% global signed 1-6% bias; Huber tolerance-normalized loss instead of log1p';p.write_text(json.dumps(d,indent=2))
