#!/usr/bin/env python3
"""Report whether an RGB-D manifest actually includes non-liquid depth context."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np

def load(path):
 a=np.load(path) if path.suffix=='.npy' else cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
 if a is None:raise FileNotFoundError(path)
 return a

def audit(manifest):
 manifest=Path(manifest);groups={}
 for row in csv.DictReader(manifest.open()):
  raw=load(manifest.parent/row['raw_depth_path']);mask=load(manifest.parent/row['mask_path'])>0
  if raw.ndim==3:raw=raw[...,0]
  if mask.ndim==3:mask=mask.any(2)
  if raw.shape!=mask.shape:raise ValueError('Depth/mask shapes differ')
  valid=np.isfinite(raw)&(raw>0);g=groups.setdefault(row.get('scenario','all'),dict(frames=0,outside_possible=0,outside_measured=0,inside_possible=0,inside_measured=0))
  g['frames']+=1;g['outside_possible']+=int((~mask).sum());g['outside_measured']+=int((valid&~mask).sum());g['inside_possible']+=int(mask.sum());g['inside_measured']+=int((valid&mask).sum())
 for g in groups.values():
  g['outside_valid_fraction']=g['outside_measured']/g['outside_possible'] if g['outside_possible'] else None
  g['inside_valid_fraction']=g['inside_measured']/g['inside_possible'] if g['inside_possible'] else None
 return groups

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output');ap.add_argument('--expect-full-frame',action='store_true');a=ap.parse_args()
 result=audit(a.manifest);text=json.dumps(result,indent=2)
 if a.output:Path(a.output).write_text(text)
 print(text)
 if a.expect_full_frame and any(g['outside_possible']>0 and g['outside_measured']==0 for g in result.values()):raise SystemExit('No measured background depth: do not describe this as a full-frame RGB-D benchmark without verifying its input contract.')
if __name__=='__main__':main()
