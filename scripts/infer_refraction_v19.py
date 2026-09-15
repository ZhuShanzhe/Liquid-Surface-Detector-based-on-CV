#!/usr/bin/env python3
"""Run the experimental V19 inverse model from observable files only."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from liquid_depth.refraction_inverse_v19 import estimate_layers
from liquid_depth.refraction_evidence_v19 import add_chromatic_evidence,qualify_profile

def image(path):
 x=cv2.imread(str(path))
 if x is None:raise ValueError('Cannot read image: '+str(path))
 return cv2.cvtColor(x,cv2.COLOR_BGR2RGB).astype(np.float32)/255
def depth(path,key):
 if path.suffix=='.npy':return np.load(path,allow_pickle=False)
 with np.load(path,allow_pickle=False) as data:return data[key]
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['rgb','raw','empty-rgb','empty-depth','camera','output']:p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--geometry-only',action='store_true',help='Diagnostic only: bypass the competing homogeneous-medium check')
 p.add_argument('--raw-key',default='raw');p.add_argument('--empty-depth-key',default='empty_raw');p.add_argument('--raw-only',action='store_true')
 a=p.parse_args()
 rgb=image(a.rgb);empty=image(a.empty_rgb);empty_z=depth(a.empty_depth,a.empty_depth_key);camera=json.loads(a.camera.read_text())
 result=estimate_layers(rgb,depth(a.raw,a.raw_key),empty,empty_z,camera,use_rgb=not a.raw_only)
 if not a.geometry_only:
  result=qualify_profile(result,weight=0.,channel='photo_error') if a.raw_only else qualify_profile(add_chromatic_evidence(result,rgb,empty,empty_z,camera))
 result['result_scope']='geometry_under_declared_single_path_tof_proxy'
 result['liquid_identity_verified']=False
 result['note']='Geometric acceptance is not proof that a stable return is a physical liquid interface. Real camera qualification is required.'
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='profile'},indent=2))
if __name__=='__main__':main()
