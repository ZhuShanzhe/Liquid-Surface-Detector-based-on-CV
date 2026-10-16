#!/usr/bin/env python3
"""Label-free command-line inference with an explicit metric calibration file."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from evaluate_liquid_height_v21 import Predictor,WEIGHTS,digest
from liquid_depth.liquid_height_metrics import HeightCalibration,estimate_height
from liquid_depth.models.liquid_height_head import CalibratedHeightPredictor

p=argparse.ArgumentParser();p.add_argument('--rgb',required=True);p.add_argument('--raw',required=True)
p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
p.add_argument('--backend',choices=['v6','v8','v17','v16_plane','raw','v17_raw'],default='v17');p.add_argument('--head')
a=p.parse_args();out=Path(a.output)
if a.head and a.backend in ('raw','v17_raw'):
    raise ValueError('The frozen residual head was not trained for a raw-point backend')
if out.exists():raise FileExistsError(out)
torch.set_num_threads(4);cv2.setNumThreads(0)
m=json.loads(Path(a.calibration).read_text())
# No simulation metadata or target fields accepted by this inference interface.
allowed={'intrinsics','camera_to_world_cv','bottom_z_m','radii_m','footprint_exponent'}
if set(m)!=allowed:raise ValueError('Calibration must contain exactly the declared geometry fields')
cal=HeightCalibration(np.asarray(m['intrinsics']),np.asarray(m['camera_to_world_cv']),
    float(m['bottom_z_m']),tuple(m['radii_m']),float(m['footprint_exponent']))
rgb=cv2.cvtColor(cv2.imread(a.rgb),cv2.COLOR_BGR2RGB).astype(np.float32)/255
raw=np.load(a.raw).astype(np.float32)
model=Predictor(a.backend)
depth,mask,_=model(torch.from_numpy(rgb.transpose(2,0,1)[None]).cuda(),torch.from_numpy(raw[None,None]).cuda())
depth,mask=depth[0,0].cpu().numpy(),mask[0,0].cpu().numpy()>=.5
prediction=estimate_height(depth,mask,cal);raw_prediction=estimate_height(raw,mask,cal)
if a.head:
    prediction=CalibratedHeightPredictor(a.head,digest(WEIGHTS[a.backend])).correct(prediction,raw_prediction,raw,mask,cal)
prediction.update(metric_contract_version=3,measurand='area_mean_vertical_liquid_depth_above_container_bottom_m',
    research_only=True,hardware_qualified=False,liquid_identity_verified=False,
    calibrated_probability_available=False,base_checkpoint_sha256=digest(WEIGHTS.get(a.backend,WEIGHTS['v8'])))
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(prediction,indent=2,allow_nan=False))
print(json.dumps(prediction))
