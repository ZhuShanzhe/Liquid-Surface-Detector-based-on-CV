"""No-USB measurement entry point using ordinary installation artifacts."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor
from liquid_depth.supported_scene_meter import FormalSceneMeter


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--dry',type=Path,required=True)
    p.add_argument('--depth',type=Path,required=True);p.add_argument('--rgb',type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--gate',type=Path);p.add_argument('--visual',choices=['off','on','auto'],default='off')
    p.add_argument('--profile',type=Path)
    p.add_argument('--scene',choices=['ordinary_diffuse','partial_depth_loss','low_light','glare','floating_objects','uneven_surface'],default='ordinary_diffuse')
    a=p.parse_args()
    if a.profile is not None:
        profile=json.loads(a.profile.read_text())
        if a.model is None:a.model=Path(profile['pixel_checkpoint'])
        if a.gate is None:a.gate=Path(profile['candidate_checkpoint'])
    torch.set_num_threads(4);bundle=json.loads(a.bundle.read_text());dry=np.load(a.dry)
    base=FormalSceneMeter.from_empty_geometry(bundle,dry,max_return_depth_m=10.5)
    if a.visual!='off' and (a.model is None or a.rgb is None):raise ValueError('RGB_and_pixel_model_required_unless_visual_off')
    predictor=None if a.visual=='off' or a.model is None else RGBDEvidencePredictor(a.model,'cuda' if torch.cuda.is_available() else 'cpu')
    meter=HybridSceneMeterV2(base=base,dry_depth_m=dry,predictor=predictor,gate_checkpoint=None if a.visual=='off' else a.gate)
    rgb=None if a.rgb is None else cv2.imread(str(a.rgb))
    result=meter.measure(a.scene,rgb_bgr=rgb,depth_m=np.load(a.depth),visual_mode=a.visual)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
