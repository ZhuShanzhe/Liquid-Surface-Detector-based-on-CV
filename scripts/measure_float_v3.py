"""Replay/install entry: optional RGB-D evidence, original metric depth samples."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from liquid_depth.float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3
from liquid_depth.supported_scene_meter import FormalSceneMeter


def main():
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--dry', type=Path, required=True); p.add_argument('--depth', type=Path, required=True)
    p.add_argument('--rgb', type=Path); p.add_argument('--profile', type=Path)
    p.add_argument('--visual', choices=['off', 'auto'], default='off'); args = p.parse_args()
    torch.set_num_threads(4); dry = np.load(args.dry)
    base = FormalSceneMeter.from_empty_geometry(json.loads(args.bundle.read_text()), dry, max_return_depth_m=10.5)
    predictor = None; verifier_path = None; rgb = None
    if args.visual == 'auto':
        if args.profile is None or args.rgb is None: raise ValueError('RGB_and_profile_required_for_optional_visual')
        profile = json.loads(args.profile.read_text()); rgb = cv2.imread(str(args.rgb))
        if rgb is None: raise ValueError('RGB_image_unreadable')
        predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'], 'cuda' if torch.cuda.is_available() else 'cpu')
        verifier_path = profile['verifier_checkpoint']
    meter = OptionalFloatMeterV3(base, dry, predictor, verifier_path)
    print(json.dumps(meter.measure(depth_m=np.load(args.depth), rgb_bgr=rgb, visual_mode=args.visual), allow_nan=False))


if __name__ == '__main__': main()
