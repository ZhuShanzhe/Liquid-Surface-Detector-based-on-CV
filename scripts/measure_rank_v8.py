"""No-USB entry for explicit experimental floating-object peak estimation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.floating_rank_application_v8 import from_empty_geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, choices=['ordinary_diffuse', 'partial_depth_loss', 'low_light', 'glare', 'floating_objects', 'uneven_surface'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dry', type=Path, required=True)
    parser.add_argument('--depth', type=Path, required=True)
    parser.add_argument('--rgb', type=Path)
    parser.add_argument('--profile', type=Path, default=Path(__file__).resolve().parents[1]/'configs/floating_rank_v8.json')
    parser.add_argument('--enable-floating-peak-vision', action='store_true')
    parser.add_argument('--policy-variant', choices=['selected', 'no_rank', 'research'], default='selected')
    args = parser.parse_args()
    rgb = None
    if args.scene == 'floating_objects' and args.enable_floating_peak_vision:
        if args.rgb is None:
            raise ValueError('aligned_RGB_required')
        rgb = cv2.imread(str(args.rgb))
        if rgb is None:
            raise ValueError('RGB_unreadable')
        import torch
        torch.set_num_threads(4)
    meter = from_empty_geometry(json.loads(args.bundle.read_text()), np.load(args.dry), profile_path=args.profile,
        policy_variant=args.policy_variant, max_return_depth_m=10.5)
    print(json.dumps(meter.measure(args.scene, depth_m=np.load(args.depth), rgb_bgr=rgb, visual_enabled=args.enable_floating_peak_vision), allow_nan=False))


if __name__ == '__main__':
    main()
