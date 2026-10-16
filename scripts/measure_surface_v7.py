"""No-USB application entry with explicit floating peak vision switch."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.floating_surface_application_v7 import from_empty_geometry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True, choices=['ordinary_diffuse', 'partial_depth_loss', 'low_light', 'glare', 'floating_objects', 'uneven_surface'])
    p.add_argument('--bundle', type=Path, required=True); p.add_argument('--dry', type=Path, required=True)
    p.add_argument('--depth', type=Path, required=True); p.add_argument('--rgb', type=Path)
    p.add_argument('--profile', type=Path, default=Path(__file__).resolve().parents[1]/'configs/floating_surface_v7.json')
    p.add_argument('--enable-floating-peak-vision', action='store_true'); p.add_argument('--guards-only', action='store_true'); a = p.parse_args()
    rgb = None
    if a.scene == 'floating_objects' and a.enable_floating_peak_vision:
        if a.rgb is None: raise ValueError('aligned_RGB_required')
        rgb = cv2.imread(str(a.rgb))
        if rgb is None: raise ValueError('RGB_unreadable')
        import torch
        torch.set_num_threads(4)
    meter = from_empty_geometry(json.loads(a.bundle.read_text()), np.load(a.dry), profile_path=a.profile, guards_only=a.guards_only, max_return_depth_m=10.5)
    print(json.dumps(meter.measure(a.scene, depth_m=np.load(a.depth), rgb_bgr=rgb, visual_enabled=a.enable_floating_peak_vision), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
