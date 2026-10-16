"""No-USB experimental single-frame or explicit timestamped sequence entry."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.floating_joint_application_v9 import from_empty_geometry


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', default='floating_objects')
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--dry', type=Path, required=True)
    p.add_argument('--depth', type=Path)
    p.add_argument('--rgb', type=Path)
    p.add_argument('--sequence', type=Path)
    p.add_argument('--stream-id')
    p.add_argument('--frame-id', type=int)
    p.add_argument('--timestamp-s', type=float)
    p.add_argument('--profile', type=Path, default=Path(__file__).resolve().parents[1]/'configs/floating_joint_v9.json')
    p.add_argument('--enable-floating-peak-vision', action='store_true')
    p.add_argument('--stateless', action='store_true')
    args = p.parse_args()
    if args.sequence is None and args.depth is None:
        p.error('--depth or --sequence is required')
    if args.sequence is not None and args.depth is not None:
        p.error('choose --depth or --sequence, not both')
    if args.enable_floating_peak_vision and args.scene == 'floating_objects':
        import torch
        torch.set_num_threads(4)
    meter = from_empty_geometry(json.loads(args.bundle.read_text()), np.load(args.dry), profile_path=args.profile,
        temporal=not args.stateless, max_return_depth_m=10.5)
    frames = json.loads(args.sequence.read_text()) if args.sequence else [{'depth': str(args.depth), 'rgb': str(args.rgb) if args.rgb else None,
        'frame_id': args.frame_id, 'timestamp_s': args.timestamp_s}]
    for frame in frames:
        rgb = None
        if args.enable_floating_peak_vision and args.scene == 'floating_objects':
            if not frame.get('rgb'):
                raise ValueError('aligned_RGB_required')
            rgb = cv2.imread(frame['rgb'])
            if rgb is None:
                raise ValueError('RGB_unreadable')
        result = meter.measure(args.scene, depth_m=np.load(frame['depth']), rgb_bgr=rgb, visual_enabled=args.enable_floating_peak_vision,
            stream_id=args.stream_id, frame_id=frame.get('frame_id'), timestamp_s=frame.get('timestamp_s'))
        print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
