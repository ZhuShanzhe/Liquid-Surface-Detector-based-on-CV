#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an ordinary-scene checkpoint with robust RGB-D sensor anchoring"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--bias-limit-m", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 < args.mask_threshold < 1.0:
        parser.error("--mask-threshold must be in (0, 1)")
    if args.bias_limit_m < 0.0:
        parser.error("--bias-limit-m must be non-negative")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint.update(
        model_family="universal_liquid_surface_v8_ordinary_sensor_anchor",
        robust_depth_anchor_enabled=True,
        robust_anchor_mask_threshold=float(args.mask_threshold),
        robust_anchor_bias_limit_m=float(args.bias_limit_m),
        derived_from_checkpoint=str(args.source.resolve()),
        route_scope="ordinary_scene_with_reliable_rgbd_returns",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Created {args.output} from {args.source}")


if __name__ == "__main__":
    main()
