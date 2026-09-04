#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the project multi-task model to TorchScript")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import torch

    from liquid_depth.models import LiquidSurfaceMultiTaskNet
    from liquid_depth.models.universal import UniversalLiquidSurfaceNet

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    width, height = map(int, checkpoint["image_size"])
    model_family = str(checkpoint.get("model_family", "multitask"))
    if model_family.startswith("universal_liquid_surface"):
        model = UniversalLiquidSurfaceNet(
            int(checkpoint["base_channels"]),
            float(checkpoint.get("min_depth_m", 0.1)),
            float(checkpoint["max_depth_m"]),
            rgb_prior_enabled=bool(checkpoint.get("rgb_prior_enabled", False)),
            separate_confidence_head=bool(checkpoint.get("separate_confidence_head", False)),
            level_calibration_enabled=bool(checkpoint.get("level_calibration_enabled", False)),
            calibration_scale_limit=float(checkpoint.get("calibration_scale_limit", 0.05)),
            calibration_bias_limit_m=float(checkpoint.get("calibration_bias_limit_m", 0.02)),
        )
    else:
        model = LiquidSurfaceMultiTaskNet(
            int(checkpoint["base_channels"]),
            float(checkpoint["max_depth_m"]),
        )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(args.device)
    example = torch.zeros(1, 5, height, width, device=args.device)
    traced = torch.jit.trace(model, example, strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    print(f"Exported {args.output} ({model_family}) with input shape [1,5,{height},{width}]")


if __name__ == "__main__":
    main()
