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

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    width, height = map(int, checkpoint["image_size"])
    model = LiquidSurfaceMultiTaskNet(int(checkpoint["base_channels"]), float(checkpoint["max_depth_m"]))
    model.load_state_dict(checkpoint["model"])
    model.eval().to(args.device)
    example = torch.zeros(1, 5, height, width, device=args.device)
    traced = torch.jit.trace(model, example, strict=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    print(f"Exported {args.output} with input shape [1,5,{height},{width}]")


if __name__ == "__main__":
    main()
