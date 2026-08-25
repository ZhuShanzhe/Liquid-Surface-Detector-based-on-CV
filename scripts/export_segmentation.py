#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained DeepLabV3 checkpoint to the runtime contract")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    width, height = map(int, checkpoint.get("image_size", (512, 512)))
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2)
    model.load_state_dict(checkpoint["model"])
    model.eval().to(args.device)

    class ForegroundLogit(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            return self.network(image)["out"][:, 1:2]

    wrapper = ForegroundLogit(model).eval()
    example = torch.zeros(1, 3, height, width, device=args.device)
    traced = torch.jit.trace(wrapper, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(args.output))
    print(f"Exported {args.output} with input shape [1,3,{height},{width}]")


if __name__ == "__main__":
    main()
