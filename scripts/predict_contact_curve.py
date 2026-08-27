#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise FileNotFoundError(path)
    if value.ndim == 3:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError("Depth input must be a single-channel array")
    return value


def _parse_crop(value: str | None, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    if value is None:
        return 0, 0, width, height
    crop = tuple(int(item) for item in value.split(","))
    if len(crop) != 4:
        raise ValueError("--crop-xyxy requires x0,y0,x1,y1")
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Crop {crop} is outside image size {(width, height)}")
    return crop


def _pose(payload, instance_index: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(payload, list):
        payload = payload[instance_index]
    elif "instances" in payload:
        payload = payload["instances"][instance_index]
    rotation = np.asarray(
        payload.get("cam_R_m2c", payload.get("rotation_m2c")),
        dtype=np.float32,
    ).reshape(9)
    translation = np.asarray(
        payload.get("cam_t_m2c", payload.get("translation_m2c")),
        dtype=np.float32,
    ).reshape(3)
    if float(np.linalg.norm(translation)) > 10.0:
        translation *= 0.001
    return rotation, translation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a confidence-scored liquid contact curve from one RGB-D crop"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--depth-scale-to-m", type=float, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--object-index", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--crop-xyxy",
        help="Container crop in original pixels; defaults to the full frame",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.depth_scale_to_m <= 0:
        raise ValueError("--depth-scale-to-m must be positive")

    import torch

    from liquid_depth.training.dtld_contact import build_dtld_contact_model

    rgb = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(args.rgb)
    depth = _read_depth(args.depth)
    if depth.shape != rgb.shape[:2]:
        raise ValueError(f"RGB/depth shapes differ: {rgb.shape[:2]} vs {depth.shape}")
    crop = _parse_crop(args.crop_xyxy, rgb.shape[:2])
    x0, y0, x1, y1 = crop

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    width, height = tuple(int(item) for item in state.get("image_size", (320, 180)))
    max_depth_m = float(state.get("max_depth_m", 3.0))
    rgb_crop = cv2.resize(
        rgb[y0:y1, x0:x1],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    depth_crop = cv2.resize(
        depth[y0:y1, x0:x1],
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    depth_m = depth_crop * args.depth_scale_to_m
    depth_m = np.where(np.isfinite(depth_m) & (depth_m > 0), depth_m, 0.0)
    valid = ((depth_m > 0) & (depth_m <= max_depth_m)).astype(np.float32)

    rgb_unit = cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_normalized = (rgb_unit - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    inputs = np.concatenate(
        (
            rgb_normalized,
            np.clip(depth_m / max_depth_m, 0.0, 1.0)[..., None],
            valid[..., None],
        ),
        axis=2,
    )

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    model = build_dtld_contact_model(
        state.get("backbone", "unet"),
        int(state.get("base_channels", 24)),
        pretrained_backbone=False,
        geometry_conditioning=bool(state.get("geometry_conditioning", False)),
        object_experts=bool(state.get("object_experts", False)),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(requested_device).eval()
    rotation, translation = _pose(_read_json(args.pose_json), args.instance_index)
    pose_features = np.concatenate((rotation, translation / 2.0)).astype(np.float32)

    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(inputs.transpose(2, 0, 1))[None].to(requested_device),
            torch.tensor([args.object_index], device=requested_device),
            torch.from_numpy(pose_features)[None].to(requested_device),
        )
    normalized_curve = prediction["contact_curve"][0].detach().cpu().numpy()
    point_confidence = prediction["contact_curve_point_confidence"][0].detach().cpu().numpy()
    curve_pixels = np.column_stack(
        (
            x0 + normalized_curve[:, 0] * (x1 - x0),
            y0 + normalized_curve[:, 1] * (y1 - y0),
        )
    )
    payload = {
        "contact_curve_pixels": curve_pixels.tolist(),
        "point_confidences": point_confidence.tolist(),
        "curve_confidence_uncalibrated": float(prediction["curve_confidence"][0].detach().cpu()),
        "crop_xyxy": list(crop),
        "input_size": [width, height],
        "checkpoint": str(args.checkpoint.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
