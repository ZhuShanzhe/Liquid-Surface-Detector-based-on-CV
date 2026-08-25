from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import resolve_config_path
from .geometry import depth_to_meters


@dataclass(frozen=True)
class RefinedDepth:
    depth_m: np.ndarray
    confidence: np.ndarray
    backend: str


class DepthRefiner(Protocol):
    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        """Return metric depth and per-pixel confidence in [0, 1]."""


class IdentityDepthRefiner:
    """Pass raw sensor depth through while exposing its validity mask as confidence."""

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        del rgb_bgr
        depth_m = depth_to_meters(raw_depth)
        valid = np.isfinite(depth_m) & (depth_m > 0)
        confidence = valid.astype(np.float32)
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32)
        return RefinedDepth(depth_m, confidence, "identity")


class TorchScriptDepthRefiner:
    """Run a portable RGB-D refinement model with a stable five-channel contract.

    Input is ``[RGB ImageNet-normalized, raw_depth/max_depth_m, validity]``.
    Output must be either one normalized depth channel or two channels containing
    normalized depth and a confidence logit. The adapter restores metric meters.
    """

    def __init__(self, model_path: Path, input_size: list[int], max_depth_m: float):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the 'train' extra to use TorchScript depth refinement") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.input_size = tuple(map(int, input_size))
        self.max_depth_m = float(max_depth_m)

    def predict(self, rgb_bgr: np.ndarray, raw_depth: np.ndarray) -> RefinedDepth:
        torch = self.torch
        height, width = rgb_bgr.shape[:2]
        depth_m = depth_to_meters(raw_depth).astype(np.float32)
        valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= self.max_depth_m)
        depth_m = np.where(valid, depth_m, 0.0)

        target_width, target_height = self.input_size
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        normalized_depth = cv2.resize(
            depth_m / self.max_depth_m, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
        validity = cv2.resize(
            valid.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
        model_input = np.concatenate((rgb, normalized_depth[..., None], validity[..., None]), axis=2)
        tensor = torch.from_numpy(model_input.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, dict):
            output = output.get("out", next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output[0].float().cpu().numpy()
        if output.ndim == 2:
            output = output[None]
        refined = np.clip(output[0], 0.0, 1.0) * self.max_depth_m
        confidence = 1.0 / (1.0 + np.exp(-output[1])) if output.shape[0] > 1 else np.ones_like(refined)
        refined = cv2.resize(refined, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return RefinedDepth(refined, np.clip(confidence, 0.0, 1.0), "torchscript")


def make_depth_refiner(config: dict) -> DepthRefiner:
    options = config.get("depth_refinement", {"backend": "identity"})
    backend = str(options.get("backend", "identity")).lower()
    if backend in {"none", "identity"}:
        return IdentityDepthRefiner()
    if backend == "torchscript":
        model = options["torchscript"]
        path = resolve_config_path(config, model["model_path"])
        return TorchScriptDepthRefiner(path, model["input_size"], model["max_depth_m"])
    raise ValueError(f"Unknown depth refinement backend: {backend}")
