"""Reusable RGB-D video inference, separate from operator interfaces."""

from __future__ import annotations

import inspect
from time import perf_counter

import cv2
import numpy as np
import torch

from .models.universal import UniversalLiquidSurfaceNet
from .surface_memory import MetricSurfaceMemory


class SequencePredictor:
    def __init__(self, checkpoint, device=None):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        options = {
            key: state[key] for key in inspect.signature(UniversalLiquidSurfaceNet).parameters if key in state
        }
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = UniversalLiquidSurfaceNet(**options).eval().to(self.device)
        self.model.load_state_dict(state["model"], strict=True)
        self.size = tuple(state["image_size"])
        self.minimum = float(state.get("min_depth_m", 0.1))
        self.maximum = float(state.get("max_depth_m", 10.0))
        self.predict(np.zeros((180, 320, 3), np.uint8), np.ones((180, 320), np.float32))

    def predict(self, rgb, depth):
        h, w = depth.shape
        color = cv2.cvtColor(cv2.resize(rgb, self.size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        color = (color - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        z = cv2.resize(depth, self.size, interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(z) & (z > 0)
        encoded = np.where(
            valid,
            np.log(np.clip(z, self.minimum, self.maximum) / self.minimum)
            / np.log(self.maximum / self.minimum),
            0,
        )
        x = np.concatenate((color, encoded[..., None], valid[..., None]), axis=2).astype(np.float32)
        with torch.inference_mode():
            out = self.model(torch.from_numpy(x.transpose(2, 0, 1)[None]).to(self.device))
        result = {}
        for key, dest in [("mask_logits", "mask"), ("confidence", "confidence"), ("depth_m", "depth_m")]:
            value = out[key].sigmoid() if key == "mask_logits" else out[key]
            array = cv2.resize(value[0, 0].cpu().numpy(), (w, h))
            result[dest] = (array > 0.5) if dest == "mask" else array
        return result


class UniversalSurfaceVideoSystem:
    """Opt-in top-view route; requires metric bottom, gravity and camera pose."""

    def __init__(self, checkpoint, *, device=None, memory_options=None):
        self.predictor = SequencePredictor(checkpoint, device=device)
        self.memory = MetricSurfaceMemory(**(memory_options or {}))

    def reset_reference(self):
        """Call only after validating the new operating point/calibration."""
        self.memory.reset()

    def process(
        self, rgb_bgr, raw_depth_m, intrinsics, camera_to_world_cv, bottom_world_m, *, pose_valid=True
    ):
        started = perf_counter()
        prediction = self.predictor.predict(rgb_bgr, raw_depth_m)
        model_ms = (perf_counter() - started) * 1000
        result = self.memory.estimate(
            rgb_bgr,
            raw_depth_m,
            prediction,
            intrinsics,
            camera_to_world_cv,
            bottom_world_m,
            pose_valid=pose_valid,
        )
        result["model_ms"] = model_ms
        result["total_ms"] = (perf_counter() - started) * 1000
        result["route"] = "experimental_metric_surface_memory"
        if result["total_ms"] > 500:
            result["accepted"] = False
            result["level_m"] = None
            result["reasons"].append("latency_deadline_exceeded")
        return result
