"""Reusable RGB-D video inference, separate from operator interfaces."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from .models.universal import UniversalLiquidSurfaceNet
from .range_calibration import RangeNoiseCalibration
from .rgb_witness import RGBContourWitness
from .surface_candidates import SurfaceCandidateEstimator
from .surface_memory import MetricSurfaceMemory
from .surface_refinement import RefinedSurfaceEstimator
from .verified_tracking import VerifiedSurfaceTracker


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

    def __init__(
        self,
        checkpoint,
        *,
        device=None,
        memory_options=None,
        rgb_witness=None,
        range_profile=None,
        sensor_family=None,
        strict_rgb=False,
    ):
        if strict_rgb and rgb_witness is None:
            raise ValueError("Strict verification requires an independent RGB witness")
        self.predictor = SequencePredictor(checkpoint, device=device)
        self.memory_options = dict(memory_options or {})
        if range_profile is not None:
            payload = json.loads(Path(range_profile).read_text())
            if payload["checkpoint_sha256"] != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
                raise ValueError("Range confidence profile does not match model weights")
            if "range_calibration" in self.memory_options:
                raise ValueError("Specify the range calibration only once")
            self.memory_options["range_calibration"] = RangeNoiseCalibration(payload, sensor_family)
        self.memory = MetricSurfaceMemory(**self.memory_options)
        self.rgb_witness = rgb_witness
        self.strict_rgb = strict_rgb
        self._surface_candidate_engines = {}
        self.verified_tracker = VerifiedSurfaceTracker(
            memory_options=self.memory_options, strict_rgb=self.strict_rgb
        )

    def reset_reference(self):
        """Require fresh independent confirmation after operator revalidation."""
        self.memory.reset()
        self._surface_candidate_engines = {}
        self.verified_tracker = VerifiedSurfaceTracker(
            memory_options=self.memory_options, strict_rgb=self.strict_rgb
        )

    def set_rgb_witness(self, witness: RGBContourWitness):
        if self.strict_rgb and witness is None:
            raise ValueError("Cannot remove independent witness in strict mode")
        self.rgb_witness = witness
        self.reset_reference()

    def process_surface_candidates(
        self,
        rgb_bgr,
        raw_depth_m,
        intrinsics,
        camera_to_world_cv,
        bottom_world_m,
        *,
        area_xy,
        radii,
        mode="early",
        surface_mode="quasistatic",
        pose_valid=True,
    ):
        """Research only: known footprint in the fixed world frame, not trusted output."""
        started = perf_counter()
        prediction = self.predictor.predict(rgb_bgr, raw_depth_m)
        engines = getattr(self, "_surface_candidate_engines", {})
        self._surface_candidate_engines = engines
        key = (mode, surface_mode)
        if key not in engines:
            engines[key] = SurfaceCandidateEstimator(
                mode=mode,
                surface_mode=surface_mode,
                range_calibration=self.memory_options.get("range_calibration"),
            )
        result = engines[key].estimate(
            rgb_bgr,
            raw_depth_m,
            prediction,
            intrinsics,
            camera_to_world_cv,
            bottom_world_m,
            area_xy,
            radii,
            pose_valid=pose_valid,
        )
        result["route"] = "experimental_unverified_surface_candidate"
        result["total_ms"] = (perf_counter() - started) * 1000
        if result["total_ms"] > 500:
            result["quality_flags"].append("latency_deadline_exceeded")
        return result

    def process_refined_surface(
        self,
        rgb_bgr,
        raw_depth_m,
        intrinsics,
        camera_to_world_cv,
        bottom_world_m,
        *,
        area_xy,
        radii,
        mode="balanced",
        stereo_noise=None,
        max_surface_slope=None,
        pose_valid=True,
    ):
        """Opt-in v6 diagnostics; device noise/slope priors must be supplied explicitly."""
        started = perf_counter()
        prediction = self.predictor.predict(rgb_bgr, raw_depth_m)
        selected = raw_depth_m[prediction["mask"] & np.isfinite(raw_depth_m) & (raw_depth_m > 0)]
        calibration = self.memory_options.get("range_calibration")
        sigma = calibration.sigma(float(np.median(selected))) if calibration and selected.size else 0.003
        engine = RefinedSurfaceEstimator(
            mode=mode,
            stereo_noise=stereo_noise,
            sigma_m=max(0.003, sigma),
            max_surface_slope=max_surface_slope,
        )
        out = engine.estimate(
            rgb_bgr,
            raw_depth_m,
            prediction,
            intrinsics,
            camera_to_world_cv,
            bottom_world_m,
            area_xy,
            radii,
            pose_valid=pose_valid,
        )
        out["route"] = "experimental_unverified_surface_refinement_v6"
        out["total_ms"] = (perf_counter() - started) * 1000
        if out["total_ms"] > 500:
            out["quality_flags"].append("latency_deadline_exceeded")
        return out

    def process(
        self,
        rgb_bgr,
        raw_depth_m,
        intrinsics,
        camera_to_world_cv,
        bottom_world_m,
        *,
        pose_valid=True,
        witness_frame=None,
    ):
        started = perf_counter()
        prediction = self.predictor.predict(rgb_bgr, raw_depth_m)
        model_ms = (perf_counter() - started) * 1000
        if self.rgb_witness is not None:
            if witness_frame is None:
                cue_rgb, cue_k, cue_pose = rgb_bgr, intrinsics, camera_to_world_cv
            else:
                required = {"rgb_bgr", "intrinsics", "camera_to_world_cv", "synchronized"}
                if not required.issubset(witness_frame) or witness_frame["synchronized"] is not True:
                    raise ValueError(
                        "Independent RGB frame requires calibrated geometry and explicit synchronization"
                    )
                cue_rgb = witness_frame["rgb_bgr"]
                cue_k = witness_frame["intrinsics"]
                cue_pose = witness_frame["camera_to_world_cv"]
            cue = self.rgb_witness.estimate(
                cue_rgb,
                cue_k,
                cue_pose,
                resolution_checks=self.strict_rgb,
                source_pixel_scale=(witness_frame or {}).get("source_pixel_scale", 1),
            )
            result = self.verified_tracker.process(
                rgb_bgr,
                raw_depth_m,
                prediction,
                intrinsics,
                camera_to_world_cv,
                bottom_world_m,
                pose_valid=pose_valid,
                witness=cue,
            )
            result["route"] = "experimental_rgb_verified_surface"
            result["strict_rgb_resolution_control"] = self.strict_rgb
        else:
            result = self.memory.estimate(
                rgb_bgr,
                raw_depth_m,
                prediction,
                intrinsics,
                camera_to_world_cv,
                bottom_world_m,
                pose_valid=pose_valid,
            )
            result["route"] = "experimental_metric_surface_memory"
        result["model_ms"] = model_ms
        result["total_ms"] = (perf_counter() - started) * 1000
        if result["total_ms"] > 500:
            result["accepted"] = False
            result["level_m"] = None
            result["reasons"].append("latency_deadline_exceeded")
        return result
