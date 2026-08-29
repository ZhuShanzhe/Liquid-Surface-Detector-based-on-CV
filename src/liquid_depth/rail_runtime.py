from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .calibration import CameraCalibration
from .contact_model import load_contact_specialists
from .io import load_frame, write_json
from .rail_calibration import intersect_curve_with_rail, rail_depth_from_y
from .scenario_policy import ComplexScenePolicy, load_scene_context, measure_scene_signals
from .system_runtime import (
    _camera_depth_correction,
    _depth_input,
    _fixed_crop,
    load_system_profile,
    resolve_profile_path,
)
from .temporal import RobustKalmanFilter


def estimate_reference_motion_px(
    reference_bgr: np.ndarray,
    current_bgr: np.ndarray,
    crop: tuple[int, int, int, int],
) -> float | None:
    """Estimate container-image motion from robust local features."""
    x0, y0, x1, y1 = crop
    reference = cv2.cvtColor(reference_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    current = cv2.cvtColor(current_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    key_ref, desc_ref = detector.detectAndCompute(reference, None)
    key_cur, desc_cur = detector.detectAndCompute(current, None)
    if desc_ref is None or desc_cur is None or len(key_ref) < 12 or len(key_cur) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_ref, desc_cur, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(good) < 10:
        return None
    source = np.float32([key_ref[item.queryIdx].pt for item in good])
    target = np.float32([key_cur[item.trainIdx].pt for item in good])
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 2.5)
    if homography is None or mask is None or int(mask.sum()) < 8:
        return None
    width, height = x1 - x0, y1 - y0
    corners = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
    moved = cv2.perspectiveTransform(corners, homography)[0]
    return float(np.median(np.linalg.norm(moved - corners[0], axis=1)))


class RailLiquidDepthSystem:
    """Fixed-camera product runtime using a five-point image measurement rail."""

    def __init__(
        self,
        profile_path: str | Path,
        *,
        device: str | None = None,
        temporal: bool = False,
    ) -> None:
        self.profile = load_system_profile(profile_path)
        if self.profile.get("measurement", {}).get("mode") != "fixed_rail":
            raise ValueError("Profile is not a fixed-rail measurement profile")
        self.calibration = CameraCalibration.from_dict(self.profile["camera"])
        self.depth_scale_to_m = float(self.profile["camera"].get("depth_scale_to_m", 0.001))
        if self.depth_scale_to_m <= 0:
            raise ValueError("camera.depth_scale_to_m must be positive")
        self.depth_correction_scale, self.depth_correction_offset_m = _camera_depth_correction(
            self.profile["camera"]
        )
        measurement = self.profile["measurement"]
        self.reference_bgr = None
        if measurement.get("reference_image_path"):
            reference_path = resolve_profile_path(self.profile, measurement["reference_image_path"])
            self.reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
            if self.reference_bgr is None:
                raise FileNotFoundError(f"Reference image not readable: {reference_path}")
        self._load_model(device)
        specialist_device = device or self.profile["perception"].get("device", "cuda")
        self.complex_models = load_contact_specialists(
            self.profile, resolve_path=resolve_profile_path, device=specialist_device
        )
        self.scene_policy = ComplexScenePolicy(
            self.profile.get("complex_scene", {}), available_variants=self.complex_models
        )
        self.temporal_filter = None
        if temporal:
            options = self.profile.get("temporal", {})
            self.temporal_filter = RobustKalmanFilter(
                process_variance=float(options.get("process_variance_m2", 1e-6)),
                measurement_variance=float(options.get("measurement_variance_m2", 2.5e-5)),
                gate_sigma=float(options.get("gate_sigma", 3.5)),
                max_jump=float(options.get("max_jump_m", 0.02)),
                min_confidence=float(options.get("min_confidence", 0.2)),
            )

    def _load_model(self, device: str | None) -> None:
        import torch

        from .training.dtld_contact import build_dtld_contact_model

        checkpoint = resolve_profile_path(
            self.profile,
            self.profile["perception"]["checkpoint_path"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.input_size = tuple(int(item) for item in state.get("image_size", (320, 180)))
        self.max_depth_m = float(state.get("max_depth_m", 3.0))
        requested = torch.device(device or self.profile["perception"].get("device", "cuda"))
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.torch = torch
        self.device = requested
        self.model = build_dtld_contact_model(
            state.get("backbone", "unet"),
            int(state.get("base_channels", 24)),
            pretrained_backbone=False,
            geometry_conditioning=bool(state.get("geometry_conditioning", False)),
            object_experts=bool(state.get("object_experts", False)),
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(requested).eval()

    def _predict(
        self,
        rgb_bgr: np.ndarray,
        depth_raw: np.ndarray,
        crop: tuple[int, int, int, int],
        bundle=None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if bundle is None:
            model = self.model
            torch = self.torch
            device = self.device
            input_size = self.input_size
            max_depth_m = self.max_depth_m
        else:
            model = bundle.model
            torch = bundle.torch
            device = bundle.device
            input_size = bundle.input_size
            max_depth_m = bundle.max_depth_m

        inputs = _depth_input(
            rgb_bgr,
            depth_raw,
            crop,
            input_size,
            self.depth_scale_to_m,
            max_depth_m,
            self.depth_correction_scale,
            self.depth_correction_offset_m,
        )
        pose_features = np.zeros(12, dtype=np.float32)
        with torch.inference_mode():
            prediction = model(
                torch.from_numpy(inputs.transpose(2, 0, 1))[None].to(device),
                torch.tensor(
                    [int(self.profile["perception"]["object_index"])],
                    device=device,
                ),
                torch.from_numpy(pose_features)[None].to(device),
            )
        normalized = prediction["contact_curve"][0].detach().cpu().numpy()
        point_confidence = prediction["contact_curve_point_confidence"][0].detach().cpu().numpy()
        x0, y0, x1, y1 = crop
        curve = np.column_stack(
            (
                x0 + normalized[:, 0] * (x1 - x0),
                y0 + normalized[:, 1] * (y1 - y0),
            )
        )
        global_confidence = float(prediction["curve_confidence"][0].detach().cpu())
        return curve, point_confidence, global_confidence

    @staticmethod
    def _overlay(
        rgb_bgr: np.ndarray,
        crop: tuple[int, int, int, int],
        curve: np.ndarray,
        rail_x: float,
        intersection_y: float | None,
        accepted: bool,
    ) -> np.ndarray:
        image = rgb_bgr.copy()
        x0, y0, x1, y1 = crop
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 180, 0), 2)
        curve_int = np.rint(curve).astype(np.int32)
        cv2.polylines(image, [curve_int], False, (0, 220, 0), 2)
        rail = round(rail_x)
        cv2.line(image, (rail, y0), (rail, y1), (255, 0, 255), 2)
        if intersection_y is not None:
            cv2.circle(
                image,
                (rail, round(intersection_y)),
                7,
                (0, 255, 255),
                -1,
            )
        cv2.putText(
            image,
            "ACCEPTED" if accepted else "REJECTED",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 0) if accepted else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return image

    def measure(
        self,
        frame_dir: str | Path,
        output_dir: str | Path | None = None,
        *,
        apply_output_calibration: bool = True,
    ) -> dict[str, Any]:
        frame = load_frame(frame_dir)
        if frame.rgb_bgr.shape[1::-1] != self.calibration.image_size:
            raise ValueError("Frame resolution differs from rail calibration")
        crop = _fixed_crop(
            self.profile["perception"]["crop_xyxy"],
            frame.rgb_bgr.shape[:2],
        )
        reasons: list[str] = []
        started = perf_counter()
        signals = measure_scene_signals(
            frame.rgb_bgr,
            frame.depth,
            roi=crop,
            depth_scale_to_m=self.depth_scale_to_m,
            max_depth_m=max(self.max_depth_m, 10.0),
        )
        scene_context = dict(self.profile.get("complex_scene", {}).get("scene_context", {}))
        scene_context.update(load_scene_context(frame.source_dir))
        scene_decision = self.scene_policy.decide(signals, scene_context)
        specialist = self.complex_models.get(scene_decision.model_variant)
        reference_motion_px = None
        if self.reference_bgr is not None:
            reference_motion_px = estimate_reference_motion_px(self.reference_bgr, frame.rgb_bgr, crop)
            limit = float(self.profile["measurement"].get("max_reference_motion_px", 4.0))
            if reference_motion_px is None:
                reasons.append("camera_motion_unobservable")
            elif reference_motion_px > limit:
                reasons.append("camera_moved_recalibration_required")
        prediction_started = perf_counter()
        curve, point_confidence, global_confidence = self._predict(
            frame.rgb_bgr,
            frame.depth,
            crop,
            bundle=specialist if scene_decision.activated else None,
        )
        prediction_ms = (perf_counter() - prediction_started) * 1000.0
        rail = self.profile["measurement"]["rail_calibration"]
        rail_x = float(rail["rail_x_px"])
        intersection_y = None
        raw_depth = None
        local_confidence = 0.0
        try:
            intersection_y, local_confidence = intersect_curve_with_rail(
                curve,
                point_confidence,
                rail_x,
            )
            raw_depth = rail_depth_from_y(
                rail,
                intersection_y,
                extrapolation_margin_px=float(
                    self.profile["measurement"].get("extrapolation_margin_px", 2.0)
                ),
            )
        except ValueError as exc:
            reasons.append(str(exc))
        min_confidence = float(self.profile["measurement"].get("min_intersection_confidence", 0.5))
        if local_confidence < min_confidence:
            reasons.append("low_rail_intersection_confidence")
        calibration = self.profile.get("output_calibration", {})
        scale = float(calibration.get("scale", 1.0)) if apply_output_calibration else 1.0
        offset = float(calibration.get("offset_m", 0.0)) if apply_output_calibration else 0.0
        candidate = None if raw_depth is None else scale * raw_depth + offset
        depth_range = rail["calibration_depth_range_m"]
        if candidate is not None and not (
            float(depth_range[0]) - 1e-6 <= candidate <= float(depth_range[1]) + 1e-6
        ):
            reasons.append("depth_outside_calibrated_range")
        if not scene_decision.result_allowed:
            reasons.append("complex_model_required_but_unavailable")

        accepted = candidate is not None and not reasons
        confidence = local_confidence
        temporal_payload = None
        filtered = None
        if self.temporal_filter is not None:
            temporal = self.temporal_filter.update(
                float(candidate) if candidate is not None else float("nan"),
                confidence,
                accepted,
            )
            temporal_payload = {
                "accepted": temporal.accepted,
                "value_m": temporal.value,
                "variance_m2": temporal.variance,
                "confidence": temporal.confidence,
                "innovation_m": temporal.innovation,
                "rejection_reason": temporal.reason,
            }
            if accepted and not temporal.accepted:
                reasons.append(temporal.reason or "temporal_rejection")
            accepted = accepted and temporal.accepted
            confidence = min(confidence, temporal.confidence)
            filtered = temporal.value
        total_ms = (perf_counter() - started) * 1000.0
        latency_within_budget = total_ms <= scene_decision.latency_budget_ms
        if not latency_within_budget and bool(
            self.profile.get("complex_scene", {}).get("enforce_latency_budget", True)
        ):
            accepted = False
            reasons.append("latency_budget_exceeded")
        reasons = list(dict.fromkeys(reasons))

        reported = filtered if accepted and filtered is not None else candidate
        result = {
            "schema_version": 1,
            "measurement_mode": "fixed_rail",
            "reference_motion_px": reference_motion_px,
            "frame_id": frame.frame_id,
            "accepted": accepted,
            "liquid_depth_m": reported if accepted else None,
            "liquid_depth_candidate_m": candidate,
            "uncertainty_m": float(rail["loocv_mae_m"]),
            "confidence_uncalibrated": confidence,
            "complex_scene": scene_decision.to_dict(),
            "latency": {
                "inference_total_ms": total_ms,
                "model_prediction_ms": prediction_ms,
                "budget_ms": scene_decision.latency_budget_ms,
                "within_budget": latency_within_budget,
                "budget_scope": "warm_measurement_frame_load_excluded_artifact_serialization_excluded",
            },
            "rejection_reasons": list(dict.fromkeys(reasons)),
            "crop_xyxy": list(crop),
            "rail_x_px": rail_x,
            "rail_intersection_y_px": intersection_y,
            "rail_intersection_confidence": local_confidence,
            "curve_confidence_uncalibrated": global_confidence,
            "output_calibration": {"scale": scale, "offset_m": offset},
            "temporal": temporal_payload,
        }
        if output_dir is not None:
            target = Path(output_dir)
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "depth_result.json", result)
            write_json(
                target / "contact_curve.json",
                {
                    "contact_curve_pixels": curve.tolist(),
                    "point_confidences": point_confidence.tolist(),
                },
            )
            overlay = self._overlay(
                frame.rgb_bgr,
                crop,
                curve,
                rail_x,
                intersection_y,
                accepted,
            )
            cv2.imwrite(str(target / "measurement_overlay.png"), overlay)
        return result


def make_product_system(
    profile_path: str | Path,
    *,
    device: str | None = None,
    temporal: bool = False,
):
    profile = load_system_profile(profile_path)
    if profile.get("measurement", {}).get("mode") == "fixed_rail":
        return RailLiquidDepthSystem(
            profile_path,
            device=device,
            temporal=temporal,
        )
    from .system_runtime import LiquidDepthSystem

    return LiquidDepthSystem(profile_path, device=device, temporal=temporal)
