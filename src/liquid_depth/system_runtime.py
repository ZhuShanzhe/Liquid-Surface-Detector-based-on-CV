from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
import yaml

from .calibration import (
    CameraCalibration,
    PoseEstimate,
    compose_container_pose,
    detect_aruco_marker_pose,
)
from .contact_model import load_contact_specialists
from .container_geometry import load_container_model, project_model_points
from .io import load_frame, write_json
from .scenario_policy import ComplexScenePolicy, load_scene_context, measure_scene_signals
from .sparse_contact import estimate_level_from_sparse_contact
from .temporal import RobustKalmanFilter


def load_system_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    with profile_path.open("r", encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError("Unsupported or invalid system profile")
    profile["_profile_path"] = str(profile_path)
    return profile


def save_system_profile(path: str | Path, profile: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    payload = {key: value for key, value in profile.items() if not key.startswith("_")}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def resolve_profile_path(profile: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(profile["_profile_path"]).parent / path


def validate_system_profile(profile: dict[str, Any]) -> None:
    required = {"camera", "container", "perception", "pose"}
    missing = required - set(profile)
    if missing:
        raise ValueError("System profile missing: " + ", ".join(sorted(missing)))
    CameraCalibration.from_dict(profile["camera"])
    container = profile["container"]
    for key in ("model_path", "level_axis", "level_origin_m"):
        if key not in container:
            raise ValueError(f"container.{key} is required")
    perception = profile["perception"]
    if "checkpoint_path" not in perception or "object_index" not in perception:
        raise ValueError("perception checkpoint_path and object_index are required")
    if profile["pose"].get("mode") not in {"fixed", "marker_tracking"}:
        raise ValueError("pose.mode must be fixed or marker_tracking")


def _projected_crop(
    points_px: np.ndarray,
    image_shape: tuple[int, int],
    margin_ratio: float = 0.18,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    finite = points_px[np.isfinite(points_px).all(axis=1)]
    if len(finite) < 10:
        raise ValueError("container_projection_not_visible")
    low = np.percentile(finite, 1.0, axis=0)
    high = np.percentile(finite, 99.0, axis=0)
    extent = np.maximum(high - low, 2.0)
    low -= extent * margin_ratio
    high += extent * margin_ratio
    x0, y0 = np.floor(low).astype(int)
    x1, y1 = np.ceil(high).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 - x0 < 32 or y1 - y0 < 32:
        raise ValueError("container_projection_too_small")
    return x0, y0, x1, y1


def _fixed_crop(value: Any, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    crop = tuple(int(item) for item in value)
    if len(crop) != 4:
        raise ValueError("crop_xyxy must contain four integers")
    x0, y0, x1, y1 = crop
    height, width = image_shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop_xyxy is outside the image")
    return crop


def _depth_input(
    rgb_bgr: np.ndarray,
    depth_raw: np.ndarray,
    crop: tuple[int, int, int, int],
    image_size: tuple[int, int],
    depth_scale_to_m: float,
    max_depth_m: float,
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    width, height = image_size
    rgb = cv2.resize(
        rgb_bgr[y0:y1, x0:x1],
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    depth = cv2.resize(
        depth_raw[y0:y1, x0:x1],
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    depth *= depth_scale_to_m
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    valid = ((depth > 0) & (depth <= max_depth_m)).astype(np.float32)
    rgb_unit = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_normalized = (rgb_unit - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.concatenate(
        (
            rgb_normalized,
            np.clip(depth / max_depth_m, 0.0, 1.0)[..., None],
            valid[..., None],
        ),
        axis=2,
    )


def render_measurement_overlay(
    rgb_bgr: np.ndarray,
    crop: tuple[int, int, int, int],
    curve_px: np.ndarray,
    confidence: np.ndarray,
    selected_indices: np.ndarray,
    accepted: bool,
) -> np.ndarray:
    image = rgb_bgr.copy()
    x0, y0, x1, y1 = crop
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 180, 0), 2)
    selected = {int(item) for item in selected_indices}
    for index, ((x, y), score) in enumerate(zip(curve_px, confidence, strict=True)):
        color = (0, 220, 0) if index in selected else (80, 80, 220)
        radius = 4 if index in selected else 2
        cv2.circle(image, (round(x), round(y)), radius, color, -1)
        if index in selected:
            cv2.putText(
                image,
                f"{score:.2f}",
                (round(x) + 3, round(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )
    status = "ACCEPTED" if accepted else "REJECTED"
    cv2.putText(
        image,
        status,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 0) if accepted else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image


class LiquidDepthSystem:
    def __init__(
        self,
        profile_path: str | Path,
        *,
        device: str | None = None,
        temporal: bool = False,
    ) -> None:
        self.profile = load_system_profile(profile_path)
        validate_system_profile(self.profile)
        self.calibration = CameraCalibration.from_dict(self.profile["camera"])
        container = self.profile["container"]
        self.container_model = load_container_model(
            resolve_profile_path(self.profile, container["model_path"]),
            container["level_axis"],
            container["level_origin_m"],
        )
        self.depth_scale_to_m = float(self.profile["camera"].get("depth_scale_to_m", 0.001))
        if self.depth_scale_to_m <= 0:
            raise ValueError("camera.depth_scale_to_m must be positive")
        self._load_model(device)
        specialist_device = device or self.profile["perception"].get("device", "cuda")
        self.complex_models = load_contact_specialists(
            self.profile,
            resolve_path=resolve_profile_path,
            device=specialist_device,
        )
        self.scene_policy = ComplexScenePolicy(
            self.profile.get("complex_scene", {}),
            available_variants=self.complex_models,
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

        checkpoint_path = resolve_profile_path(
            self.profile,
            self.profile["perception"]["checkpoint_path"],
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.input_size = tuple(int(item) for item in state.get("image_size", (320, 180)))
        self.max_depth_m = float(state.get("max_depth_m", 3.0))
        requested = torch.device(device or self.profile["perception"].get("device", "cuda"))
        if requested.type == "cuda" and not torch.cuda.is_available():
            requested = torch.device("cpu")
        self.device = requested
        self.torch = torch
        self.model = build_dtld_contact_model(
            state.get("backbone", "unet"),
            int(state.get("base_channels", 24)),
            pretrained_backbone=False,
            geometry_conditioning=bool(state.get("geometry_conditioning", False)),
            object_experts=bool(state.get("object_experts", False)),
        )
        self.model.load_state_dict(state["model"], strict=True)
        self.model.to(self.device).eval()

    def _pose(self, rgb_bgr: np.ndarray) -> PoseEstimate:
        pose = self.profile["pose"]
        if pose["mode"] == "fixed":
            return PoseEstimate(
                np.asarray(pose["rotation_m2c"], dtype=np.float64).reshape(3, 3),
                np.asarray(pose["translation_m2c_m"], dtype=np.float64).reshape(3),
                float(pose.get("calibration_reprojection_rmse_px", 0.0)),
                "fixed_profile",
            )
        marker = pose["marker"]
        marker_pose = detect_aruco_marker_pose(
            rgb_bgr,
            self.calibration,
            marker_id=int(marker["id"]),
            marker_size_m=float(marker["size_m"]),
            dictionary_name=str(marker.get("dictionary", "DICT_4X4_50")),
        )
        return compose_container_pose(
            marker_pose,
            np.asarray(marker["container_to_marker"], dtype=np.float64),
        )

    def _predict_curve(
        self,
        rgb_bgr: np.ndarray,
        depth: np.ndarray,
        crop: tuple[int, int, int, int],
        pose: PoseEstimate,
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
            depth,
            crop,
            input_size,
            self.depth_scale_to_m,
            max_depth_m,
        )
        rotation_features = pose.rotation_m2c.reshape(-1)
        pose_features = np.concatenate((rotation_features, pose.translation_m2c_m / 2.0)).astype(np.float32)
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
        confidence = prediction["contact_curve_point_confidence"][0].detach().cpu().numpy()
        x0, y0, x1, y1 = crop
        pixels = np.column_stack(
            (
                x0 + normalized[:, 0] * (x1 - x0),
                y0 + normalized[:, 1] * (y1 - y0),
            )
        )
        global_confidence = float(prediction["curve_confidence"][0].detach().cpu())
        return pixels, confidence, global_confidence

    def measure(
        self,
        frame_dir: str | Path,
        output_dir: str | Path | None = None,
        *,
        apply_output_calibration: bool = True,
    ) -> dict[str, Any]:
        frame = load_frame(frame_dir)
        if frame.rgb_bgr.shape[1::-1] != self.calibration.image_size:
            raise ValueError(
                f"Frame size {frame.rgb_bgr.shape[1::-1]} differs from calibrated "
                f"size {self.calibration.image_size}"
            )
        started = perf_counter()
        pose = self._pose(frame.rgb_bgr)
        projected, _, _ = project_model_points(
            self.container_model,
            self.calibration.camera_matrix,
            pose.rotation_m2c,
            pose.translation_m2c_m,
        )
        perception = self.profile["perception"]
        crop = (
            _fixed_crop(perception["crop_xyxy"], frame.rgb_bgr.shape[:2])
            if perception.get("crop_xyxy") is not None
            else _projected_crop(
                projected,
                frame.rgb_bgr.shape[:2],
                float(perception.get("crop_margin_ratio", 0.18)),
            )
        )
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
        prediction_started = perf_counter()
        curve, point_confidence, curve_confidence = self._predict_curve(
            frame.rgb_bgr,
            frame.depth,
            crop,
            pose,
            bundle=specialist if scene_decision.activated else None,
        )
        prediction_ms = (perf_counter() - prediction_started) * 1000.0
        selection = self.profile.get("selection", {})
        geometry = self.profile.get("geometry", {})
        estimate = estimate_level_from_sparse_contact(
            self.container_model,
            curve,
            self.calibration.camera_matrix,
            pose.rotation_m2c,
            pose.translation_m2c_m,
            point_confidences=point_confidence,
            min_point_confidence=float(selection.get("min_point_confidence", 0.5)),
            max_selected_points=int(selection.get("max_selected_points", 24)),
            horizontal_bins=int(selection.get("horizontal_bins", 8)),
            min_reliable_points=int(selection.get("min_reliable_points", 6)),
            min_horizontal_span_ratio=float(selection.get("min_horizontal_span_ratio", 0.5)),
            min_occupied_bins=int(selection.get("min_occupied_bins", 3)),
            geometry_options={
                "neighbors": int(geometry.get("neighbors", 8)),
                "max_reprojection_px": float(geometry.get("max_reprojection_px", 6.0)),
                "max_local_ambiguity_m": float(geometry.get("max_local_ambiguity_m", 0.015)),
                "max_global_spread_m": float(geometry.get("max_global_spread_m", 0.01)),
            },
        )
        reasons = list(estimate.rejection_reasons)
        max_pose_rmse = float(self.profile["pose"].get("max_reprojection_rmse_px", 2.5))
        if pose.reprojection_rmse_px > max_pose_rmse:
            reasons.append("container_pose_reprojection_too_large")
        raw_depth_m = estimate.level_m
        output = self.profile.get("output_calibration", {})
        scale = float(output.get("scale", 1.0)) if apply_output_calibration else 1.0
        offset = float(output.get("offset_m", 0.0)) if apply_output_calibration else 0.0
        candidate_m = None if raw_depth_m is None else scale * raw_depth_m + offset
        if not scene_decision.result_allowed:
            reasons.append("complex_model_required_but_unavailable")

        accepted = estimate.accepted and not reasons and candidate_m is not None
        confidence = estimate.confidence * float(
            np.exp(-pose.reprojection_rmse_px / max(max_pose_rmse, 1e-6))
        )
        filtered_m = None
        temporal_payload = None
        if self.temporal_filter is not None:
            temporal = self.temporal_filter.update(
                float(candidate_m) if candidate_m is not None else float("nan"),
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
            filtered_m = temporal.value
        total_ms = (perf_counter() - started) * 1000.0
        latency_within_budget = total_ms <= scene_decision.latency_budget_ms
        if not latency_within_budget and bool(
            self.profile.get("complex_scene", {}).get("enforce_latency_budget", True)
        ):
            accepted = False
            reasons.append("latency_budget_exceeded")
        reasons = list(dict.fromkeys(reasons))

        reported_m = filtered_m if accepted and filtered_m is not None else candidate_m
        result = {
            "schema_version": 1,
            "frame_id": frame.frame_id,
            "accepted": accepted,
            "liquid_depth_m": reported_m if accepted else None,
            "liquid_depth_candidate_m": candidate_m,
            "raw_geometry_level_m": raw_depth_m,
            "uncertainty_m": estimate.uncertainty_m,
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
            "pose": pose.to_dict(),
            "crop_xyxy": list(crop),
            "curve_confidence_uncalibrated": curve_confidence,
            "geometry": estimate.to_dict(),
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
            overlay = render_measurement_overlay(
                frame.rgb_bgr,
                crop,
                curve,
                point_confidence,
                estimate.selection.source_indices,
                accepted,
            )
            cv2.imwrite(str(target / "measurement_overlay.png"), overlay)
        return result
