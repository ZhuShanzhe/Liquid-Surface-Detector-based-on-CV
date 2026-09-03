from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .confidence_policy import select_confidence_gate
from .geometry import (
    PlaneFit,
    depth_to_meters,
    fit_plane_from_mask,
    load_plane,
    plane_angle_degrees,
    save_plane,
    split_surface_mask,
)
from .illumination import adaptive_exposure_correction
from .io import load_frame, write_json
from .quality import assess_quality
from .refinement import (
    make_complex_depth_refiners,
    make_depth_refiner,
)
from .scenario_policy import (
    ComplexScenePolicy,
    load_scene_context,
    measure_scene_signals,
)
from .segmentation import make_bottom_mask, make_segmenter, overlay_mask
from .surface_support import assess_planar_support
from .temporal import RobustKalmanFilter


def _fit(
    frame,
    mask: np.ndarray,
    config: dict,
    erode_key: str,
    depth: np.ndarray | None = None,
    depth_confidence: np.ndarray | None = None,
    min_depth_confidence: float | None = None,
) -> PlaneFit:
    geometry = config["geometry"]
    minimum_confidence = (
        float(geometry.get("min_depth_confidence", 0.0))
        if min_depth_confidence is None
        else float(min_depth_confidence)
    )
    return fit_plane_from_mask(
        frame.depth if depth is None else depth,
        mask,
        frame.camera_matrix,
        erode_px=int(geometry[erode_key]),
        threshold_m=float(geometry["ransac_threshold_m"]),
        max_points=int(geometry["max_points"]),
        seed=int(geometry["seed"]),
        confidence=depth_confidence,
        min_confidence=minimum_confidence,
    )


def fit_bottom(frame_dir: str | Path, output_path: str | Path, config: dict) -> dict:
    frame = load_frame(frame_dir)
    mask = make_bottom_mask(frame.rgb_bgr.shape, config)
    fit = _fit(frame, mask, config, "bottom_erode_px")
    save_plane(output_path, fit, "bucket_bottom", frame.frame_id)
    output_dir = Path(output_path).parent
    cv2.imwrite(str(output_dir / "bottom_mask.png"), mask)
    cv2.imwrite(str(output_dir / "bottom_mask_vis.png"), overlay_mask(frame.rgb_bgr, mask))
    return {"frame_id": frame.frame_id, "plane_path": str(output_path), **fit.to_dict()}


def infer_frame(
    frame_dir: str | Path,
    bottom_plane_path: str | Path,
    output_dir: str | Path,
    config: dict,
    temporal_filter: RobustKalmanFilter | None = None,
    segmenter=None,
    depth_refiner=None,
    complex_depth_refiners=None,
    scene_policy: ComplexScenePolicy | None = None,
) -> dict:
    frame = load_frame(frame_dir)
    segmenter = segmenter or make_segmenter(config)
    depth_refiner = depth_refiner or make_depth_refiner(config)
    if complex_depth_refiners is None:
        complex_depth_refiners = make_complex_depth_refiners(config)
    specialists = complex_depth_refiners
    scene_policy = scene_policy or ComplexScenePolicy(
        config.get("complex_scene", {}),
        available_variants=specialists,
    )
    started = perf_counter()
    exposure = adaptive_exposure_correction(frame.rgb_bgr, config.get("illumination", {}))
    perception_rgb = exposure.image_bgr
    mask, segmentation_confidence = segmenter.predict(perception_rgb)
    mask_pixels = mask > 0
    mask_area = int(mask_pixels.sum())
    raw_depth_m = depth_to_meters(frame.depth)
    raw_valid_ratio = float(
        (np.isfinite(raw_depth_m) & (raw_depth_m > 0) & mask_pixels).sum() / max(mask_area, 1)
    )
    signals = measure_scene_signals(perception_rgb, frame.depth)
    signals = replace(
        signals,
        raw_depth_valid_ratio=raw_valid_ratio,
    )
    scene_context = dict(config.get("complex_scene", {}).get("scene_context", {}))
    scene_context.update(load_scene_context(frame.source_dir))
    scene_decision = scene_policy.decide(signals, scene_context)
    confidence_gate = select_confidence_gate(
        config.get("confidence_policy", {}),
        model_variant=scene_decision.model_variant,
        triggers=scene_decision.triggers,
        context=scene_context,
    )
    base_min_confidence = float(config["geometry"].get("min_depth_confidence", 0.0))
    fit_min_confidence = max(
        base_min_confidence,
        confidence_gate.threshold if confidence_gate.enabled else 0.0,
    )
    active_refiner = depth_refiner
    if scene_decision.activated:
        active_refiner = specialists[scene_decision.model_variant]
    depth_started = perf_counter()
    refined = active_refiner.predict(perception_rgb, frame.depth)
    depth_refinement_ms = (perf_counter() - depth_started) * 1000.0
    geometry = config["geometry"]
    interior_mask, meniscus_mask = split_surface_mask(
        mask,
        interior_erode_px=int(geometry["liquid_erode_px"]),
        meniscus_width_px=int(geometry.get("meniscus_width_px", geometry["liquid_erode_px"])),
    )
    support_config = config.get("surface_support", {})
    support_options = {key: value for key, value in support_config.items() if key != "enabled"}
    try:
        fit = _fit(
            frame,
            mask,
            config,
            "liquid_erode_px",
            depth=refined.depth_m,
            depth_confidence=refined.confidence,
            min_depth_confidence=fit_min_confidence,
        )
    except (RuntimeError, ValueError) as error:
        planar_support = assess_planar_support(
            interior_mask,
            np.empty((0, 2), dtype=np.int32),
            fit_inlier_ratio=0.0,
            **support_options,
        )
        primary_reason = (
            "liquid_surface_mask_empty" if mask_area == 0 else "insufficient_liquid_depth_support"
        )
        rejection_reasons = [
            primary_reason,
            *planar_support.rejection_reasons,
        ]
        if not scene_decision.result_allowed:
            rejection_reasons.append("complex_model_required_but_unavailable")
        if confidence_gate.enabled:
            rejection_reasons.append("scenario_confidence_support_insufficient")
        if not confidence_gate.result_allowed:
            rejection_reasons.append("scenario_confidence_not_qualified")
        inference_total_ms = (perf_counter() - started) * 1000.0
        latency_budget_ms = scene_decision.latency_budget_ms
        latency_within_budget = inference_total_ms <= latency_budget_ms
        if not latency_within_budget and bool(
            config.get("complex_scene", {}).get(
                "enforce_latency_budget",
                True,
            )
        ):
            rejection_reasons.append("latency_budget_exceeded")
        rejection_reasons = list(dict.fromkeys(rejection_reasons))
        mean_segmentation_confidence = (
            float(segmentation_confidence[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
        )
        mean_depth_confidence = float(refined.confidence[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
        result = {
            "frame_id": frame.frame_id,
            "accepted": False,
            "confidence": 0.0,
            "rejection_reasons": rejection_reasons,
            "fit_error": f"{type(error).__name__}: {error}",
            "quality_scores": {},
            "segmentation_backend": config["segmentation"]["backend"],
            "depth_refinement_backend": refined.backend,
            "illumination": exposure.to_dict(),
            "planar_support": planar_support.to_dict(),
            "mask_area_px": mask_area,
            "complex_scene": scene_decision.to_dict(),
            "confidence_gate": confidence_gate.to_dict(),
            "latency": {
                "inference_total_ms": inference_total_ms,
                "depth_refinement_ms": depth_refinement_ms,
                "budget_ms": latency_budget_ms,
                "within_budget": latency_within_budget,
                "budget_scope": ("frame_load_excluded_artifact_serialization_excluded"),
            },
            "interior_mask_area_px": int((interior_mask > 0).sum()),
            "meniscus_mask_area_px": int((meniscus_mask > 0).sum()),
            "mean_mask_confidence": mean_segmentation_confidence,
            "raw_depth_valid_ratio_in_mask": raw_valid_ratio,
            "mean_refined_depth_confidence": mean_depth_confidence,
            "liquid_bottom_plane_angle_deg": None,
            "raw_bottom_gap_m": None,
            "liquid_depth_raw": None,
            "liquid_depth_filtered": None,
            "liquid_depth": None,
            "liquid_depth_unit": config["output"]["depth_unit"],
            "calibration_scale_per_meter": config["output"]["calibration_scale_per_meter"],
            "temporal": None,
            "liquid_plane": None,
        }
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target / "liquid_mask.png"), mask)
        cv2.imwrite(
            str(target / "liquid_interior_mask.png"),
            interior_mask,
        )
        cv2.imwrite(
            str(target / "liquid_meniscus_mask.png"),
            meniscus_mask,
        )
        cv2.imwrite(
            str(target / "liquid_mask_vis.png"),
            overlay_mask(frame.rgb_bgr, mask),
        )
        if exposure.applied:
            cv2.imwrite(
                str(target / "illumination_corrected.png"),
                perception_rgb,
            )
        np.save(target / "liquid_confidence.npy", segmentation_confidence)
        np.save(target / "refined_depth_m.npy", refined.depth_m)
        np.save(
            target / "refined_depth_confidence.npy",
            refined.confidence,
        )
        write_json(target / "depth_result.json", result)
        return result
    planar_support = assess_planar_support(
        interior_mask,
        fit.inlier_pixels,
        fit_inlier_ratio=fit.inlier_ratio,
        **support_options,
    )
    bottom = load_plane(bottom_plane_path)
    raw_gap_m = abs(bottom.signed_distance(fit.plane.centroid))
    plane_angle_deg = plane_angle_degrees(bottom, fit.plane)
    output = config["output"]
    scale = float(output["calibration_scale_per_meter"])
    raw_liquid_depth = raw_gap_m * scale

    quality = config.get("quality", {})
    mean_segmentation_confidence = (
        float(segmentation_confidence[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
    )
    illumination_metrics = exposure.before.to_dict()
    mean_depth_confidence = float(refined.confidence[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
    assessment = assess_quality(
        {
            "inlier_ratio": fit.inlier_ratio,
            "median_residual_m": fit.median_residual_m,
            "mask_area_px": mask_area,
            "mean_segmentation_confidence": mean_segmentation_confidence,
            "mean_depth_confidence": mean_depth_confidence,
            "plane_angle_deg": plane_angle_deg,
            **illumination_metrics,
        },
        quality,
    )
    rejection_reasons = list(assessment.rejection_reasons)
    accepted = assessment.accepted
    final_confidence = assessment.confidence
    if bool(support_config.get("enabled", False)):
        accepted = accepted and planar_support.accepted
        if not planar_support.accepted:
            rejection_reasons.extend(planar_support.rejection_reasons)
        support_confidence = (
            planar_support.tile_coverage
            * max(planar_support.horizontal_span_ratio, 1e-6)
            * max(planar_support.vertical_span_ratio, 1e-6)
        ) ** (1.0 / 3.0)
        final_confidence = min(final_confidence, support_confidence)
    filtered_depth: float | None = None
    if not scene_decision.result_allowed:
        accepted = False
        rejection_reasons.append("complex_model_required_but_unavailable")
    if not confidence_gate.result_allowed:
        accepted = False
        rejection_reasons.append("scenario_confidence_not_qualified")

    temporal_payload: dict | None = None
    if temporal_filter is not None:
        temporal = temporal_filter.update(raw_liquid_depth, final_confidence, accepted)
        filtered_depth = temporal.value
        temporal_payload = {
            "enabled": True,
            "accepted": temporal.accepted,
            "variance": temporal.variance,
            "confidence": temporal.confidence,
            "innovation": temporal.innovation,
            "rejection_reason": temporal.reason,
            "recovered": temporal.recovered,
            "hold_frames": temporal.hold_frames,
        }
        accepted = accepted and temporal.accepted
        final_confidence = min(final_confidence, temporal.confidence)
        if temporal.reason and temporal.reason != "upstream_quality_rejection":
            rejection_reasons.append(temporal.reason)

    reported_depth = filtered_depth if filtered_depth is not None else raw_liquid_depth
    inference_total_ms = (perf_counter() - started) * 1000.0
    latency_budget_ms = scene_decision.latency_budget_ms
    latency_within_budget = inference_total_ms <= latency_budget_ms
    if not latency_within_budget and bool(
        config.get("complex_scene", {}).get("enforce_latency_budget", True)
    ):
        accepted = False
        rejection_reasons.append("latency_budget_exceeded")
    rejection_reasons = list(dict.fromkeys(rejection_reasons))

    result = {
        "frame_id": frame.frame_id,
        "accepted": accepted,
        "confidence": final_confidence,
        "rejection_reasons": rejection_reasons,
        "quality_scores": assessment.scores,
        "segmentation_backend": config["segmentation"]["backend"],
        "depth_refinement_backend": refined.backend,
        "illumination": exposure.to_dict(),
        "planar_support": planar_support.to_dict(),
        "mask_area_px": mask_area,
        "complex_scene": scene_decision.to_dict(),
        "confidence_gate": confidence_gate.to_dict(),
        "latency": {
            "inference_total_ms": inference_total_ms,
            "depth_refinement_ms": depth_refinement_ms,
            "budget_ms": latency_budget_ms,
            "within_budget": latency_within_budget,
            "budget_scope": "frame_load_excluded_artifact_serialization_excluded",
        },
        "interior_mask_area_px": int((interior_mask > 0).sum()),
        "meniscus_mask_area_px": int((meniscus_mask > 0).sum()),
        "mean_mask_confidence": mean_segmentation_confidence,
        "raw_depth_valid_ratio_in_mask": raw_valid_ratio,
        "mean_refined_depth_confidence": mean_depth_confidence,
        "liquid_bottom_plane_angle_deg": plane_angle_deg,
        "raw_bottom_gap_m": raw_gap_m,
        "liquid_depth_raw": raw_liquid_depth,
        "liquid_depth_filtered": filtered_depth,
        "liquid_depth": reported_depth,
        "liquid_depth_unit": output["depth_unit"],
        "calibration_scale_per_meter": scale,
        "temporal": temporal_payload,
        "liquid_plane": fit.to_dict(),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / "liquid_mask.png"), mask)
    cv2.imwrite(str(target / "liquid_interior_mask.png"), interior_mask)
    cv2.imwrite(str(target / "liquid_meniscus_mask.png"), meniscus_mask)
    cv2.imwrite(str(target / "liquid_mask_vis.png"), overlay_mask(frame.rgb_bgr, mask))
    if exposure.applied:
        cv2.imwrite(str(target / "illumination_corrected.png"), perception_rgb)
    np.save(target / "liquid_confidence.npy", segmentation_confidence)
    np.save(target / "refined_depth_m.npy", refined.depth_m)
    np.save(target / "refined_depth_confidence.npy", refined.confidence)
    save_plane(target / "liquid_plane.json", fit, "liquid", frame.frame_id)
    write_json(target / "depth_result.json", result)
    return result
