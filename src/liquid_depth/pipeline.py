from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import PlaneFit, fit_plane_from_mask, load_plane, plane_angle_degrees, save_plane
from .io import load_frame, write_json
from .refinement import make_depth_refiner
from .segmentation import make_bottom_mask, make_segmenter, overlay_mask


def _fit(
    frame,
    mask: np.ndarray,
    config: dict,
    erode_key: str,
    depth: np.ndarray | None = None,
    depth_confidence: np.ndarray | None = None,
) -> PlaneFit:
    geometry = config["geometry"]
    return fit_plane_from_mask(
        frame.depth if depth is None else depth,
        mask,
        frame.camera_matrix,
        erode_px=int(geometry[erode_key]),
        threshold_m=float(geometry["ransac_threshold_m"]),
        max_points=int(geometry["max_points"]),
        seed=int(geometry["seed"]),
        confidence=depth_confidence,
        min_confidence=float(geometry.get("min_depth_confidence", 0.0)),
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
) -> dict:
    frame = load_frame(frame_dir)
    segmenter = make_segmenter(config)
    mask, segmentation_confidence = segmenter.predict(frame.rgb_bgr)
    refined = make_depth_refiner(config).predict(frame.rgb_bgr, frame.depth)
    fit = _fit(
        frame,
        mask,
        config,
        "liquid_erode_px",
        depth=refined.depth_m,
        depth_confidence=refined.confidence,
    )
    bottom = load_plane(bottom_plane_path)
    raw_gap_m = abs(bottom.signed_distance(fit.plane.centroid))
    plane_angle_deg = plane_angle_degrees(bottom, fit.plane)
    output = config["output"]
    scale = float(output["calibration_scale_per_meter"])
    depth = raw_gap_m * scale

    quality = config.get("quality", {})
    mask_pixels = mask > 0
    mask_area = int(mask_pixels.sum())
    raw_depth_m = frame.depth.astype(np.float64)
    finite_depth = raw_depth_m[np.isfinite(raw_depth_m)]
    if finite_depth.size and np.any(finite_depth > 10.0):
        raw_depth_m /= 1000.0
    raw_valid_ratio = float(
        (np.isfinite(raw_depth_m) & (raw_depth_m > 0) & mask_pixels).sum() / max(mask_area, 1)
    )
    mean_segmentation_confidence = (
        float(segmentation_confidence[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
    )
    mean_depth_confidence = float(refined.confidence[mask_pixels].mean()) if np.any(mask_pixels) else 0.0
    accepted = (
        fit.inlier_ratio >= float(quality.get("min_inlier_ratio", 0.30))
        and fit.median_residual_m <= float(quality.get("max_median_residual_m", 0.006))
        and mask_area >= int(quality.get("min_mask_area", 10000))
        and mean_segmentation_confidence >= float(quality.get("min_mean_segmentation_confidence", 0.0))
        and mean_depth_confidence >= float(quality.get("min_mean_depth_confidence", 0.0))
        and plane_angle_deg <= float(quality.get("max_plane_angle_deg", 15.0))
    )
    result = {
        "frame_id": frame.frame_id,
        "accepted": accepted,
        "segmentation_backend": config["segmentation"]["backend"],
        "depth_refinement_backend": refined.backend,
        "mask_area_px": mask_area,
        "mean_mask_confidence": mean_segmentation_confidence,
        "raw_depth_valid_ratio_in_mask": raw_valid_ratio,
        "mean_refined_depth_confidence": mean_depth_confidence,
        "liquid_bottom_plane_angle_deg": plane_angle_deg,
        "raw_bottom_gap_m": raw_gap_m,
        "liquid_depth": depth,
        "liquid_depth_unit": output["depth_unit"],
        "calibration_scale_per_meter": scale,
        "liquid_plane": fit.to_dict(),
    }
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / "liquid_mask.png"), mask)
    cv2.imwrite(str(target / "liquid_mask_vis.png"), overlay_mask(frame.rgb_bgr, mask))
    np.save(target / "liquid_confidence.npy", segmentation_confidence)
    np.save(target / "refined_depth_m.npy", refined.depth_m)
    np.save(target / "refined_depth_confidence.npy", refined.confidence)
    save_plane(target / "liquid_plane.json", fit, "liquid", frame.frame_id)
    write_json(target / "depth_result.json", result)
    return result
