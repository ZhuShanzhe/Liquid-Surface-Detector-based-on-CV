from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import PlaneFit, fit_plane_from_mask, load_plane, save_plane
from .io import load_frame, write_json
from .segmentation import make_bottom_mask, make_segmenter, overlay_mask


def _fit(frame, mask: np.ndarray, config: dict, erode_key: str) -> PlaneFit:
    geometry = config["geometry"]
    return fit_plane_from_mask(
        frame.depth,
        mask,
        frame.camera_matrix,
        erode_px=int(geometry[erode_key]),
        threshold_m=float(geometry["ransac_threshold_m"]),
        max_points=int(geometry["max_points"]),
        seed=int(geometry["seed"]),
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
    mask, confidence = segmenter.predict(frame.rgb_bgr)
    fit = _fit(frame, mask, config, "liquid_erode_px")
    bottom = load_plane(bottom_plane_path)
    raw_gap_m = abs(bottom.signed_distance(fit.plane.centroid))
    output = config["output"]
    scale = float(output["calibration_scale_per_meter"])
    depth = raw_gap_m * scale

    quality = config.get("quality", {})
    accepted = (
        fit.inlier_ratio >= float(quality.get("min_inlier_ratio", 0.30))
        and fit.median_residual_m <= float(quality.get("max_median_residual_m", 0.006))
        and int(cv2.countNonZero(mask)) >= int(quality.get("min_mask_area", 1000))
    )
    result = {
        "frame_id": frame.frame_id,
        "accepted": accepted,
        "segmentation_backend": config["segmentation"]["backend"],
        "mask_area_px": int(cv2.countNonZero(mask)),
        "mean_mask_confidence": float(confidence[mask > 0].mean()) if np.any(mask > 0) else 0.0,
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
    np.save(target / "liquid_confidence.npy", confidence)
    save_plane(target / "liquid_plane.json", fit, "liquid", frame.frame_id)
    write_json(target / "depth_result.json", result)
    return result

