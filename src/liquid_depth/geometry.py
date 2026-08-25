from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import write_json


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray
    d: float
    centroid: np.ndarray

    def distances(self, points: np.ndarray) -> np.ndarray:
        return np.abs(points @ self.normal + self.d)

    def signed_distance(self, point: np.ndarray) -> float:
        return float(np.dot(self.normal, point) + self.d)

    def to_dict(self) -> dict:
        return {
            "coordinate_system": "camera coordinates in meters; x right, y down, z forward",
            "plane_ax_by_cz_d_eq_0": [*map(float, self.normal), float(self.d)],
            "normal_unit": self.normal.astype(float).tolist(),
            "point_on_plane_centroid_m": self.centroid.astype(float).tolist(),
        }


@dataclass(frozen=True)
class PlaneFit:
    plane: Plane
    input_points: int
    inlier_points: int
    median_residual_m: float
    mean_residual_m: float
    inlier_pixels: np.ndarray

    @property
    def inlier_ratio(self) -> float:
        return self.inlier_points / self.input_points

    def to_dict(self) -> dict:
        return {
            **self.plane.to_dict(),
            "input_points_after_mask_and_depth_filter": self.input_points,
            "final_inliers": self.inlier_points,
            "final_inlier_ratio_vs_used_points": self.inlier_ratio,
            "median_abs_residual_m": self.median_residual_m,
            "mean_abs_residual_m": self.mean_residual_m,
        }


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    result = depth.astype(np.float64, copy=True)
    valid = np.isfinite(result) & (result > 0)
    if not np.any(valid):
        raise ValueError("Depth image contains no positive finite values")
    if float(np.median(result[valid])) > 10.0:
        result /= 1000.0
    return result


def masked_points(
    depth: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_to_meters(depth)
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)
    low, high = np.percentile(depth_m[valid], percentiles)
    valid &= (depth_m >= low) & (depth_m <= high)
    v, u = np.where(valid)
    z = depth_m[v, u]
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return points, np.column_stack((u, v)).astype(np.int32)


def _normalize_plane(normal: np.ndarray, d: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("Degenerate plane")
    normal = normal / norm
    d /= norm
    if normal[2] > 0:
        normal, d = -normal, -d
    return normal, float(d)


def _least_squares(points: np.ndarray) -> Plane:
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal, d = _normalize_plane(normal, -float(np.dot(normal, centroid)))
    return Plane(normal, d, centroid)


def fit_plane(
    points: np.ndarray,
    pixels: np.ndarray,
    threshold_m: float = 0.006,
    max_points: int = 30000,
    iterations: int = 1000,
    seed: int = 7,
) -> PlaneFit:
    if len(points) < 30:
        raise ValueError(f"At least 30 valid depth points are required; got {len(points)}")
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        selected = rng.choice(len(points), max_points, replace=False)
        points, pixels = points[selected], pixels[selected]

    best = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        p0, p1, p2 = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        candidate = np.abs(points @ normal - np.dot(normal, p0)) < threshold_m
        if candidate.sum() > best.sum():
            best = candidate
    if best.sum() < 3:
        raise RuntimeError("RANSAC could not find a plane")

    plane = _least_squares(points[best])
    residuals = plane.distances(points[best])
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_limit = max(0.001, median + 2.5 * 1.4826 * mad)
    refined = residuals <= robust_limit
    plane = _least_squares(points[best][refined])
    final_points = points[best][refined]
    final_pixels = pixels[best][refined]
    residuals = plane.distances(final_points)
    return PlaneFit(
        plane=plane,
        input_points=len(points),
        inlier_points=len(final_points),
        median_residual_m=float(np.median(residuals)),
        mean_residual_m=float(np.mean(residuals)),
        inlier_pixels=final_pixels,
    )


def fit_plane_from_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    erode_px: int,
    threshold_m: float,
    max_points: int,
    seed: int,
) -> PlaneFit:
    import cv2

    fit_mask = mask
    if erode_px > 0:
        size = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(mask, kernel)
        if cv2.countNonZero(eroded) >= 500:
            fit_mask = eroded
    points, pixels = masked_points(depth, fit_mask, camera_matrix)
    return fit_plane(points, pixels, threshold_m=threshold_m, max_points=max_points, seed=seed)


def save_plane(path: str | Path, fit: PlaneFit, kind: str, frame_id: str) -> None:
    write_json(path, {"frame_id": frame_id, "kind": kind, **fit.to_dict()})


def load_plane(path: str | Path) -> Plane:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    coefficients = np.asarray(payload["plane_ax_by_cz_d_eq_0"], dtype=np.float64)
    normal, d = _normalize_plane(coefficients[:3], float(coefficients[3]))
    centroid = np.asarray(payload["point_on_plane_centroid_m"], dtype=np.float64)
    return Plane(normal, d, centroid)

