from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ContainerModel:
    """Container surface in model coordinates with a metric liquid-level axis."""

    points_m: np.ndarray
    level_axis: np.ndarray
    level_origin_m: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=np.float64)
        axis = np.asarray(self.level_axis, dtype=np.float64)
        origin = np.asarray(self.level_origin_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 10:
            raise ValueError("points_m must contain at least ten 3D points")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_m contains non-finite values")
        if axis.shape != (3,) or origin.shape != (3,):
            raise ValueError("level_axis and level_origin_m must be three-vectors")
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            raise ValueError("level_axis must be non-zero")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "level_axis", axis / norm)
        object.__setattr__(self, "level_origin_m", origin)

    @property
    def point_levels_m(self) -> np.ndarray:
        return (self.points_m - self.level_origin_m) @ self.level_axis

    @property
    def level_range_m(self) -> tuple[float, float]:
        levels = self.point_levels_m
        return float(levels.min()), float(levels.max())


@dataclass(frozen=True)
class ContactGeometryEstimate:
    level_m: float | None
    uncertainty_m: float | None
    curve_points: int
    matched_points: int
    inlier_points: int
    median_reprojection_px: float | None
    p95_reprojection_px: float | None
    coverage: float
    inlier_ratio: float
    rejection_reasons: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.level_m is not None and not self.rejection_reasons

    @property
    def geometric_confidence(self) -> float:
        if self.level_m is None:
            return 0.0
        reprojection = self.median_reprojection_px or 0.0
        uncertainty = self.uncertainty_m or 0.0
        scores = (
            max(self.coverage, 1e-6),
            max(self.inlier_ratio, 1e-6),
            np.exp(-reprojection / 3.0),
            np.exp(-uncertainty / 0.01),
        )
        return float(np.prod(scores) ** (1.0 / len(scores)))

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "geometric_confidence_uncalibrated": self.geometric_confidence,
            "level_m": self.level_m,
            "uncertainty_m": self.uncertainty_m,
            "curve_points": self.curve_points,
            "matched_points": self.matched_points,
            "inlier_points": self.inlier_points,
            "coverage": self.coverage,
            "inlier_ratio": self.inlier_ratio,
            "median_reprojection_px": self.median_reprojection_px,
            "p95_reprojection_px": self.p95_reprojection_px,
            "rejection_reasons": list(self.rejection_reasons),
        }


def project_model_points(
    model: ContainerModel,
    camera_matrix: np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a metric model surface and retain the model-space liquid levels."""
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    rotation = np.asarray(rotation_m2c, dtype=np.float64)
    translation = np.asarray(translation_m2c_m, dtype=np.float64)
    if matrix.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Expected camera_matrix/rotation/translation shapes (3,3)/(3,3)/(3,)")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-3):
        raise ValueError("rotation_m2c is not orthonormal")
    camera_points = model.points_m @ rotation.T + translation
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    camera_points = camera_points[valid]
    levels = model.point_levels_m[valid]
    homogeneous = camera_points @ matrix.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    finite = np.isfinite(pixels).all(axis=1)
    return pixels[finite], levels[finite], camera_points[finite]


def _local_matches(
    curve_pixels: np.ndarray,
    projected_pixels: np.ndarray,
    point_levels_m: np.ndarray,
    neighbors: int,
    chunk_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = min(neighbors, len(projected_pixels))
    local_level = np.empty(len(curve_pixels), dtype=np.float64)
    local_spread = np.empty(len(curve_pixels), dtype=np.float64)
    nearest_distance = np.empty(len(curve_pixels), dtype=np.float64)
    for start in range(0, len(curve_pixels), chunk_size):
        stop = min(start + chunk_size, len(curve_pixels))
        squared = np.sum(
            (curve_pixels[start:stop, None, :] - projected_pixels[None, :, :]) ** 2,
            axis=2,
        )
        indices = np.argpartition(squared, neighbors - 1, axis=1)[:, :neighbors]
        distances = np.sqrt(np.take_along_axis(squared, indices, axis=1))
        levels = point_levels_m[indices]
        weights = 1.0 / np.maximum(distances, 0.25) ** 2
        weights /= weights.sum(axis=1, keepdims=True)
        estimates = np.sum(weights * levels, axis=1)
        local_level[start:stop] = estimates
        local_spread[start:stop] = np.sqrt(
            np.sum(weights * (levels - estimates[:, None]) ** 2, axis=1)
        )
        nearest_distance[start:stop] = distances.min(axis=1)
    return local_level, local_spread, nearest_distance


def estimate_level_from_contact_curve(
    model: ContainerModel,
    contact_curve_pixels: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_m2c: np.ndarray,
    translation_m2c_m: np.ndarray,
    *,
    neighbors: int = 8,
    max_reprojection_px: float = 6.0,
    max_local_ambiguity_m: float = 0.015,
    min_matches: int = 12,
    min_coverage: float = 0.6,
    min_inlier_ratio: float = 0.65,
    outlier_sigma: float = 3.5,
    max_global_spread_m: float = 0.01,
    min_outlier_limit_m: float = 0.002,
) -> ContactGeometryEstimate:
    """Map a 2D air-liquid contact curve to a robust metric model height.

    This implements the CAD-projection stage of TCLD, but uses reprojection and
    height-ambiguity gates plus a robust consensus estimate instead of an
    unconditional average. The returned uncertainty is geometric consistency,
    not a calibrated sensor confidence interval.
    """
    curve = np.asarray(contact_curve_pixels, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError("contact_curve_pixels must have shape (N, 2)")
    if len(curve) < 2 or not np.all(np.isfinite(curve)):
        raise ValueError("contact_curve_pixels must contain finite curve samples")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")

    projected, levels, _ = project_model_points(
        model,
        camera_matrix,
        rotation_m2c,
        translation_m2c_m,
    )
    if len(projected) < neighbors:
        raise ValueError("Too few visible model points for contact-curve matching")
    local_level, local_spread, distance = _local_matches(curve, projected, levels, neighbors)
    matched = (distance <= max_reprojection_px) & (local_spread <= max_local_ambiguity_m)
    matched_count = int(matched.sum())
    coverage = matched_count / len(curve)
    reasons: list[str] = []
    if matched_count < min_matches:
        reasons.append("insufficient_geometry_matches")
    if coverage < min_coverage:
        reasons.append("low_curve_coverage")
    if matched_count == 0:
        return ContactGeometryEstimate(
            None,
            None,
            len(curve),
            0,
            0,
            None,
            None,
            coverage,
            0.0,
            tuple(reasons),
        )

    candidate_level = local_level[matched]
    candidate_spread = local_spread[matched]
    candidate_distance = distance[matched]
    median = float(np.median(candidate_level))
    mad = float(np.median(np.abs(candidate_level - median)))
    robust_sigma = 1.4826 * mad
    limit = max(min_outlier_limit_m, outlier_sigma * robust_sigma)
    inliers = np.abs(candidate_level - median) <= limit
    if robust_sigma > max_global_spread_m:
        reasons.append("inconsistent_or_multimodal_contact_height")
    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / matched_count
    if (
        inlier_ratio < min_inlier_ratio
        and "inconsistent_or_multimodal_contact_height" not in reasons
    ):
        reasons.append("inconsistent_or_multimodal_contact_height")

    inlier_level = candidate_level[inliers]
    inlier_spread = candidate_spread[inliers]
    inlier_distance = candidate_distance[inliers]
    if inlier_count == 0:
        return ContactGeometryEstimate(
            None,
            None,
            len(curve),
            matched_count,
            0,
            float(np.median(candidate_distance)),
            float(np.percentile(candidate_distance, 95)),
            coverage,
            0.0,
            tuple(reasons),
        )

    weights = 1.0 / np.maximum(inlier_distance, 0.5) ** 2
    weights /= weights.sum()
    level_m = float(np.sum(weights * inlier_level))
    consensus_sigma = 1.4826 * float(
        np.median(np.abs(inlier_level - np.median(inlier_level)))
    )
    effective_count = float(1.0 / np.sum(weights**2))
    uncertainty_m = float(
        np.hypot(
            consensus_sigma / np.sqrt(max(effective_count, 1.0)),
            np.median(inlier_spread),
        )
    )
    minimum, maximum = model.level_range_m
    if level_m < minimum - min_outlier_limit_m or level_m > maximum + min_outlier_limit_m:
        reasons.append("level_outside_container_range")
    return ContactGeometryEstimate(
        level_m,
        uncertainty_m,
        len(curve),
        matched_count,
        inlier_count,
        float(np.median(inlier_distance)),
        float(np.percentile(inlier_distance, 95)),
        coverage,
        inlier_ratio,
        tuple(reasons),
    )


def sample_axisymmetric_container(
    profile_heights_m: np.ndarray,
    profile_radii_m: np.ndarray,
    *,
    vertical_samples: int = 300,
    angular_samples: int = 360,
    level_axis: np.ndarray | tuple[float, float, float] = (0.0, 1.0, 0.0),
    radial_axis: np.ndarray | tuple[float, float, float] = (1.0, 0.0, 0.0),
    origin_m: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ContainerModel:
    """Create a metric surface from a measured radius-versus-height profile."""
    heights = np.asarray(profile_heights_m, dtype=np.float64)
    radii = np.asarray(profile_radii_m, dtype=np.float64)
    if heights.ndim != 1 or radii.shape != heights.shape or len(heights) < 2:
        raise ValueError("Profile heights and radii must be equal one-dimensional arrays")
    if np.any(np.diff(heights) <= 0) or np.any(radii <= 0):
        raise ValueError("Profile heights must increase and all radii must be positive")
    if vertical_samples < 2 or angular_samples < 8:
        raise ValueError("Insufficient surface sampling")
    axis = np.array(level_axis, dtype=np.float64, copy=True)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        raise ValueError("level_axis must be non-zero")
    axis /= axis_norm
    radial = np.array(radial_axis, dtype=np.float64, copy=True)
    radial -= np.dot(radial, axis) * axis
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1e-12:
        raise ValueError("radial_axis must not be parallel to level_axis")
    radial /= radial_norm
    tangent = np.cross(axis, radial)
    sampled_heights = np.linspace(heights[0], heights[-1], vertical_samples)
    sampled_radii = np.interp(sampled_heights, heights, radii)
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    rings = (
        np.asarray(origin_m, dtype=np.float64)[None, None, :]
        + sampled_heights[:, None, None] * axis[None, None, :]
        + sampled_radii[:, None, None]
        * (
            np.cos(theta)[None, :, None] * radial[None, None, :]
            + np.sin(theta)[None, :, None] * tangent[None, None, :]
        )
    )
    return ContainerModel(
        rings.reshape(-1, 3),
        axis,
        np.asarray(origin_m, dtype=np.float64),
    )


def load_container_model(
    path: str | Path,
    level_axis: np.ndarray,
    level_origin_m: np.ndarray,
) -> ContainerModel:
    """Load a metric point cloud from NPY, NPZ, XYZ/TXT/CSV, or ASCII PLY."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        key = "points" if "points" in payload else payload.files[0]
        points = payload[key]
    elif suffix in {".xyz", ".txt", ".csv"}:
        points = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    elif suffix == ".ply":
        lines = path.read_text(encoding="ascii").splitlines()
        if not lines or lines[0].strip() != "ply" or "format ascii 1.0" not in lines[:5]:
            raise ValueError("Only ASCII PLY is supported directly; convert binary PLY to NPY")
        vertex_count = next(
            int(line.split()[-1]) for line in lines if line.startswith("element vertex ")
        )
        header_end = lines.index("end_header") + 1
        points = np.asarray(
            [
                [float(value) for value in line.split()[:3]]
                for line in lines[header_end : header_end + vertex_count]
            ]
        )
    else:
        raise ValueError(f"Unsupported container model format: {suffix}")
    return ContainerModel(np.asarray(points)[:, :3], level_axis, level_origin_m)
