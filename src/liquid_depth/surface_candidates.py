"""Research-only surface candidates, early normal memory and nonplanar statistics.

All outputs remain unverified candidates. Hard severe-loss/pose/support checks
are never relaxed; a deployment verifier must decide whether to publish them.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .surface_memory import robust_plane, world_points


def area_statistics(height):
    height = np.asarray(height, dtype=float)
    if height.size == 0 or not np.isfinite(height).all():
        return None
    return {
        "min_depth_m": float(height.min()),
        "max_depth_m": float(height.max()),
        "mean_depth_m": float(height.mean()),
        "p05_depth_m": float(np.percentile(height, 5)),
        "p95_depth_m": float(np.percentile(height, 95)),
        "peak_to_peak_m": float(np.ptp(height)),
    }


def design(xy, radii, degree):
    x, y = (xy / np.asarray(radii)).T
    if degree == 1:
        return np.column_stack((x, y, np.ones(len(x))))
    return np.column_stack((x * x, x * y, y * y, x, y, np.ones(len(x))))


class SurfaceCandidateEstimator:
    def __init__(
        self,
        *,
        mode="early",
        surface_mode="quasistatic",
        range_calibration=None,
        history_age=50,
        max_points=2048,
    ):
        if mode not in ("free", "gravity", "early") or surface_mode not in ("quasistatic", "waves"):
            raise ValueError("Unknown research surface mode")
        self.mode, self.surface_mode = mode, surface_mode
        if history_age < 1 or max_points < 24:
            raise ValueError("Positive history age and at least 24 points required")
        self.range_calibration = range_calibration
        self.history_age = history_age
        self.max_points = max_points
        self.frame = 0
        self.history = deque(maxlen=4)

    def reset(self):
        self.history.clear()

    def estimate(self, rgb, raw, prediction, k, pose, bottom, area_xy, radii, *, pose_valid=True):
        self.frame += 1
        while self.history and self.frame - self.history[0]["frame"] > self.history_age:
            self.history.popleft()
        result = {
            "accepted": False,
            "candidate_available": False,
            "level_m": None,
            "quality_flags": [],
            "requires_independent_verification": True,
            "mode": self.mode,
            "surface_mode": self.surface_mode,
            "history_records": len(self.history),
        }
        if not pose_valid or not np.isfinite(pose).all():
            self.reset()
            result["quality_flags"] = ["invalid_pose"]
            return result
        area_xy = np.asarray(area_xy, dtype=float)
        radii = np.asarray(radii, dtype=float)
        if (
            area_xy.ndim != 2
            or area_xy.shape[1] != 2
            or len(area_xy) == 0
            or not np.isfinite(area_xy).all()
            or radii.shape != (2,)
            or not np.isfinite(radii).all()
            or (radii <= 0).any()
        ):
            raise ValueError("Known footprint XY and positive radii required")
        mask = np.asarray(prediction["mask"], bool)
        valid = np.isfinite(raw) & (raw > 0) & mask
        ratio = float(valid.sum() / max(mask.sum(), 1))
        result["raw_valid_ratio"] = ratio
        if ratio <= 0.05:
            result["quality_flags"] = ["unsupported_95_100_percent_depth_failure"]
            return result
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        valid &= interior
        # Diagnostic selection intentionally avoids learned confidence rejection.
        y, x = np.nonzero(valid)
        if len(x) < 24:
            result["quality_flags"] = ["insufficient_current_depth"]
            return result
        take = np.linspace(0, len(x) - 1, min(self.max_points, len(x))).astype(int)
        y, x = y[take], x[take]
        points = world_points(raw[y, x], np.column_stack((x, y)), k, pose)
        distance = float(np.median(raw[y, x]))
        sigma = self.range_calibration.sigma(distance) if self.range_calibration else 0.003
        free = robust_plane(points)
        if free is None:
            result["quality_flags"] = ["degenerate_geometry"]
            return result
        shape_cov = np.cov((points[:, :2] / np.asarray(radii)).T)
        eigenvalues = np.linalg.eigvalsh(shape_cov)
        condition = float(max(eigenvalues[0], 0) / max(eigenvalues[-1], 1e-12))
        result.update(
            points=len(points),
            tilt_deg=free["tilt_deg"],
            residual_m=free["residual_m"],
            spatial_condition=condition,
            local_plane_level_m=free["level_world_m"] - bottom,
        )
        if free["tilt_deg"] > 12:
            result["quality_flags"].append("tilt_over_12")
        if free["residual_m"] > min(0.25, max(0.004, 1.5 * sigma)):
            result["quality_flags"].append("residual_over_noise_gate")
        if condition < 0.03:
            result["quality_flags"].append("narrow_spatial_support")
        # Historical normals intervene before any tilt/residual rejection.
        # Intercept is always re-estimated from fresh data; stale height is never output.
        prior = np.median([r["slope"] for r in self.history], axis=0) if self.history else np.zeros(2)
        result["early_history_used"] = bool(
            self.mode == "early" and self.history and self.surface_mode != "waves"
        )
        degree = 2 if self.surface_mode == "waves" else 1
        basis = design(points[:, :2], radii, degree)
        target = points[:, 2]
        if self.surface_mode == "quasistatic" and self.mode == "free":
            z = (area_xy - free["center"]) @ free["coef"][:2] + free["coef"][2]
        elif self.surface_mode == "quasistatic" and self.mode == "gravity":
            z = np.full(len(area_xy), np.median(target))
        else:
            penalty = np.zeros(basis.shape[1])
            center = np.zeros(basis.shape[1])
            if degree == 1:
                # 16 spatial blocks, 3 degree normal prior: no truth-driven tuning.
                penalty[:2] = (
                    len(points) * sigma * sigma / (16 * (np.tan(np.radians(3)) * np.asarray(radii)) ** 2)
                )
                center[:2] = prior * np.asarray(radii)
            else:
                # Weak smoothness only; do not flatten real waves toward old planes.
                penalty[:3] = len(points) * 1e-4
            weights = np.ones(len(points))
            for _ in range(4):
                lhs = basis.T @ (weights[:, None] * basis) + np.diag(penalty)
                rhs = basis.T @ (weights * target) + penalty * center
                coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
                residual = target - basis @ coef
                scale = max(0.001, sigma, 1.4826 * np.median(abs(residual - np.median(residual))))
                weights = np.minimum(1.0, 1.5 * scale / np.maximum(abs(residual), 1e-9))
            z = design(area_xy, radii, degree) @ coef
        stats = area_statistics(z - bottom)
        result.update(
            candidate_available=stats is not None,
            statistics=stats,
            level_m=stats["mean_depth_m"] if stats else None,
            observed_statistics=area_statistics(target - bottom),
        )
        if self.surface_mode == "waves":
            result["quality_flags"].append("global_wave_extrema_need_spatial_validation")
        # Geometrically reliable history, not a claim of independent metric truth.
        if (
            self.surface_mode == "quasistatic"
            and ratio >= 0.6
            and condition >= 0.03
            and free["tilt_deg"] <= 8
            and free["residual_m"] <= max(0.004, 1.5 * sigma)
        ):
            self.history.append({"frame": self.frame, "slope": free["coef"][:2].copy()})
        return result
