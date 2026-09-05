"""Research-only balanced height, sensor-domain likelihood and local surface bounds."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

from .surface_candidates import area_statistics
from .surface_memory import world_points


@dataclass(frozen=True)
class StereoNoiseModel:
    """Explicit device calibration; the proxy presets describe simulation, NOT hardware."""

    baseline_m: float
    disparity_sigma_px: float
    depth_noise_floor_m: float
    depth_noise_quadratic: float
    min_depth_m: float = 0.12
    max_depth_m: float = 10.5

    def __post_init__(self):
        values = np.array(list(self.__dict__.values()), dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any() or self.min_depth_m >= self.max_depth_m:
            raise ValueError("Invalid calibrated stereo noise model")

    @classmethod
    def simulation_proxy(cls, sensor):
        if sensor == "active_stereo":
            return cls(0.055, 0.18, 0.0006, 0.0011)
        if sensor == "structured_light":
            return cls(0.035, 0.26, 0.0008, 0.0015)
        raise ValueError("ToF must not use a stereo disparity likelihood")


def support(raw, mask, k, pose, radii, max_points=4096):
    mask = np.asarray(mask, bool)
    valid = mask & np.isfinite(raw) & (raw > 0)
    if valid.sum() <= 0.05 * max(mask.sum(), 1):
        return None, "unsupported_95_100_percent_depth_failure"
    valid &= cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    y, x = np.nonzero(valid)
    if len(x) < 24:
        return None, "insufficient_current_depth"
    ids = np.linspace(0, len(x) - 1, min(max_points, len(x))).astype(int)
    pixels = np.column_stack((x[ids], y[ids]))
    z = raw[y[ids], x[ids]]
    points = world_points(z, pixels, k, pose)
    # Balance tiles in angular/image coordinates, not noisy depth-dependent XY.
    tile = (x[ids] * 8 // raw.shape[1]) + 8 * (y[ids] * 8 // raw.shape[0])
    _, inverse, counts = np.unique(tile, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    weights /= weights.sum()
    return (points, pixels, z, weights, tile), None


def weighted_median(x, w):
    ids = np.argsort(x)
    return float(x[ids[np.searchsorted(np.cumsum(w[ids]), 0.5 * w.sum())]])


def censored_clearance(depth, cosine, weights, fx, model):
    fb = fx * model.baseline_m
    measured = fb / depth
    lo, hi = fb / model.max_depth_m, fb / model.min_depth_m
    good = (measured >= lo) & (measured <= hi) & (cosine > 0.1)
    if good.sum() < 24:
        return None
    measured, c, w = measured[good], cosine[good], weights[good]
    w = w / w.sum()
    initial = weighted_median(depth[good] * c, w)

    def loss(clearance):
        z = clearance / c
        mu = fb / z
        sigma_z = model.depth_noise_floor_m + model.depth_noise_quadratic * z**2
        # Independent pre-disparity depth noise, disparity noise, and quantization.
        sd = np.sqrt(model.disparity_sigma_px**2 + (fb * sigma_z / z**2) ** 2 + (1 / 16) ** 2 / 12)
        log_upper = log_ndtr((hi - mu) / sd)
        log_lower = log_ndtr((lo - mu) / sd)
        log_norm = log_upper + np.log1p(-np.exp(np.minimum(-1e-12, log_lower - log_upper)))
        log_density = -0.5 * ((measured - mu) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi) - log_norm
        mixture = np.logaddexp(np.log(0.95) + log_density, np.log(0.05 / (hi - lo)))
        return float(-np.dot(w, mixture))

    bounds = (max(0.02, initial * 0.6), max(0.04, initial * 1.5))
    # The robust mixture has flat outlier plateaus. A broad bounded search can
    # miss a narrow near-range inlier basin: bracket around an explicit grid
    # including the robust initial estimate before continuous refinement.
    grid = np.unique(np.r_[np.linspace(*bounds, 41), initial])
    costs = np.array([loss(value) for value in grid])
    best = int(np.argmin(costs))
    if best in (0, len(grid) - 1):
        return None
    fit = minimize_scalar(
        loss,
        bounds=(grid[best - 1], grid[best + 1]),
        method="bounded",
        options={"xatol": 1e-6},
    )
    if not fit.success or min(fit.x - bounds[0], bounds[1] - fit.x) < 1e-4:
        return None
    return float(fit.x)


class RefinedSurfaceEstimator:
    def __init__(
        self, *, mode="balanced", stereo_noise=None, sigma_m=0.003, max_surface_slope=None, grid_size=10
    ):
        if mode not in ("balanced", "sensor", "partition"):
            raise ValueError("Unknown refinement mode")
        if not np.isfinite(sigma_m) or sigma_m <= 0 or grid_size < 3:
            raise ValueError("Invalid noise or grid")
        if max_surface_slope is not None and (not np.isfinite(max_surface_slope) or max_surface_slope <= 0):
            raise ValueError("Slope bound must be positive and physically justified")
        self.mode, self.stereo_noise = mode, stereo_noise
        self.sigma_m, self.max_surface_slope, self.grid_size = sigma_m, max_surface_slope, grid_size

    def estimate(self, rgb, raw, prediction, k, pose, bottom, area_xy, radii, *, pose_valid=True):
        out = {
            "accepted": False,
            "candidate_available": False,
            "level_m": None,
            "statistics": None,
            "quality_flags": [],
            "mode": self.mode,
            "requires_independent_verification": True,
        }
        area_xy, radii = np.asarray(area_xy, float), np.asarray(radii, float)
        if area_xy.ndim != 2 or area_xy.shape[1] != 2 or not len(area_xy) or not np.isfinite(area_xy).all():
            raise ValueError("Finite uniform footprint required")
        if radii.shape != (2,) or not np.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError("Positive footprint radii required")
        if not pose_valid or not np.isfinite(pose).all():
            out["quality_flags"] = ["invalid_pose"]
            return out
        data, reason = support(raw, prediction["mask"], k, pose, radii)
        if data is None:
            out["quality_flags"] = [reason]
            return out
        points, pixels, z, weights, _ = data
        height = weighted_median(points[:, 2] - bottom, weights)
        if self.mode == "sensor" and self.stereo_noise is not None:
            camera = np.column_stack(
                ((pixels[:, 0] - k[0, 2]) / k[0, 0], (pixels[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(z)))
            )
            cosine = -(camera @ pose[:3, :3].T)[:, 2]
            clearance = censored_clearance(z, cosine, weights, k[0, 0], self.stereo_noise)
            if clearance is None:
                out["quality_flags"] = ["sensor_likelihood_unobservable"]
                return out
            height = float(pose[2, 3] - clearance - bottom)
            out["quality_flags"].append("sensor_model_requires_device_calibration")
        elif self.mode == "sensor":
            out["quality_flags"].append("no_stereo_model_balanced_height_only")
        stats = area_statistics(np.full(len(area_xy), height))
        if self.mode == "partition":
            return self._partition(out, points, bottom, area_xy, radii)
        out.update(candidate_available=True, statistics=stats, level_m=height)
        out["quality_flags"].append("systematic_echo_not_independently_verified")
        return out

    def _partition(self, out, points, bottom, area, radii):
        n = self.grid_size
        index = np.floor((points[:, :2] / radii + 1) * n / 2).astype(int)
        inside = ((index >= 0) & (index < n)).all(axis=1)
        points, index = points[inside], index[inside]
        anchors, errors = [], []
        slope = self.max_surface_slope
        for cell in np.unique(index, axis=0):
            q = points[(index == cell).all(axis=1)]
            if len(q) < 4:
                continue
            anchor = np.median(q, axis=0)
            spread = max(self.sigma_m, 1.4826 * np.median(abs(q[:, 2] - anchor[2])))
            errors.append(3 * spread + (slope or 0) * np.max(np.linalg.norm(q[:, :2] - anchor[:2], axis=1)))
            anchors.append(anchor)
        if len(anchors) < 4:
            out["quality_flags"] = ["insufficient_observed_surface_cells"]
            return out
        a = np.asarray(anchors)
        d = np.linalg.norm(area[:, None, :] - a[None, :, :2], axis=2)
        nearest = np.argsort(d, axis=1)[:, : min(4, len(a))]
        dd = np.take_along_axis(d, nearest, axis=1)
        w = 1 / np.maximum(dd, 1e-5) ** 2
        heights = a[:, 2] - bottom
        fitted = np.sum(w * heights[nearest], axis=1) / w.sum(axis=1)
        # Coverage means footprint cells with current data, not trustworthy output.
        area_cells = np.floor((area / radii + 1) * n / 2).astype(int).clip(0, n - 1)
        anchor_cells = np.floor((a[:, :2] / radii + 1) * n / 2).astype(int).clip(0, n - 1)
        observed = np.isin(
            area_cells[:, 0] + n * area_cells[:, 1], anchor_cells[:, 0] + n * anchor_cells[:, 1]
        )
        out.update(
            candidate_available=True,
            level_m=float(fitted.mean()),
            statistics=area_statistics(fitted),
            observed_statistics=area_statistics(fitted[observed]),
            observed_area_fraction=float(observed.mean()),
            support_cells=len(a),
            interval_is_statistically_calibrated=False,
            statistics_intervals=None,
        )
        out["quality_flags"] = ["missing_area_is_inferred_not_measured", "no_temporal_wave_shape_reuse"]
        if slope is None:
            out["quality_flags"].append("global_bounds_require_validated_slope_prior")
            return out
        lower = np.max(heights[None, :] - np.asarray(errors)[None, :] - slope * d, axis=1)
        upper = np.min(heights[None, :] + np.asarray(errors)[None, :] + slope * d, axis=1)
        if (lower > upper).any():
            out["quality_flags"].append("anchor_or_slope_bound_inconsistent")
            return out
        low, high = area_statistics(lower), area_statistics(upper)
        names = ("min_depth_m", "max_depth_m", "mean_depth_m", "p05_depth_m", "p95_depth_m")
        out["statistics_intervals"] = {key: [low[key], high[key]] for key in names}
        out["quality_flags"].append("bounds_conditional_on_slope_and_anchor_error")
        return out
