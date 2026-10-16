"""Quantized-disparity ambiguity detection and censored likelihood fit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


def _mode_bin(disparity: np.ndarray, step_px: float) -> tuple[int, float]:
    bins = np.rint(disparity / step_px).astype(int)
    values, counts = np.unique(bins, return_counts=True)
    index = int(np.argmax(counts))
    return int(values[index]), float(counts[index] / len(bins))


def fit_quantized_disparity(
    raw_depth_m: np.ndarray, height_returns_m: np.ndarray, *,
    camera_z_m: float, bottom_z_m: float, fx_px: float, baseline_m: float,
    disparity_step_px: float, disparity_noise_px: float,
    max_return_depth_m: float, coarse_height_m: float,
    upper_height_m: float, radius_m: float,
) -> dict:
    """Fit a horizontal surface from quantized SDK returns only."""
    depth = np.asarray(raw_depth_m, dtype=float)
    height = np.asarray(height_returns_m, dtype=float)
    result = {"applied": False, "ambiguous": False,
              "height_m": None, "reason": "insufficient_profile_or_support"}
    if (depth.shape != height.shape or depth.ndim != 1 or len(depth) < 24 or
            not np.all(np.isfinite(depth)) or np.any(depth <= 0) or
            not np.all(np.isfinite(height)) or
            min(camera_z_m, fx_px, baseline_m, disparity_step_px,
                disparity_noise_px, max_return_depth_m, radius_m) <= 0):
        return result
    scale = fx_px * baseline_m
    disparity = scale / depth
    bins = np.rint(disparity / disparity_step_px)
    on_grid = np.abs(disparity - bins * disparity_step_px) < min(
        .001, .04 * disparity_step_px)
    grid_fraction = float(np.mean(on_grid))
    result["grid_fraction"] = grid_fraction
    low_band = on_grid & (np.abs(height - coarse_height_m) <= radius_m)
    high_band = on_grid & (np.abs(height - upper_height_m) <= radius_m)
    if grid_fraction < .70 or min(low_band.sum(), high_band.sum()) < 24:
        result["reason"] = "not_quantized_adjacent_layer_evidence"
        return result
    low_bin, low_share = _mode_bin(disparity[low_band], disparity_step_px)
    high_bin, high_share = _mode_bin(disparity[high_band], disparity_step_px)
    result.update(low_bin=low_bin, high_bin=high_bin,
                  low_mode_fraction=low_share, high_mode_fraction=high_share,
                  low_support=int(low_band.sum()), high_support=int(high_band.sum()))
    if abs(low_bin - high_bin) != 1 or min(low_share, high_share) < .80:
        result["reason"] = "not_quantized_adjacent_layer_evidence"
        return result
    result["ambiguous"] = True
    result["reason"] = "adjacent_disparity_bins"
    use = on_grid & (depth <= max_return_depth_m)
    if use.sum() < 64:
        result["reason"] = "too_few_quantized_returns"
        return result
    slope = (height[use] + bottom_z_m - camera_z_m) / depth[use]
    quantized = bins[use]
    lower = (quantized - .5) * disparity_step_px
    upper = (quantized + .5) * disparity_step_px
    first_valid = np.ceil(scale / max_return_depth_m / disparity_step_px)
    survival_cut = (first_valid - .5) * disparity_step_px
    low_bound = min(coarse_height_m, upper_height_m) - 1.
    high_bound = max(coarse_height_m, upper_height_m) + 1.
    if low_bound + bottom_z_m >= camera_z_m:
        result["reason"] = "invalid_camera_plane_geometry"
        return result

    def objective(candidate_height_m: float) -> float:
        denominator = candidate_height_m + bottom_z_m - camera_z_m
        if denominator >= -.01:
            return float("inf")
        expected = scale * slope / denominator
        bin_probability = (ndtr((upper - expected) / disparity_noise_px) -
                           ndtr((lower - expected) / disparity_noise_px))
        survive = 1. - ndtr((survival_cut - expected) / disparity_noise_px)
        terms = (np.log(np.maximum(bin_probability, 1e-15)) -
                 np.log(np.maximum(survive, 1e-15)))
        return float(-np.mean(terms))

    fit = minimize_scalar(objective, bounds=(low_bound, high_bound),
                          method="bounded", options={"xatol": 1e-5})
    result["fit_returns"] = int(use.sum())
    result["fit_objective_per_return"] = float(fit.fun)
    if (not fit.success or not np.isfinite(fit.x) or
            fit.x <= low_bound + .01 or fit.x >= high_bound - .01):
        result["reason"] = "quantized_fit_failed_or_boundary"
        return result
    result.update(applied=True, height_m=float(fit.x),
                  reason="censored_quantized_disparity_likelihood")
    return result
