"""Evidence-based repair experiments for opaque uneven liquid surfaces."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import uniform_filter

from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .ordinary_surface_v35 import interior_ray_roi


def prepare_evidence(base, dry, depth):
    cal = height_calibration(base.bundle)
    xyz = world_points(depth, cal)
    wet_h = xyz[..., 2] - cal.bottom_z_m
    dry_h = world_points(dry, cal)[..., 2] - cal.bottom_z_m
    finite = np.isfinite(depth) & (depth > 0)
    paired = base.ordinary_container_roi & finite
    paired &= np.isfinite(dry) & (dry > 0) & (np.abs(dry_h) < 0.03)
    diff = wet_h - dry_h
    vals = diff[paired]
    candidate = float(np.median(vals)) if len(vals) >= 24 else None
    tiles = []
    for y in range(0, depth.shape[0], 20):
        for x in range(0, depth.shape[1], 20):
            take = paired[y : y + 20, x : x + 20]
            values = diff[y : y + 20, x : x + 20][take]
            if len(values) >= 4:
                tiles.append(float(np.median(values)))
    tile = np.asarray(tiles)
    se = None
    if len(tile) >= 4:
        mad = float(np.median(np.abs(tile - np.median(tile))))
        se = 1.4826 * mad / np.sqrt(len(tile))
    rim = float(base.bundle["rim_z_m"])
    opening = interior_ray_roi(replace(cal, bottom_z_m=rim - 1e-5), rim, depth.shape, 0.90)
    valid = opening & finite
    # Select plausible returns, without clamping a reported estimate.
    valid &= (wet_h >= -0.03) & (wet_h <= rim + 0.03)
    raw_valid = base.ordinary_container_roi & finite
    quantile = 50 if base.bundle["sensor_family"] == "tof" else 58
    raw_q = float(np.percentile(wet_h[raw_valid], quantile)) if raw_valid.any() else None
    return {
        "cal": cal,
        "height": wet_h,
        "valid": valid,
        "paired_mask": paired,
        "paired_diff": diff,
        "paired_height_m": candidate,
        "paired_support": len(vals),
        "paired_tiles": len(tile),
        "paired_tile_se_m": se,
        "raw_quantile_m": raw_q,
    }


def repaired_liquid(evidence, current, value_source, *, blend=False):
    value = current
    c = evidence["paired_height_m"]
    q = evidence["raw_quantile_m"]
    se = evidence["paired_tile_se_m"]
    reason = "current_preserved"
    if current is None or c is None or q is None or se is None:
        return value, reason
    spatial = evidence["paired_tiles"] >= 4
    camera_height = evidence["cal"].camera_to_world_cv[2, 3] - evidence["cal"].bottom_z_m
    if (
        "dry_anchored_layer" in value_source
        and abs(current) < 0.15
        and c - 3.0 * se > max(0.25, 0.15 * camera_height)
        and q > 0.25
        and spatial
        and abs(c - q) <= max(0.25, 0.05 * abs(c))
    ):
        return q, "dry_layer_contradicted_by_two_wet_candidates"
    # Stronger spatial evidence can resolve an arbitrary 0.20 m boundary.
    if -0.1 <= c <= 1.05 and -0.1 <= current <= 1.15 and abs(c - current) <= 0.35 and spatial and se <= 0.035:
        if blend and c > 0.1:
            value = 0.5 * (c + q)
            reason = "paired_and_raw_quantile_average"
        elif abs(c - current) > 0.20 or current > 1.0:
            value = c
            reason = "spatially_consistent_shallow_override"
    return value, reason


def peak_candidates(evidence, liquid, *, windows=(5, 9, 15, 25), minimum_density=0.15, radius_m=None):
    height = evidence["height"]
    valid = evidence["valid"]
    result = {}
    if liquid is None:
        return result
    radius = 0.10 + 0.01 * abs(liquid) if radius_m is None else radius_m
    selected = valid & (np.abs(height - liquid) <= radius)
    if selected.sum() < 6:
        return result
    values = np.where(selected, height, 0.0)
    for size in windows:
        density = uniform_filter(selected.astype(float), size=size, mode="constant")
        total = uniform_filter(values, size=size, mode="constant")
        local = np.divide(total, density, out=np.zeros_like(total), where=density > 0)
        supported = (density * size * size >= 6) & (density >= minimum_density)
        peaks = local[supported]
        if not len(peaks):
            continue
        result[f"local{size}_max"] = float(np.max(peaks))
        result[f"local{size}_q99"] = float(np.percentile(peaks, 99))
        result[f"local{size}_q995"] = float(np.percentile(peaks, 99.5))
    result["selected_q99"] = float(np.percentile(height[selected], 99))
    result["selected_count"] = int(selected.sum())
    return result


def paired_near_bottom_peak(evidence):
    mask = evidence["paired_mask"]
    if mask.sum() < 100:
        return None
    density = uniform_filter(mask.astype(float), size=9, mode="constant")
    values = uniform_filter(np.where(mask, evidence["paired_diff"], 0.0), size=9, mode="constant")
    local = np.divide(values, density, out=np.zeros_like(values), where=density > 0)
    supported = (density * 81 >= 6) & (density >= 0.15)
    if not supported.any():
        return None
    return float(np.percentile(local[supported], 90))


def repair_measurement(base, dry, depth, liquid, value_source, original_peak):
    evidence = prepare_evidence(base, dry, depth)
    repaired, liquid_reason = repaired_liquid(evidence, liquid, value_source)
    peaks = peak_candidates(evidence, repaired, windows=(9,))
    candidate = peaks.get("local9_max")
    sparse_fallback = False
    if candidate is None and (
        original_peak is None or (repaired is not None and original_peak < repaired - 0.005)
    ):
        peaks = peak_candidates(evidence, repaired, windows=(15,), minimum_density=0.03)
        candidate = peaks.get("local15_max")
        sparse_fallback = candidate is not None
    wide_fallback = False
    if candidate is None and repaired is not None and (
        original_peak is None or original_peak < repaired - 0.10
    ):
        paired_se = evidence["paired_tile_se_m"] or 0.0
        wider = peak_candidates(evidence, repaired, windows=(9,),
                                radius_m=0.25 + min(paired_se, 0.12))
        candidate = wider.get("local9_q99")
        if candidate is not None:
            peaks = wider
            wide_fallback = True
    peak = original_peak
    peak_reason = "original_peak_preserved"
    if candidate is not None:
        if original_peak is None:
            peak = candidate
            peak_reason = "supported_mean_rescues_missing_peak"
        elif repaired is not None and original_peak < repaired - 0.005:
            peak = candidate
            peak_reason = "supported_mean_resolves_low_echo_layer"
        elif original_peak - candidate > 0.02:
            peak = candidate
            peak_reason = "supported_mean_suppresses_isolated_high_echo"
    paired_peak = None
    paired = evidence["paired_height_m"]
    se = evidence["paired_tile_se_m"]
    raw = evidence["raw_quantile_m"]
    if (
        repaired is not None and paired is not None and se is not None and raw is not None
        and abs(repaired) <= 0.025 and abs(paired) <= 0.025
        and (abs(raw) <= 0.030 or (abs(raw) <= 0.050
        and abs(repaired) <= 0.007 and abs(paired) <= 0.007))
        and se <= 0.010 and evidence["paired_tiles"] >= 4
    ):
        paired_peak = paired_near_bottom_peak(evidence)
        if paired_peak is not None and (peak is None or peak > paired_peak + 0.010):
            camera_height = evidence["cal"].camera_to_world_cv[2, 3] - evidence["cal"].bottom_z_m
            if camera_height < 1.0 and candidate is not None and candidate - paired_peak > 0.030:
                peak = candidate
                peak_reason = "near_bottom_coherent_peak_preserved"
            else:
                peak = paired_peak
                peak_reason = "paired_near_bottom_supported_peak"
    common_offset = None
    if (
        peak is not None and repaired is not None and paired is not None and se is not None
        and 1.0 <= repaired <= 3.0 and 0.030 <= repaired - paired <= 0.200
        and se <= 0.040 and evidence["paired_tiles"] >= 8
        and evidence["paired_support"] >= 100
    ):
        common_offset = (paired - repaired) / (1.0 + (se / 0.030) ** 2)
        peak += common_offset
        peak_reason += "+paired_common_mode_offset"
    diag = {
        key: evidence[key]
        for key in ("paired_height_m", "paired_support", "paired_tiles", "paired_tile_se_m", "raw_quantile_m")
    }
    diag.update(
        original_liquid_height_m=liquid,
        original_peak_height_m=original_peak,
        liquid_reason=liquid_reason,
        peak_reason=peak_reason,
        peak_supported_candidate_m=candidate,
        peak_selected_returns=peaks.get("selected_count", 0),
        peak_sparse_fallback=sparse_fallback,
        peak_wide_fallback=wide_fallback,
        paired_near_bottom_peak_m=paired_peak,
        peak_common_mode_offset_m=common_offset,
        numerical_value_clipped=False,
    )
    paired = evidence["paired_height_m"]
    diag["echo_layer_conflict"] = bool(
        paired is not None and repaired is not None and abs(paired - repaired) > max(0.30, 0.20 * abs(paired))
    )
    return repaired, peak, diag
