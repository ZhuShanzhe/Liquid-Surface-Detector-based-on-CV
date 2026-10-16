"""Merged V11 two-height research route for opaque uneven surfaces."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .ordinary_surface_v35 import interior_ray_roi
from .supported_scene_meter import FormalSceneMeter
from .uneven_surface_repair import repair_measurement


def _top_envelope(meter: FormalSceneMeter, depth: np.ndarray):
    cal = height_calibration(meter.bundle)
    xyz = world_points(depth, cal)
    height = xyz[..., 2] - cal.bottom_z_m
    valid = meter.ordinary_container_roi & (depth > 0)
    valid &= np.isfinite(depth)
    source = "static_empty_roi"
    if not valid.any():
        rim = float(meter.bundle["rim_z_m"])
        opening = interior_ray_roi(replace(cal, bottom_z_m=rim - 1e-5), rim, depth.shape, 0.90)
        scaled = xyz[..., :2] / np.asarray(cal.radii_m)
        inside = abs(scaled) ** cal.footprint_exponent
        inside = inside.sum(-1)
        valid = opening & (inside <= 1.05**cal.footprint_exponent)
        valid &= np.isfinite(depth) & (depth > 0)
        source = "dynamic_geometry_rescue"
    rim = float(meter.bundle["rim_z_m"])
    valid &= (height >= -0.03) & (height <= rim + 0.03)
    count = int(valid.sum())
    if count < 9:
        return None, count, source
    local = median_filter(np.where(valid, height, -np.inf), size=3, mode="constant", cval=-np.inf)
    support = uniform_filter(valid.astype(float), size=3, mode="constant") * 9
    peaks = local[(support >= 5) & np.isfinite(local)]
    if len(peaks) == 0:
        return None, count, source
    return float(peaks.max()), count, source


def _dry_paired_shallow(
    base: FormalSceneMeter,
    dry: np.ndarray,
    wet: np.ndarray,
    preliminary: float | None,
) -> dict:
    out = {
        "applied": False,
        "height_m": preliminary,
        "paired_height_m": None,
        "support_pixels": 0,
        "reason": "preliminary_outside_shallow_scope",
    }
    if preliminary is None or not -0.1 <= preliminary <= 1.0:
        return out
    cal = height_calibration(base.bundle)
    dry_h = world_points(dry, cal)[..., 2] - cal.bottom_z_m
    wet_h = world_points(wet, cal)[..., 2] - cal.bottom_z_m
    valid = base.ordinary_container_roi.copy()
    valid &= np.isfinite(dry) & (dry > 0)
    valid &= np.isfinite(wet) & (wet > 0)
    valid &= np.abs(dry_h) < 0.03
    count = int(valid.sum())
    out["support_pixels"] = count
    if count < 24:
        out["reason"] = "insufficient_matched_bottom_pixels"
        return out
    candidate = float(np.median((wet_h - dry_h)[valid]))
    out["paired_height_m"] = candidate
    if not -0.1 <= candidate <= 1.0:
        out["reason"] = "paired_candidate_outside_shallow_scope"
    elif abs(candidate - preliminary) > 0.20:
        out["reason"] = "paired_candidate_disagrees_with_primary"
    else:
        out["applied"] = True
        out["height_m"] = candidate
        out["reason"] = "matched_dry_wet_support"
    return out


class UnevenSurfaceMeter:
    """Bottom-referenced representative and visible peak heights.

    V11 uses dry-paired shallow evidence and supported peak repair by default.
    The legacy peak route is available only with evidence_repair_enabled=False
    for ablation. This is research-only, not field-qualified: the representative
    value is a robust-return proxy for area mean, and an occluded apex can be
    missed by the supported local peak.
    """

    def __init__(
        self,
        base: FormalSceneMeter,
        dry_depth_m: np.ndarray | None = None,
    ):
        if base.bundle is None or base.dry_reference is None:
            raise ValueError("empty_geometry_reference_required")
        self.base = base
        self.dry_depth_m = np.asarray(dry_depth_m) if dry_depth_m is not None else None
        if self.dry_depth_m is not None and (self.dry_depth_m.shape != base.dry_reference.image_shape):
            raise ValueError("aligned_dry_depth_reference_required")

    @classmethod
    def from_empty_geometry(cls, bundle: dict, dry_depth_m: np.ndarray, **meter_kwargs) -> UnevenSurfaceMeter:
        """Build the current uneven-surface route from a matched empty frame."""
        base = FormalSceneMeter.from_empty_geometry(bundle, dry_depth_m, **meter_kwargs)
        return cls(base, dry_depth_m)

    def measure(
        self,
        depth_m: np.ndarray,
        *,
        recent_depth_frames: list[np.ndarray] | None = None,
        stationary_confirmed: bool = False,
        dry_paired_shallow_enabled: bool = True,
        evidence_repair_enabled: bool = True,
    ) -> dict:
        depth = np.asarray(depth_m)
        liquid_input = depth
        temporal_count = 1
        if recent_depth_frames and stationary_confirmed:
            recent = [np.asarray(x) for x in recent_depth_frames[-3:]]
            if any(x.shape != depth.shape for x in recent):
                raise ValueError("aligned_recent_depth_frames_required")
            stack = np.stack([*recent, depth]).astype(float)
            stack = np.where(np.isfinite(stack) & (stack > 0), stack, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fused = np.nanmedian(stack, axis=0)
            liquid_input = np.nan_to_num(fused).astype(depth.dtype)
            temporal_count = len(stack)
        representative = self.base.measure("ordinary_diffuse", depth_m=liquid_input)
        peak, count, source = _top_envelope(self.base, depth)
        liquid = representative["liquid_depth_m"]
        paired = None
        if dry_paired_shallow_enabled:
            if self.dry_depth_m is None:
                raise ValueError("dry_depth_required_for_paired_shallow")
            paired = _dry_paired_shallow(self.base, self.dry_depth_m, depth, liquid)
            if paired["applied"]:
                liquid = paired["height_m"]
        repair = None
        if evidence_repair_enabled:
            if self.dry_depth_m is None:
                raise ValueError("dry_depth_required_for_evidence_repair")
            liquid, peak, repair = repair_measurement(
                self.base,
                self.dry_depth_m,
                depth,
                liquid,
                representative["diagnostics"]["value_source"],
                peak,
            )
            if repair["peak_reason"] != "original_peak_preserved":
                count = repair["peak_selected_returns"]
                source = "opening_roi_supported_local_mean"
        route = representative["route"]
        if paired is not None and paired["applied"]:
            route += "+dry_paired_shallow_v1"
        if repair is not None:
            route += "+uneven_echo_evidence_v3"
        flags = []
        if repair is not None and repair["echo_layer_conflict"]:
            flags.append("echo_layers_disagree_numeric_retained")
        if repair is not None and repair["peak_sparse_fallback"]:
            flags.append("highest_sparse_support_low_observability")
        if peak is None:
            flags.append("highest_support_unavailable")
        elif liquid is not None and peak + 0.005 < liquid:
            flags.append("height_order_inconsistent")
        if count < 24:
            flags.append("few_highest_roi_returns")
        evidence_warnings = []
        if peak is None:
            evidence_warnings.append("missing_peak")
        if "height_order_inconsistent" in flags:
            evidence_warnings.append("mean_peak_order_conflict")
        if repair is not None:
            if repair["echo_layer_conflict"]:
                evidence_warnings.append("echo_layers_disagree")
            if repair["peak_sparse_fallback"] or repair["peak_wide_fallback"]:
                evidence_warnings.append("peak_requires_fallback")
            if repair["peak_selected_returns"] < 200:
                evidence_warnings.append("limited_local_peak_support")
            pair_se = repair["paired_tile_se_m"]
            if pair_se is not None and pair_se > 0.040:
                evidence_warnings.append("dry_pair_spatial_uncertainty")
        evidence_status = ("missing" if peak is None else "unassessed" if repair is None
                           else "weak" if evidence_warnings else "supported")
        value_source = representative["diagnostics"]["value_source"]
        if repair is not None and repair["liquid_reason"] != "current_preserved":
            value_source = repair["liquid_reason"]
        return {
            "scene": "opaque_uneven_surface",
            "software_route_status": "merged_uneven_v11_research",
            "liquid_surface_height_m": liquid,
            "highest_visible_height_m": peak,
            "liquid_numeric": representative["numeric_output"],
            "highest_numeric": peak is not None,
            "highest_evidence_status": evidence_status,
            "highest_evidence_warnings": evidence_warnings,
            "highest_confidence_calibrated": False,
            "highest_roi_points": count,
            "highest_roi_source": source,
            "liquid_route": route,
            "liquid_value_source": value_source,
            "liquid_quality_flags": representative["diagnostics"]["quality_flags"],
            "liquid_selected_pixels": representative["diagnostics"]["selected_pixels"],
            "temporal_liquid_frames_used": temporal_count,
            "stationary_confirmed": bool(stationary_confirmed),
            "quality_flags": flags,
            "field_metrology_qualified": False,
            "dry_paired_shallow": paired,
            "uneven_evidence_repair": repair,
            "research_only": True,
        }
