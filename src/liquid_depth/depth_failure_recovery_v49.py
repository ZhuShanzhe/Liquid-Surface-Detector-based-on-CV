"""Opt-in research fallback for a missing depth-failure-route mask.

Uses only an installed dry RGB-D reference, current RGB-D image, and the
segmentation probability map. No simulator truth or height clipping enters the
route. Agreement is a *consistency check*, not independent ground truth.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .ordinary_surface_v35 import interior_ray_roi
from .scene_route_policy_v45 import surface_candidate


@dataclass(frozen=True)
class DryCenterReference:
    geometry_signature: str
    image_shape: tuple[int, int]
    central_roi: np.ndarray
    dry_height_median_m: float | None
    dry_support_pixels: int


def _signature(bundle):
    cal = height_calibration(bundle)
    vector = np.concatenate((
        cal.intrinsics.ravel(), cal.camera_to_world_cv.ravel(),
        np.asarray(cal.radii_m).ravel(),
        [cal.bottom_z_m, float(bundle['rim_z_m'])],
        np.asarray(bundle['image_size'], dtype=float),
    )).astype('<f8')
    return sha256(vector.tobytes() + bundle['sensor_family'].encode()).hexdigest()


def _central_median(raw, roi, cal):
    valid = roi & np.isfinite(raw) & (raw > 0)
    count = int(valid.sum())
    if count < 24:
        return None, count
    heights = world_points(raw, cal)[..., 2] - cal.bottom_z_m
    return float(np.median(heights[valid])), count


def prepare_dry_center_reference(bundle, dry_depth_m):
    cal = height_calibration(bundle)
    dry = np.asarray(dry_depth_m)
    if dry.ndim != 2 or [dry.shape[1], dry.shape[0]] != list(bundle['image_size']):
        raise ValueError('dry_image_shape_changed_recalibration_required')
    roi = interior_ray_roi(cal, float(bundle['rim_z_m']), dry.shape, .65)
    median, count = _central_median(dry, roi, cal)
    return DryCenterReference(_signature(bundle), dry.shape, roi, median, count)


def recover_depth_failure(bundle, rgb_bgr, depth_m, mask_probability, reference,
                          *, baseline=None):
    raw = np.asarray(depth_m)
    probability = np.asarray(mask_probability)
    if raw.ndim != 2 or probability.shape != raw.shape or reference.image_shape != raw.shape:
        raise ValueError('aligned_probability_and_depth_required')
    if _signature(bundle) != reference.geometry_signature:
        raise ValueError('stale_dry_reference_recalibration_required')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('finite_probability_in_unit_interval_required')
    base = baseline if baseline is not None else surface_candidate(
        bundle, rgb_bgr, raw, probability >= .5, scene='depth_failure')
    if base['numeric_candidate_m'] is not None:
        return {**base, 'recovery_status':'baseline_unchanged',
                'baseline_reasons':list(base['reasons'])}
    reasons = list(base['reasons'])
    if reference.dry_height_median_m is None:
        return {**base, 'recovery_status':'dry_center_unobservable',
                'baseline_reasons':reasons}
    low = surface_candidate(bundle, rgb_bgr, raw, probability >= .01,
                            scene='depth_failure')
    candidate = low['numeric_candidate_m']
    if candidate is None:
        return {**base, 'recovery_status':'low_probability_candidate_unobservable',
                'baseline_reasons':reasons}
    wet_median, wet_count = _central_median(raw, reference.central_roi,
                                            height_calibration(bundle))
    if wet_median is None:
        return {**base, 'recovery_status':'wet_center_unobservable',
                'baseline_reasons':reasons}
    center_displacement = wet_median - reference.dry_height_median_m
    disagreement_m = abs(float(candidate) - center_displacement)
    if disagreement_m > .02:
        return {**base, 'recovery_status':'candidate_center_disagreement',
                'baseline_reasons':reasons, 'candidate_center_disagreement_m':disagreement_m,
                'wet_center_support_pixels':wet_count}
    # ToF has metric per-pixel ranges rather than quantized disparity. Once the
    # weak-mask plane agrees with the paired dry displacement, report that
    # measured displacement. It is neither snapped to zero nor clipped.
    tof_center = bundle['sensor_family'] == 'tof'
    reported = center_displacement if tof_center else float(candidate)
    return {**low,
            'numeric_candidate_m':float(reported),
            'measurement_route':('depth_failure_tof_paired_center_v49' if tof_center
                                 else 'depth_failure_low_probability_dry_consistency_v49'),
            'recovery_status':'low_probability_candidate_consistent',
            'baseline_reasons':reasons,
            'candidate_center_disagreement_m':disagreement_m,
            'weak_mask_candidate_m':float(candidate),
            'dry_paired_center_displacement_m':float(center_displacement),
            'wet_center_support_pixels':wet_count,
            'dry_center_support_pixels':reference.dry_support_pixels,
            'accepted':False, 'production_qualified':False, 'research_only':True}
