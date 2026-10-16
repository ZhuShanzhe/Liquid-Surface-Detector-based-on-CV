"""Research-only consumers of one empty-vessel geometry bundle.

Scene choice and model mask are explicit inputs. No scene is auto-approved for
industrial reporting; this module returns diagnostic candidates only.
"""
from dataclasses import replace

import numpy as np

from .empty_geometry_v41 import build_geometry, height_calibration, make_refraction_meter
from .rim_geometry_v43 import build_rim_geometry
from .ordinary_surface_v35 import interior_ray_roi
from .surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel
from .liquid_height_metrics import world_points


SURFACE_SCENES = frozenset({
    'low_light', 'glare', 'floating_objects', 'uneven_surface', 'depth_failure',
})


def build_scene_geometry(empty_depth_m, intrinsics, bottom_spec, *, rim_spec=None):
    """Prefer observed dry bottom; explicitly fall back to measured dry rim.

    Only known observability failures trigger the additional-input route.
    Contradictory installation metadata and metric checks remain hard errors.
    """
    try:
        return build_geometry(empty_depth_m, intrinsics, bottom_spec)
    except ValueError as exc:
        fallback_causes = {
            'bottom_depth_support_insufficient',
            'bottom_depth_spatial_coverage_insufficient',
            'bottom_outline_too_small',
            'bottom_outline_must_be_convex_and_complete',
            'metric_span_too_short_in_image',
            'independent_dimension_check_failed',
        }
        if str(exc) not in fallback_causes or rim_spec is None:
            raise
        bundle = build_rim_geometry(empty_depth_m, intrinsics, rim_spec)
        bundle['quality']['bottom_route_failure'] = str(exc)
        return bundle


def vessel_ray_mask(cal, rim, shape, *, fraction=.85):
    roi = interior_ray_roi(cal, rim, shape, fraction)
    if roi.sum() < 64:
        opening = replace(cal, bottom_z_m=rim - 1e-5)
        roi = interior_ray_roi(opening, rim, shape, fraction)
        return roi, 'opening_only_roi_due_to_small_bottom_projection'
    return roi, None


def surface_candidate(bundle, rgb_bgr, depth_m, predicted_mask, *, scene):
    """Reuse fitted bottom/pose/footprint with a selected existing surface route.

    RGB enhancement, segmentation and temporal recovery remain independent
    upstream steps. A numeric candidate is never promoted to an accepted result.
    """
    if scene not in SURFACE_SCENES:
        raise ValueError('explicit_supported_surface_scene_required')
    cal = height_calibration(bundle)
    raw = np.asarray(depth_m)
    mask = np.asarray(predicted_mask, bool)
    rgb = np.asarray(rgb_bgr)
    if raw.ndim != 2 or mask.shape != raw.shape or rgb.shape != (*raw.shape, 3):
        raise ValueError('aligned_rgb_depth_and_predicted_mask_required')
    if list(bundle['image_size']) != [raw.shape[1], raw.shape[0]]:
        raise ValueError('image_shape_changed_recalibration_required')
    sensor = bundle['sensor_family']
    roi, roi_warning = vessel_ray_mask(cal, bundle['rim_z_m'], raw.shape)
    spatial_mask = mask & roi
    raw_valid_mask = spatial_mask & np.isfinite(raw) & (raw > 0)
    if raw_valid_mask.sum() <= .05 * max(spatial_mask.sum(), 1):
        return {'scene': scene, 'geometry_source': bundle['source'],
                'measurement_route': 'surface_' + ('partition' if scene == 'uneven_surface' else 'sensor'),
                'statistics': None, 'numeric_candidate_m': None,
                'reasons': ['unsupported_95_100_percent_depth_failure'],
                'input_support': {'predicted_mask_pixels': int(mask.sum()),
                                  'geometry_consistent_pixels': 0,
                                  'ray_roi_pixels': int(roi.sum())},
                'accepted': False, 'production_qualified': False,
                'research_only': True}
    xyz = world_points(raw, cal)
    normalized = xyz[..., :2] / np.asarray(cal.radii_m)
    inside = (abs(normalized) ** cal.footprint_exponent).sum(axis=-1) <= 1.05 ** cal.footprint_exponent
    margin = max(.03, .02 * bundle['rim_z_m'])
    plausible = (xyz[..., 2] >= cal.bottom_z_m - margin) & (xyz[..., 2] <= bundle['rim_z_m'] + margin)
    selected = raw_valid_mask & inside & plausible
    noise = None if sensor == 'tof' else StereoNoiseModel.simulation_proxy(sensor)
    mode = 'partition' if scene == 'uneven_surface' else 'sensor'
    result = RefinedSurfaceEstimator(mode=mode, stereo_noise=noise).estimate(
        rgb, raw, {'mask': selected}, cal.intrinsics, cal.camera_to_world_cv,
        cal.bottom_z_m, cal.area_grid(), cal.radii_m)
    reasons = list(result['quality_flags'])
    if roi_warning:
        reasons.append(roi_warning)
    return {
        'scene': scene, 'geometry_source': bundle['source'],
        'measurement_route': 'surface_' + mode,
        'statistics': result['statistics'],
        'numeric_candidate_m': result['level_m'],
        'reasons': reasons,
        'input_support': {'predicted_mask_pixels': int(mask.sum()),
                          'geometry_consistent_pixels': int(selected.sum()),
                          'ray_roi_pixels': int(roi.sum())},
        'accepted': False, 'production_qualified': False,
        'research_only': True,
    }


def refraction_candidate_meter(bundle, right_intrinsics, masks, prior_images,
                               known_heights, condition, method='baseline_v32',
                               sr_backend=None, *, left_intrinsics):
    """Keep empty geometry and the separate 1-empty+4-known-level calibration.

    Valid only for the existing near-vertical rectified stereo image solver; ToF,
    glass wall refraction and multilayer paths require different solvers.
    """
    if condition not in ('transparent', 'translucent_mild'):
        raise ValueError('homogeneous_refraction_scope_required')
    if bundle['sensor_family'] not in ('active_stereo', 'structured_light'):
        raise ValueError('left_right_image_sensor_required')
    return make_refraction_meter(
        bundle, right_intrinsics, masks, prior_images, known_heights,
        condition, method, sr_backend, left_intrinsics=left_intrinsics)
