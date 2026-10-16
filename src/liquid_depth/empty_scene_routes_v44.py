"""Research scene adapter with opening visibility and pointwise vessel checks.

V43's all-the-way-to-bottom ray ROI was unnecessarily restrictive for deep
vessels: a valid liquid return need only be visible through the opening and
physically lie inside the calibrated vessel at its measured depth.
"""
from dataclasses import replace

import numpy as np

from .empty_geometry_v41 import height_calibration
from .empty_scene_routes_v43 import (SURFACE_SCENES, build_scene_geometry,
                                     refraction_candidate_meter)
from .liquid_height_metrics import world_points
from .ordinary_surface_v35 import interior_ray_roi
from .surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel


def surface_candidate(bundle, rgb_bgr, depth_m, predicted_mask, *, scene):
    if scene not in SURFACE_SCENES:
        raise ValueError('explicit_supported_surface_scene_required')
    cal = height_calibration(bundle)
    raw = np.asarray(depth_m)
    rgb = np.asarray(rgb_bgr)
    mask = np.asarray(predicted_mask, bool)
    if raw.ndim != 2 or rgb.shape != (*raw.shape, 3) or mask.shape != raw.shape:
        raise ValueError('aligned_rgb_depth_and_predicted_mask_required')
    if list(bundle['image_size']) != [raw.shape[1], raw.shape[0]]:
        raise ValueError('image_shape_changed_recalibration_required')
    rim = float(bundle['rim_z_m'])
    # Through-opening visibility is an image-space check; pointwise 3-D checks
    # below are evaluated at each actual return, not at the empty bottom plane.
    opening_plane = replace(cal, bottom_z_m=rim - 1e-5)
    roi = interior_ray_roi(opening_plane, rim, raw.shape, .90)
    spatial = mask & roi
    valid = spatial & np.isfinite(raw) & (raw > 0)
    mode = 'partition' if scene == 'uneven_surface' else 'sensor'
    if valid.sum() <= .05 * max(spatial.sum(), 1):
        return {'scene': scene, 'geometry_source': bundle['source'],
                'measurement_route': 'surface_' + mode,
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
    margin = max(.03, .02 * rim)
    plausible = (xyz[..., 2] >= cal.bottom_z_m - margin) & (xyz[..., 2] <= rim + margin)
    selected = valid & inside & plausible
    sensor = bundle['sensor_family']
    noise = None if sensor == 'tof' else StereoNoiseModel.simulation_proxy(sensor)
    result = RefinedSurfaceEstimator(mode=mode, stereo_noise=noise).estimate(
        rgb, raw, {'mask': selected}, cal.intrinsics, cal.camera_to_world_cv,
        cal.bottom_z_m, cal.area_grid(), cal.radii_m)
    return {'scene': scene, 'geometry_source': bundle['source'],
            'measurement_route': 'surface_' + mode,
            'statistics': result['statistics'],
            'numeric_candidate_m': result['level_m'],
            'reasons': list(result['quality_flags']),
            'input_support': {'predicted_mask_pixels': int(mask.sum()),
                              'geometry_consistent_pixels': int(selected.sum()),
                              'ray_roi_pixels': int(roi.sum())},
            'accepted': False, 'production_qualified': False,
            'research_only': True}
