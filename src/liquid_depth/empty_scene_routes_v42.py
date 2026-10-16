"""Research-only consumers of one empty-vessel geometry bundle.

Scene choice and model mask are explicit inputs. No scene is auto-approved for
industrial reporting; this module returns diagnostic candidates only.
"""
import numpy as np

from .empty_geometry_v41 import height_calibration, make_refraction_meter
from .surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel


SURFACE_SCENES = frozenset({
    'low_light', 'glare', 'floating_objects', 'uneven_surface', 'depth_failure',
})


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
    noise = None if sensor == 'tof' else StereoNoiseModel.simulation_proxy(sensor)
    mode = 'partition' if scene == 'uneven_surface' else 'sensor'
    result = RefinedSurfaceEstimator(mode=mode, stereo_noise=noise).estimate(
        rgb, raw, {'mask': mask}, cal.intrinsics, cal.camera_to_world_cv,
        cal.bottom_z_m, cal.area_grid(), cal.radii_m)
    return {
        'scene': scene, 'geometry_source': bundle['source'],
        'measurement_route': 'surface_' + mode,
        'statistics': result['statistics'],
        'numeric_candidate_m': result['level_m'],
        'reasons': result['quality_flags'],
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
