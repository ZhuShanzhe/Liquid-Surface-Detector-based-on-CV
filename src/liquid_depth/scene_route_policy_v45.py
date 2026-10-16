"""Legacy V45 scene-boundary research adapter.

The current uneven-surface dual-height entry point is UnevenSurfaceMeter;
this observed-mask partition path remains for historical V45 replay only.

The flat-surface routes use V44's visibility and physical-return checks. A
non-planar liquid surface is different: selecting only geometry-consistent
pixels before partitioning erases valid crest/trough evidence. Preserve its
observed-mask partition route while still using the fitted empty geometry.
"""
import numpy as np

from .empty_geometry_v41 import height_calibration
from .empty_scene_routes_v44 import surface_candidate as flat_surface_candidate
from .surface_refinement import RefinedSurfaceEstimator


def surface_candidate(bundle, rgb_bgr, depth_m, predicted_mask, *, scene):
    if scene != 'uneven_surface':
        return flat_surface_candidate(bundle, rgb_bgr, depth_m, predicted_mask,
                                      scene=scene)
    cal = height_calibration(bundle)
    raw = np.asarray(depth_m)
    rgb = np.asarray(rgb_bgr)
    mask = np.asarray(predicted_mask, bool)
    if raw.ndim != 2 or rgb.shape != (*raw.shape, 3) or mask.shape != raw.shape:
        raise ValueError('aligned_rgb_depth_and_predicted_mask_required')
    if list(bundle['image_size']) != [raw.shape[1], raw.shape[0]]:
        raise ValueError('image_shape_changed_recalibration_required')
    result = RefinedSurfaceEstimator(mode='partition').estimate(
        rgb, raw, {'mask': mask}, cal.intrinsics, cal.camera_to_world_cv,
        cal.bottom_z_m, cal.area_grid(), cal.radii_m)
    return {'scene': scene, 'geometry_source': bundle['source'],
            'measurement_route': 'observed_mask_nonplanar_partition',
            'statistics': result['statistics'],
            'numeric_candidate_m': result['level_m'],
            'reasons': list(result['quality_flags']),
            'input_support': {'predicted_mask_pixels': int(mask.sum()),
                              'geometry_consistent_pixels': None,
                              'ray_roi_pixels': None},
            'accepted': False, 'production_qualified': False,
            'research_only': True}
