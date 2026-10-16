import numpy as np
import pytest

from liquid_depth.empty_scene_routes_v43 import build_scene_geometry, surface_candidate


def synthetic_inputs():
    k = np.array([[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]])
    angle = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    def outline(height):
        distance = 6. - height
        return np.column_stack((40 + 100 * np.cos(angle) / distance,
                                30 - 100 * np.sin(angle) / distance)).tolist()
    common = {
        'shape': 'circle', 'wall_height_m': 2., 'baseline_m': .055,
        'sensor_family': 'active_stereo',
        'metric_anchor': {'kind': 'perpendicular_distance', 'distance_m': 6.},
        'confirmed_empty': True, 'horizontal_flat_bottom': True,
        'fixed_camera': True, 'rectified_aligned': True,
    }
    bottom = {**common, 'bottom_outline_px': outline(0)}
    rim = {**common, 'vertical_walls': True, 'rim_outline_px': outline(2),
           'up_camera': [0., 0., -1.],
           'measured_rim_dimensions_m': [2., 2.]}
    return k, bottom, rim


def test_empty_rim_fallback_requires_explicit_independent_inputs():
    k, bottom, rim = synthetic_inputs()
    dry = np.zeros((60, 80), np.float32)
    bundle = build_scene_geometry(dry, k, bottom, rim_spec=rim)
    assert bundle['source'] == 'empty_rim_plus_independent_metric_and_gravity'
    assert bundle['quality']['bottom_depth_observed'] is False
    assert abs(bundle['calibration']['radii_m'][0] - 1.) < 1e-5
    assert abs(bundle['calibration']['camera_to_world_cv'][2][3] - 6.) < 1e-5
    with pytest.raises(ValueError, match='independently_calibrated_gravity_required'):
        build_scene_geometry(dry, k, bottom, rim_spec={**rim, 'up_camera': None})


def test_route_clips_outside_container_without_clipping_height():
    k, bottom, rim = synthetic_inputs()
    bundle = build_scene_geometry(np.zeros((60, 80), np.float32), k, bottom, rim_spec=rim)
    rgb = np.zeros((60, 80, 3), np.uint8)
    depth = np.full((60, 80), 5., np.float32)
    depth[:, :15] = 2.  # spatially outside the calibrated vessel
    out = surface_candidate(bundle, rgb, depth, np.ones(depth.shape, bool), scene='glare')
    assert out['input_support']['geometry_consistent_pixels'] < depth.size
    assert out['numeric_candidate_m'] is not None
    assert abs(out['numeric_candidate_m'] - 1.) < .1
    assert out['accepted'] is False
