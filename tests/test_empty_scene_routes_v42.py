import numpy as np
import pytest

from liquid_depth.empty_scene_routes_v42 import surface_candidate


def bundle(sensor):
    k = np.array([[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 6.
    return {
        'source': 'empty_rgbd_plus_operator_metric_anchor',
        'image_size': [80, 60], 'sensor_family': sensor,
        'calibration': {
            'intrinsics': k.tolist(), 'camera_to_world_cv': pose.tolist(),
            'bottom_z_m': 0., 'radii_m': [1.8, 1.8],
            'footprint_exponent': 2.,
        },
    }


@pytest.mark.parametrize('sensor', ['active_stereo', 'structured_light', 'tof'])
def test_scene_uses_shared_geometry_without_production_acceptance(sensor):
    rgb = np.zeros((60, 80, 3), np.uint8)
    depth = np.full((60, 80), 5., np.float32)
    yy, xx = np.indices(depth.shape)
    mask = ((xx - 40) ** 2 + (yy - 30) ** 2) < 26 ** 2
    out = surface_candidate(bundle(sensor), rgb, depth, mask, scene='low_light')
    assert out['numeric_candidate_m'] is not None
    assert not out['accepted'] and not out['production_qualified']
    assert out['geometry_source'] == 'empty_rgbd_plus_operator_metric_anchor'


def test_route_requires_explicit_scene_and_same_image_shape():
    rgb = np.zeros((60, 80, 3), np.uint8)
    depth = np.full((60, 80), 5., np.float32)
    mask = np.ones((60, 80), bool)
    with pytest.raises(ValueError, match='explicit_supported_surface_scene_required'):
        surface_candidate(bundle('tof'), rgb, depth, mask, scene='transparent')
    with pytest.raises(ValueError, match='image_shape_changed_recalibration_required'):
        surface_candidate(bundle('tof'), rgb[:50], depth[:50], mask[:50], scene='glare')
