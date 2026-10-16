import numpy as np

from liquid_depth.empty_scene_routes_v44 import surface_candidate


def bundle():
    k = np.array([[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 6.
    return {
        'source': 'empty_rim_plus_independent_metric_and_gravity',
        'sensor_family': 'active_stereo', 'image_size': [80, 60],
        'rim_z_m': 2.,
        'calibration': {'intrinsics': k.tolist(),
                        'camera_to_world_cv': pose.tolist(),
                        'bottom_z_m': 0., 'radii_m': [1., 1.],
                        'footprint_exponent': 2.},
    }


def test_opening_and_pointwise_geometry_accept_real_surface():
    raw = np.full((60, 80), 5., np.float32)
    raw[:, :12] = 2.
    rgb = np.zeros((60, 80, 3), np.uint8)
    out = surface_candidate(bundle(), rgb, raw, np.ones(raw.shape, bool), scene='glare')
    assert out['numeric_candidate_m'] is not None
    assert abs(out['numeric_candidate_m'] - 1.) < .1
    assert out['input_support']['geometry_consistent_pixels'] < raw.size
    assert not out['accepted']


def test_severe_loss_keeps_no_output():
    raw = np.zeros((60, 80), np.float32)
    raw[30, 40] = 5.
    rgb = np.zeros((60, 80, 3), np.uint8)
    out = surface_candidate(bundle(), rgb, raw, np.ones(raw.shape, bool), scene='depth_failure')
    assert out['numeric_candidate_m'] is None
    assert 'unsupported_95_100_percent_depth_failure' in out['reasons']
