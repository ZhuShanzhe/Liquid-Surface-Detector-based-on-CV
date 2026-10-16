import numpy as np

from liquid_depth.scene_route_policy_v45 import surface_candidate
from test_empty_scene_routes_v44 import bundle


def test_uneven_preserves_observed_mask_partition_route():
    raw = np.full((60, 80), 5., np.float32)
    rgb = np.zeros((60, 80, 3), np.uint8)
    mask = np.zeros(raw.shape, bool)
    mask[15:45, 20:60] = True
    result = surface_candidate(bundle(), rgb, raw, mask, scene='uneven_surface')
    assert result['measurement_route'] == 'observed_mask_nonplanar_partition'
    assert result['input_support']['predicted_mask_pixels'] == int(mask.sum())
    assert result['input_support']['geometry_consistent_pixels'] is None
    assert not result['production_qualified']


def test_flat_scene_still_dispatches_to_geometry_checks():
    raw = np.full((60, 80), 5., np.float32)
    rgb = np.zeros((60, 80, 3), np.uint8)
    result = surface_candidate(bundle(), rgb, raw, np.ones(raw.shape, bool), scene='glare')
    assert result['measurement_route'] == 'surface_sensor'
    assert result['input_support']['geometry_consistent_pixels'] is not None
