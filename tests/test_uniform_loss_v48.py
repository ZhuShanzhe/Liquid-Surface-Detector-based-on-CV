import numpy as np
from dataclasses import replace

from evaluate_uniform_loss_v48 import delete_to_actual_loss
from liquid_depth.simulation import render_geometric_labels, sample_scene, simulate_raw_depth


def test_diagnostic_switch_preserves_default_and_disables_only_dropout():
    scene = replace(sample_scene(5, seed=13, width=96, height=54,
                                 scenario_profile='calibration'),
                    scenario='depth_failure')
    assert scene.scenario == 'depth_failure'
    labels = render_geometric_labels(scene)
    default = simulate_raw_depth(scene, labels)
    explicit = simulate_raw_depth(scene, labels, components={'scenario_dropout': True})
    native = simulate_raw_depth(scene, labels, components={'scenario_dropout': False})
    np.testing.assert_array_equal(default['raw_depth_m'], explicit['raw_depth_m'])
    assert not native['simulated_dropout_mask'].any()
    assert native['raw_depth_m'].shape == default['raw_depth_m'].shape


def test_exact_actual_loss_only_deletes_existing_returns():
    raw = np.full((10, 20), 2., np.float32)
    visible = np.ones(raw.shape, bool)
    output = delete_to_actual_loss(raw, visible, 30, 7)
    assert np.count_nonzero(output[visible]) == 140
    assert np.all(output <= raw)
    assert np.count_nonzero(raw) == 200


def test_cannot_reach_lower_loss_without_backfill():
    raw = np.ones((5, 20), np.float32)
    raw[:, :5] = 0
    visible = np.ones(raw.shape, bool)
    try:
        delete_to_actual_loss(raw, visible, 10, 1)
    except ValueError as exc:
        assert str(exc) == 'native_loss_exceeds_requested_target'
    else:
        raise AssertionError('missing-return backfill must be forbidden')
