import numpy as np

from audit_natural_to50_v47 import to_total_50
from audit_depth_support_v46 import controlled_raw


def test_total_50_only_removes_existing_surface_returns():
    visible = np.zeros((4, 8), bool)
    visible[1:3] = True
    raw = np.ones((4, 8), np.float32)
    raw[1, :4] = 0
    output = to_total_50(raw, visible, 17)
    assert output is not None
    assert np.count_nonzero(output[visible]) == 8
    assert not np.any((raw == 0) & (output > 0))
    assert np.array_equal(output[~visible], raw[~visible])


def test_total_50_rejects_already_more_severe_frame():
    visible = np.ones((4, 8), bool)
    raw = np.ones((4, 8), np.float32)
    raw[:, :5] = 0
    assert to_total_50(raw, visible, 1) is None


def test_controlled_proxy_exact_surface_loss():
    visible = np.ones((10, 10), bool)
    target = np.full((10, 10), 4., np.float32)
    raw = np.zeros_like(target)
    output = controlled_raw(raw, target, visible, 'active_stereo', 25, 5)
    assert np.count_nonzero(output[visible]) == 75
