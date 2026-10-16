"""Ensure optional floating-depth proxies do not invent missing sensor returns."""

import numpy as np
from evaluate_uneven_dual_height import add_floats

from liquid_depth.liquid_height_metrics import HeightCalibration


def test_float_proxy_preserves_missing_sensor_depth():
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    k = np.array([[100.0, 0.0, 2.5], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]])
    cal = HeightCalibration(k, pose, 0.0, (1.0, 1.0))
    target = np.ones((6, 6), dtype=np.float32)
    raw = target.copy()
    raw[2, 2] = 0
    specs = [(0.0, 0.0, 0.1, 0.1, 0.05)]
    result = add_floats(raw, target, np.ones(raw.shape, bool), cal, specs, preserve_missing=True)
    assert result[2, 2] == 0
    assert result[3, 3] < raw[3, 3]
    assert np.array_equal(result == 0, raw == 0)
