from __future__ import annotations

import numpy as np

from liquid_depth.calibration import fit_output_calibration


def test_output_calibration_reports_leave_one_out_metrics():
    predicted = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])
    known = 1.02 * predicted + 0.004
    result = fit_output_calibration(predicted, known)
    assert result["recommended_samples_met"]
    assert result["robust_inliers"] == 5
    assert result["loocv_mae_m"] < 1e-10


def test_output_calibration_downweights_one_bad_reference():
    predicted = np.asarray([0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1])
    known = 1.01 * predicted + 0.003
    known[3] += 0.08
    result = fit_output_calibration(predicted, known)
    assert abs(result["scale"] - 1.01) < 0.02
    assert abs(result["offset_m"] - 0.003) < 0.01
