import numpy as np
import pytest

from liquid_depth.training.contact_metrics import (
    confidence_error_correlation,
    summarize_curve_group,
    summarize_selective_curve,
)


def test_selective_curve_retains_low_error_for_informative_confidence():
    errors = np.array([1.0, 2.0, 8.0, 9.0])
    confidence = np.array([0.9, 0.8, 0.2, 0.1])

    summary = summarize_curve_group(errors, confidence)

    assert summary["confidence_error_correlation"] < -0.9
    assert summary["selective_risk"][2]["target_coverage"] == 0.5
    assert summary["selective_risk"][2]["mean_px"] == pytest.approx(1.5)


def test_confidence_metrics_reject_shape_mismatch_and_constant_is_zero():
    assert confidence_error_correlation([1.0, 2.0], [0.5, 0.5]) == 0.0
    with pytest.raises(ValueError, match="same shape"):
        summarize_selective_curve([1.0, 2.0], [0.5])
