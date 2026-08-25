import numpy as np

from liquid_depth.depth_evaluation import DepthMetricAccumulator


def test_depth_metrics_report_coverage_error_boundary_and_confidence():
    target = np.ones((8, 8), dtype=np.float32)
    prediction = target + np.linspace(0.01, 0.20, 64, dtype=np.float32).reshape(8, 8)
    prediction[1, 1] = 0.0
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 1:7] = 255
    confidence = 1.0 - np.linspace(0.01, 0.20, 64, dtype=np.float32).reshape(8, 8)
    metrics = DepthMetricAccumulator()
    metrics.update(target, prediction, mask, confidence)
    result = metrics.finalize()
    assert np.isclose(result["prediction_coverage"], 35 / 36)
    assert 0.01 < result["depth_mae_m"] < 0.20
    assert result["boundary_depth_rmse_m"] > 0
    assert result["confidence_absolute_error_correlation"] < -0.99
