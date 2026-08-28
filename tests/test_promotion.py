from liquid_depth.promotion import (
    PromotionRequirements,
    assess_depth_specialist_promotion,
)


def _summary(glare_mae, glare_coverage, glare_tolerance, ordinary_mae=0.004):
    return {
        "frames": 100,
        "failed_frames": 0,
        "median_latency_ms": 25.0,
        "metrics": {
            "scenario:glare": {
                "depth_mae_m": glare_mae,
                "prediction_coverage": glare_coverage,
                "within_tolerance_coverage": glare_tolerance,
            },
            "scenario:ordinary": {
                "depth_mae_m": ordinary_mae,
                "prediction_coverage": 0.99,
                "within_tolerance_coverage": 0.95,
            },
        },
    }


def test_promotion_accepts_improvement_that_meets_all_gates():
    baseline = _summary(0.020, 0.70, 0.50)
    candidate = _summary(0.008, 0.95, 0.82, ordinary_mae=0.0042)
    result = assess_depth_specialist_promotion(
        baseline,
        candidate,
        target_scenario="glare",
        requirements=PromotionRequirements(
            max_mae_m=0.010,
            min_mae_improvement_fraction=0.20,
        ),
        guard_scenarios=("ordinary",),
    )
    assert result["accepted"]
    assert not result["rejection_reasons"]


def test_promotion_rejects_accuracy_coverage_failure_and_guard_regression():
    baseline = _summary(0.020, 0.80, 0.72)
    candidate = _summary(
        0.018,
        0.60,
        0.40,
        ordinary_mae=0.006,
    )
    candidate["failed_frames"] = 4
    candidate["median_latency_ms"] = 600.0
    result = assess_depth_specialist_promotion(
        baseline,
        candidate,
        target_scenario="glare",
        requirements=PromotionRequirements(
            max_mae_m=0.010,
            min_mae_improvement_fraction=0.20,
        ),
        guard_scenarios=("ordinary",),
    )
    assert not result["accepted"]
    reasons = set(result["rejection_reasons"])
    assert "target_prediction_coverage_below_gate" in reasons
    assert "target_mae_above_gate" in reasons
    assert "target_within_tolerance_coverage_below_gate" in reasons
    assert "failed_frame_ratio_above_gate" in reasons
    assert "median_latency_above_gate" in reasons
    assert "ordinary:guard_mae_regression" in reasons
