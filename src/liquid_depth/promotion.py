from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromotionRequirements:
    max_mae_m: float
    min_prediction_coverage: float = 0.80
    min_within_tolerance_coverage: float = 0.70
    min_mae_improvement_fraction: float = 0.0
    max_failed_frame_ratio: float = 0.01
    max_median_latency_ms: float = 500.0
    max_guard_mae_regression_fraction: float = 0.10
    max_guard_tolerance_coverage_loss: float = 0.02


def _metrics(summary: dict[str, Any], scenario: str) -> dict[str, Any] | None:
    metrics = summary.get("metrics", summary)
    return metrics.get(f"scenario:{scenario}")


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def assess_depth_specialist_promotion(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    target_scenario: str,
    requirements: PromotionRequirements,
    guard_scenarios: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Apply accuracy, coverage, failure, latency, and no-regression gates."""

    reasons: list[str] = []
    target = _metrics(candidate, target_scenario)
    reference = _metrics(baseline, target_scenario)
    if target is None:
        reasons.append("target_scenario_missing")
        target = {}
    if reference is None:
        reasons.append("baseline_target_scenario_missing")
        reference = {}

    coverage = target.get("prediction_coverage")
    mae = target.get("depth_mae_m")
    tolerance_coverage = target.get("within_tolerance_coverage")
    if not _finite(coverage) or coverage < requirements.min_prediction_coverage:
        reasons.append("target_prediction_coverage_below_gate")
    if not _finite(mae) or mae > requirements.max_mae_m:
        reasons.append("target_mae_above_gate")
    if (
        not _finite(tolerance_coverage)
        or tolerance_coverage
        < requirements.min_within_tolerance_coverage
    ):
        reasons.append("target_within_tolerance_coverage_below_gate")

    baseline_mae = reference.get("depth_mae_m")
    if _finite(mae) and _finite(baseline_mae) and baseline_mae > 0:
        improvement = (baseline_mae - mae) / baseline_mae
        if improvement < requirements.min_mae_improvement_fraction:
            reasons.append("target_mae_improvement_below_gate")
    else:
        improvement = None

    frames = max(int(candidate.get("frames", 0)), 1)
    failed_ratio = int(candidate.get("failed_frames", 0)) / frames
    if failed_ratio > requirements.max_failed_frame_ratio:
        reasons.append("failed_frame_ratio_above_gate")
    latency = candidate.get("median_latency_ms")
    if not _finite(latency) or latency > requirements.max_median_latency_ms:
        reasons.append("median_latency_above_gate")

    guard_results: dict[str, Any] = {}
    for scenario in guard_scenarios:
        base_guard = _metrics(baseline, scenario)
        candidate_guard = _metrics(candidate, scenario)
        guard_reasons: list[str] = []
        if base_guard is None or candidate_guard is None:
            guard_reasons.append("guard_scenario_missing")
        else:
            base_guard_mae = base_guard.get("depth_mae_m")
            candidate_guard_mae = candidate_guard.get("depth_mae_m")
            if (
                _finite(base_guard_mae)
                and _finite(candidate_guard_mae)
                and candidate_guard_mae
                > base_guard_mae
                * (1.0 + requirements.max_guard_mae_regression_fraction)
            ):
                guard_reasons.append("guard_mae_regression")
            base_tolerance = base_guard.get("within_tolerance_coverage")
            candidate_tolerance = candidate_guard.get(
                "within_tolerance_coverage"
            )
            if (
                _finite(base_tolerance)
                and _finite(candidate_tolerance)
                and candidate_tolerance
                + requirements.max_guard_tolerance_coverage_loss
                < base_tolerance
            ):
                guard_reasons.append("guard_tolerance_coverage_regression")
        if guard_reasons:
            reasons.extend(
                f"{scenario}:{reason}" for reason in guard_reasons
            )
        guard_results[scenario] = {
            "accepted": not guard_reasons,
            "reasons": guard_reasons,
            "baseline": base_guard,
            "candidate": candidate_guard,
        }

    reasons = list(dict.fromkeys(reasons))
    return {
        "accepted": not reasons,
        "target_scenario": target_scenario,
        "rejection_reasons": reasons,
        "requirements": asdict(requirements),
        "diagnostics": {
            "target": target,
            "baseline_target": reference,
            "target_mae_improvement_fraction": improvement,
            "failed_frame_ratio": failed_ratio,
            "median_latency_ms": latency,
            "guards": guard_results,
        },
    }
