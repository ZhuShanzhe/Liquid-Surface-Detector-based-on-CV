from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalResult:
    value: float | None
    variance: float | None
    confidence: float
    accepted: bool
    reason: str | None
    innovation: float | None


class RobustKalmanFilter:
    """One-dimensional Kalman filter with confidence-aware innovation gating."""

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.25,
        gate_sigma: float = 3.5,
        max_jump: float = 2.0,
        min_confidence: float = 0.20,
    ) -> None:
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.gate_sigma = float(gate_sigma)
        self.max_jump = float(max_jump)
        self.min_confidence = float(min_confidence)
        self.value: float | None = None
        self.variance: float | None = None

    def reset(self) -> None:
        self.value = None
        self.variance = None

    def update(self, measurement: float, confidence: float, upstream_accepted: bool = True) -> TemporalResult:
        confidence = max(0.0, min(1.0, float(confidence)))
        if self.variance is not None:
            self.variance += self.process_variance
        if not upstream_accepted:
            return TemporalResult(
                self.value, self.variance, confidence, False, "upstream_quality_rejection", None
            )
        if not math.isfinite(measurement):
            return TemporalResult(
                self.value, self.variance, confidence, False, "non_finite_measurement", None
            )
        if confidence < self.min_confidence:
            return TemporalResult(
                self.value, self.variance, confidence, False, "low_temporal_input_confidence", None
            )

        measurement = float(measurement)
        observation_variance = self.measurement_variance / max(confidence * confidence, 1e-4)
        if self.value is None or self.variance is None:
            self.value = measurement
            self.variance = observation_variance
            return TemporalResult(self.value, self.variance, confidence, True, None, 0.0)

        innovation = measurement - self.value
        innovation_variance = self.variance + observation_variance
        statistical_gate = self.gate_sigma * math.sqrt(innovation_variance)
        gate = min(self.max_jump, statistical_gate) if self.max_jump > 0 else statistical_gate
        if abs(innovation) > gate:
            return TemporalResult(
                self.value,
                self.variance,
                confidence,
                False,
                "temporal_innovation_too_large",
                innovation,
            )

        gain = self.variance / innovation_variance
        self.value += gain * innovation
        self.variance *= 1.0 - gain
        fused_confidence = max(0.0, min(1.0, 1.0 / (1.0 + math.sqrt(self.variance))))
        return TemporalResult(self.value, self.variance, fused_confidence, True, None, innovation)


def make_temporal_filter(config: dict) -> RobustKalmanFilter | None:
    options = config.get("temporal", {})
    if not options.get("enabled", False):
        return None
    return RobustKalmanFilter(
        process_variance=float(options.get("process_variance", 0.01)),
        measurement_variance=float(options.get("measurement_variance", 0.25)),
        gate_sigma=float(options.get("gate_sigma", 3.5)),
        max_jump=float(options.get("max_jump", 2.0)),
        min_confidence=float(options.get("min_confidence", 0.20)),
    )
