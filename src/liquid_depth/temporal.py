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
    recovered: bool = False
    hold_frames: int = 0


class RobustKalmanFilter:
    """One-dimensional Kalman filter with confidence-aware innovation gating."""

    def __init__(
        self,
        process_variance: float = 0.01,
        measurement_variance: float = 0.25,
        gate_sigma: float = 3.5,
        max_jump: float = 2.0,
        min_confidence: float = 0.20,
        max_hold_frames: int = 3,
        hold_confidence_decay: float = 0.65,
    ) -> None:
        self.process_variance = float(process_variance)
        self.measurement_variance = float(measurement_variance)
        self.gate_sigma = float(gate_sigma)
        self.max_jump = float(max_jump)
        self.min_confidence = float(min_confidence)
        self.max_hold_frames = max(0, int(max_hold_frames))
        self.hold_confidence_decay = float(hold_confidence_decay)
        self.value: float | None = None
        self.variance: float | None = None
        self.hold_frames = 0

    def reset(self) -> None:
        self.value = None
        self.variance = None
        self.hold_frames = 0

    def _prediction_hold(self, confidence: float, reason: str) -> TemporalResult:
        self.hold_frames += 1
        recovered = self.value is not None and self.hold_frames <= self.max_hold_frames
        if self.variance is None:
            state_confidence = 0.0
        else:
            state_confidence = 1.0 / (1.0 + math.sqrt(max(self.variance, 0.0)))
        decayed = min(confidence, state_confidence) * self.hold_confidence_decay**self.hold_frames
        return TemporalResult(
            self.value,
            self.variance,
            decayed,
            False,
            reason,
            None,
            recovered,
            self.hold_frames,
        )

    def update(
        self,
        measurement: float,
        confidence: float,
        upstream_accepted: bool = True,
    ) -> TemporalResult:
        confidence = max(0.0, min(1.0, float(confidence)))
        if self.variance is not None:
            self.variance += self.process_variance
        if not upstream_accepted:
            return self._prediction_hold(confidence, "upstream_quality_rejection")
        if not math.isfinite(measurement):
            return self._prediction_hold(confidence, "non_finite_measurement")
        if confidence < self.min_confidence:
            return self._prediction_hold(confidence, "low_temporal_input_confidence")

        measurement = float(measurement)
        observation_variance = self.measurement_variance / max(confidence * confidence, 1e-4)
        if self.value is None or self.variance is None:
            self.value = measurement
            self.variance = observation_variance
            self.hold_frames = 0
            return TemporalResult(self.value, self.variance, confidence, True, None, 0.0)

        innovation = measurement - self.value
        innovation_variance = self.variance + observation_variance
        statistical_gate = self.gate_sigma * math.sqrt(innovation_variance)
        gate = min(self.max_jump, statistical_gate) if self.max_jump > 0 else statistical_gate
        if abs(innovation) > gate:
            held = self._prediction_hold(confidence, "temporal_innovation_too_large")
            return TemporalResult(
                held.value,
                held.variance,
                held.confidence,
                False,
                held.reason,
                innovation,
                held.recovered,
                held.hold_frames,
            )

        gain = self.variance / innovation_variance
        self.value += gain * innovation
        self.variance *= 1.0 - gain
        self.hold_frames = 0
        fused_confidence = max(
            0.0,
            min(1.0, 1.0 / (1.0 + math.sqrt(self.variance))),
        )
        return TemporalResult(
            self.value,
            self.variance,
            fused_confidence,
            True,
            None,
            innovation,
        )


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
        max_hold_frames=int(options.get("max_hold_frames", 3)),
        hold_confidence_decay=float(options.get("hold_confidence_decay", 0.65)),
    )
