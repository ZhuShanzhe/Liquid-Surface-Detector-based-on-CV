"""Development-fitted sensor noise and range-conditioned point reliability.

Probabilities describe raw-depth support inliers, not liquid-level accuracy.
Profiles are simulator-specific and must not be treated as hardware calibration.
"""

from __future__ import annotations

import numpy as np

RANGE_EDGES = [0.05, 0.15, 0.35, 0.75, 1.5, 3.5, 5.0, 9.0, 11.0]
SCORE_EDGES = np.linspace(0, 1, 11).tolist()


class RangeNoiseCalibration:
    def __init__(self, payload, sensor, *, calibrate_confidence=True):
        if payload.get("schema_version") != 1 or sensor not in payload["sensors"]:
            raise ValueError("Unsupported noise calibration or sensor")
        self.profile = payload["sensors"][sensor]
        self.sensor = sensor
        self.calibrate_confidence = calibrate_confidence
        self.edges = np.asarray(payload["range_edges"], float)
        self.score_edges = np.asarray(payload["score_edges"], float)
        self.coefficients = np.asarray(self.profile["sigma_coefficients"], float)
        self.probabilities = np.asarray(self.profile["probabilities"], float)
        self.lower = np.asarray(self.profile["wilson_lower"], float)
        self.counts = np.asarray(self.profile["counts"], float)
        shape = (len(self.edges) - 1, len(self.score_edges) - 1)
        if (
            self.coefficients.shape != (2,)
            or self.probabilities.shape != shape
            or self.lower.shape != shape
            or self.counts.shape != shape
            or not np.isfinite(self.coefficients).all()
            or np.any(self.coefficients < 0)
            or not np.isfinite(self.edges).all()
            or np.any(np.diff(self.edges) <= 0)
            or not np.isfinite(self.score_edges).all()
            or np.any(np.diff(self.score_edges) <= 0)
            or not np.isfinite(self.probabilities).all()
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.counts).all()
            or np.any(self.counts < 0)
            or np.any((self.probabilities < 0) | (self.probabilities > 1))
            or np.any((self.lower < 0) | (self.lower > 1))
        ):
            raise ValueError("Invalid range calibration")

    def sigma(self, distance):
        return float(max(0.0001, self.coefficients[0] + self.coefficients[1] * distance**2))

    def reliability(self, confidence, distance):
        ri = int(np.clip(np.searchsorted(self.edges, distance, side="right") - 1, 0, len(self.edges) - 2))
        ci = np.clip(
            np.searchsorted(self.score_edges, np.nan_to_num(confidence), side="right") - 1,
            0,
            len(self.score_edges) - 2,
        )
        return self.probabilities[ri, ci], self.lower[ri, ci], self.counts[ri, ci]

    def select(self, raw, interior, confidence):
        support = interior & np.isfinite(raw) & (raw > 0)
        distance = float(np.median(raw[support])) if support.any() else 0.0
        available = bool(self.edges[0] <= distance <= self.edges[-1])
        probability, lower, count = self.reliability(confidence, distance)
        if self.calibrate_confidence:
            selected = support & np.isfinite(confidence) & (lower >= 0.90) & (count >= 128)
        else:
            selected = support & (confidence >= 0.3)
        if not available:
            selected[:] = False
        sigma = self.sigma(distance)
        gate = min(0.25, max(0.004, 1.5 * sigma))
        return selected, {
            "range_calibration_available": available,
            "optical_distance_m": distance,
            "sensor_sigma_m": sigma,
            "plane_gate_m": gate,
            "calibrated_confidence_used": self.calibrate_confidence,
            "point_reliability_mean": float(probability[selected].mean()) if selected.any() else None,
        }
