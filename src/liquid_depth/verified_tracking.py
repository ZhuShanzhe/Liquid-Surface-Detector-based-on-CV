"""Independent-observation gating and explicit reacquisition after lost depth."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .surface_memory import MetricSurfaceMemory


class VerifiedSurfaceTracker:
    def __init__(self, *, confirmation_frames=5, memory_options=None):
        if confirmation_frames < 2:
            raise ValueError("confirmation_frames must be at least two")
        self.metric = MetricSurfaceMemory(**(memory_options or {}))
        self.confirmation_frames = confirmation_frames
        self.state = "acquiring"
        self.pending = []

    def _reject(self, reason, candidate=None, witness=None):
        self.pending.clear()
        self.state = "lost"
        return {
            "accepted": False,
            "level_m": None,
            "state": self.state,
            "reasons": [reason],
            "candidate_level_m": candidate,
            "witness": witness,
        }

    def process(self, rgb, raw, prediction, matrix, pose, bottom, *, witness, pose_valid=True):
        mask = np.asarray(prediction["mask"], bool)
        valid = np.isfinite(raw) & (raw > 0)
        ratio = float(np.count_nonzero(mask & valid) / max(np.count_nonzero(mask), 1))
        if ratio <= 0.05:
            self.metric.frame += 1
            return self._reject("unsupported_95_100_percent_depth_failure", witness=witness)
        if not pose_valid:
            return self._reject("pose_revalidation_required", witness=witness)
        # Probe transactionally: rejected candidates cannot rewrite trusted memory.
        trial = deepcopy(self.metric)
        candidate = trial.estimate(rgb, raw, prediction, matrix, pose, bottom, guard_jumps=False)
        self.metric.frame = trial.frame
        self.metric.records = [
            r for r in self.metric.records if self.metric.frame - r.frame <= self.metric.max_age_frames
        ]
        if not candidate["accepted"]:
            return self._reject(candidate["reasons"][0], witness=witness)
        level = candidate["level_m"]
        if not witness or not witness.get("available"):
            return self._reject("independent_rgb_evidence_unavailable", level, witness)
        if witness.get("depth_input_used") is not False:
            return self._reject("evidence_is_not_depth_independent", level, witness)
        independent = witness.get("level_m")
        uncertainty = witness.get("uncertainty_proxy_m")
        if (
            independent is None
            or uncertainty is None
            or not np.isfinite(independent + uncertainty)
            or uncertainty < 0
        ):
            return self._reject("invalid_independent_evidence", level, witness)
        gate = max(0.005, 0.02 * max(abs(level), abs(independent)), 2 * uncertainty)
        if abs(level - independent) > gate:
            return self._reject("depth_rgb_metric_disagreement", level, witness)
        if (
            self.state == "tracking"
            and self.metric.last_level_m is not None
            and abs(level - self.metric.last_level_m) > self.metric.max_accepted_step_m
        ):
            self.state = "acquiring"
            self.pending.clear()
        if self.state != "tracking":
            self.pending.append(level)
            if len(self.pending) > self.confirmation_frames:
                self.pending = self.pending[-self.confirmation_frames :]
            if np.ptp(self.pending) > max(0.005, 0.02 * abs(level)):
                self.pending = [level]
            if len(self.pending) < self.confirmation_frames:
                return {
                    "accepted": False,
                    "level_m": None,
                    "state": "confirming",
                    "reasons": ["reacquisition_pending"],
                    "confirmation_count": len(self.pending),
                    "candidate_level_m": level,
                    "witness": witness,
                }
            # Fresh independent evidence establishes a new reference. Old points
            # are discarded even when their age limit has not yet expired.
            trial.records = trial.records[-1:]
            self.state = "tracking"
            self.pending.clear()
            reacquired = True
        else:
            reacquired = False
        self.metric = trial
        return {
            "accepted": True,
            "level_m": level,
            "state": self.state,
            "reasons": [],
            "reacquired": reacquired,
            "witness": witness,
            "agreement_gate_m": gate,
            "geometry": candidate,
        }
