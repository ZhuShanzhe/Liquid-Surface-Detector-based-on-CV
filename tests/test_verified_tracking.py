from __future__ import annotations

import numpy as np
from test_surface_memory import fixture

from liquid_depth.verified_tracking import VerifiedSurfaceTracker


def cue(level):
    return {"available": True, "level_m": level, "uncertainty_proxy_m": 0.001, "depth_input_used": False}


def warm(tracker, level=0.3):
    for _ in range(5):
        result = tracker.process(*fixture(level), witness=cue(level))
    assert result["accepted"]


def test_slow_echo_does_not_rewrite_trusted_reference():
    tracker = VerifiedSurfaceTracker()
    warm(tracker)
    for i in range(1, 12):
        args = list(fixture())
        args[1] = args[1].copy()
        args[1][args[1] > 0] += 0.003 * i
        result = tracker.process(*args, witness=cue(0.3))
        if i >= 3:
            assert not result["accepted"]
    assert tracker.metric.last_level_m > 0.29


def test_changed_level_reacquires_only_after_five_independent_confirmations():
    tracker = VerifiedSurfaceTracker()
    warm(tracker)
    args = list(fixture())
    args[1] = np.zeros_like(args[1])
    for _ in range(60):
        assert not tracker.process(*args, witness=cue(0.3))["accepted"]
    for i in range(5):
        result = tracker.process(*fixture(0.34), witness=cue(0.34))
        assert result["accepted"] == (i == 4)
    assert result["reacquired"]
    assert abs(result["level_m"] - 0.34) < 0.001


def test_wrong_echo_after_outage_cannot_reacquire():
    tracker = VerifiedSurfaceTracker()
    warm(tracker)
    args = list(fixture())
    args[1] = np.zeros_like(args[1])
    tracker.process(*args, witness=cue(0.3))
    for _ in range(10):
        result = tracker.process(*fixture(0.24), witness=cue(0.3))
        assert not result["accepted"]
        assert result["reasons"] == ["depth_rgb_metric_disagreement"]


def test_unavailable_or_depth_derived_witness_is_not_confirmation():
    tracker = VerifiedSurfaceTracker()
    for _ in range(8):
        assert not tracker.process(*fixture(), witness={"available": False})["accepted"]
    witness = cue(0.3)
    witness["depth_input_used"] = True
    assert tracker.process(*fixture(), witness=witness)["reasons"] == ["evidence_is_not_depth_independent"]


def test_lost_pose_requires_reconfirmation():
    tracker = VerifiedSurfaceTracker()
    warm(tracker)
    assert tracker.process(*fixture(), witness=cue(0.3), pose_valid=False)["reasons"] == [
        "pose_revalidation_required"
    ]
    for i in range(5):
        assert tracker.process(*fixture(), witness=cue(0.3))["accepted"] == (i == 4)
