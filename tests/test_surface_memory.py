from __future__ import annotations

import numpy as np

from liquid_depth.surface_memory import MetricSurfaceMemory, robust_plane, world_points


def fixture(level=0.3, x=0.0):
    h, w = 90, 120
    k = np.array([[120.0, 0.0, 60.0], [0.0, 120.0, 45.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = [x, 0.0, 1.0]
    raw = np.full((h, w), 1.0 - (level - 0.3), np.float32)
    mask = np.zeros((h, w), bool)
    mask[15:75, 20:100] = True
    raw[~mask] = 0
    y, xgrid = np.indices((h, w))
    rgb = np.repeat((((xgrid // 4 + y // 4) % 2) * 100 + 70)[..., None], 3, axis=2).astype(np.uint8)
    prediction = {"mask": mask, "confidence": np.ones((h, w)), "depth_m": raw}
    return rgb, raw, prediction, k, pose, -0.3


def test_memory_cannot_output_when_raw_anchors_disappear():
    memory = MetricSurfaceMemory()
    args = fixture()
    assert memory.estimate(*args)["accepted"]
    failed = list(args)
    failed[1] = np.zeros_like(args[1])
    result = memory.estimate(*failed)
    assert not result["accepted"]
    assert "insufficient_fresh_metric_anchors" in result["reasons"]


def test_jump_is_latched_and_explicit_reacquisition_is_required():
    memory = MetricSurfaceMemory()
    assert memory.estimate(*fixture())["accepted"]
    for _ in range(6):
        result = memory.estimate(*fixture(0.36))
        assert not result["accepted"]
        assert result["reasons"] == ["unverified_liquid_level_jump"]
    memory.reset()
    result = memory.estimate(*fixture(0.36))
    assert result["accepted"]
    assert abs(result["level_m"] - 0.36) < 0.001


def test_rising_liquid_tracks_current_height_without_old_level_drag():
    memory = MetricSurfaceMemory()
    for i in range(20):
        level = 0.3 + i * 0.002
        result = memory.estimate(*fixture(level))
        assert result["accepted"]
        assert abs(result["level_m"] - level) < 0.001


def test_pose_motion_preserves_world_liquid_depth():
    memory = MetricSurfaceMemory()
    for x in np.linspace(0, 0.10, 10):
        result = memory.estimate(*fixture(x=x))
        assert result["accepted"]
        assert abs(result["level_m"] - 0.3) < 0.001


def test_invalid_pose_clears_history():
    memory = MetricSurfaceMemory()
    memory.estimate(*fixture())
    result = memory.estimate(*fixture(), pose_valid=False)
    assert not result["accepted"]
    assert not memory.records


def test_collinear_support_is_not_a_plane():
    points = np.column_stack((np.linspace(0, 1, 50), np.zeros(50), np.zeros(50)))
    assert robust_plane(points) is None


def test_backprojection_opencv_world_contract():
    _, _, _, k, pose, _ = fixture()
    points = world_points(np.ones(1), np.array([[60.0, 45.0]]), k, pose)
    np.testing.assert_allclose(points, [[0.0, 0.0, 0.0]], atol=1e-7)


def test_memory_only_activates_when_fresh_spatial_support_is_weak():
    memory = MetricSurfaceMemory()
    good = memory.estimate(*fixture())
    assert good["accepted"]
    assert not good["memory_activated"]
    limited = list(fixture(0.304))
    raw = np.zeros_like(limited[1])
    raw[18:40, 25:38] = limited[1][18:40, 25:38]
    limited[1] = raw
    baseline = MetricSurfaceMemory().estimate(*limited, use_memory=False)
    assert not baseline["accepted"]
    recovered = memory.estimate(*limited)
    assert recovered["accepted"]
    assert recovered["memory_activated"]
    assert recovered["history_points"] >= 64
    assert abs(recovered["level_m"] - 0.304) < 0.001


def test_jump_gate_is_independent_of_memory_ablation():
    guarded = MetricSurfaceMemory()
    guarded.estimate(*fixture(), use_memory=False)
    assert not guarded.estimate(*fixture(0.36), use_memory=False)["accepted"]
    diagnostic = MetricSurfaceMemory()
    diagnostic.estimate(*fixture(), use_memory=False, guard_jumps=False)
    assert diagnostic.estimate(*fixture(0.36), use_memory=False, guard_jumps=False)["accepted"]
