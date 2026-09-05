import numpy as np

from liquid_depth.anchor_memory import TemporalAnchorMemory
from liquid_depth.container_geometry import ContainerModel, project_model_points


def _scene():
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 21), np.linspace(-0.3, 0.3, 13))
    model = ContainerModel(
        np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))),
        np.array([0.0, 1.0, 0.0]),
        np.zeros(3),
    )
    camera = np.array([[180.0, 0.0, 160.0], [0.0, 180.0, 120.0], [0.0, 0.0, 1.0]])
    rotation = np.eye(3)
    translation = np.array([0.0, 0.0, 2.0])
    projected, levels, _ = project_model_points(model, camera, rotation, translation)
    curve = projected[np.argsort(np.abs(levels))[:16]]
    curve = curve[np.argsort(curve[:, 0])]
    image = np.full((240, 320, 3), (80, 120, 160), dtype=np.uint8)
    return model, camera, rotation, translation, curve, image


def _memory(**overrides):
    options = {
        "min_confidence": 0.55,
        "min_current_points": 1,
        "min_total_points": 6,
        "min_occupied_bins": 2,
        "max_memory_fraction": 0.95,
        "max_model_match_px": 2.0,
        "spatial_match_px": 12.0,
    }
    options.update(overrides)
    return TemporalAnchorMemory(**options)


def test_pose_aligned_history_recovers_weak_curve_points():
    model, camera, rotation, translation, curve, image = _scene()
    memory = _memory()
    committed = memory.commit(
        image,
        curve,
        np.full(len(curve), 0.95),
        model,
        camera,
        rotation,
        translation,
    )
    assert committed == len(curve)
    moved_translation = np.array([0.05, 0.0, 2.0])
    weak_curve = curve + np.array([4.5, 0.0])
    weak_confidence = np.full(len(curve), 0.15)
    weak_confidence[len(curve) // 2] = 0.9
    fused = memory.fuse(
        image,
        weak_curve,
        weak_confidence,
        camera,
        rotation,
        moved_translation,
        roi_xyxy=(60, 60, 260, 180),
    )
    assert fused.accepted
    assert fused.recovered_points >= 6
    assert fused.occupied_bins >= 2
    assert fused.mean_rgb_similarity is not None


def test_memory_only_output_is_rejected_by_observability_gate():
    model, camera, rotation, translation, curve, image = _scene()
    memory = _memory(min_current_points=2, max_memory_fraction=0.7)
    memory.commit(
        image,
        curve,
        np.full(len(curve), 0.95),
        model,
        camera,
        rotation,
        translation,
    )
    fused = memory.fuse(
        image,
        curve,
        np.full(len(curve), 0.05),
        camera,
        rotation,
        translation,
    )
    assert not fused.accepted
    assert "insufficient_current_anchor_observability" in fused.rejection_reasons
    assert "temporal_memory_fraction_too_high" in fused.rejection_reasons


def test_rgb_change_blocks_stale_anchor_reuse():
    model, camera, rotation, translation, curve, image = _scene()
    memory = _memory(min_current_points=0, min_rgb_similarity=0.8)
    memory.commit(
        image,
        curve,
        np.full(len(curve), 0.95),
        model,
        camera,
        rotation,
        translation,
    )
    changed = np.full_like(image, (250, 20, 20))
    fused = memory.fuse(
        changed,
        curve,
        np.full(len(curve), 0.05),
        camera,
        rotation,
        translation,
    )
    assert not fused.accepted
    assert fused.recovered_points == 0
    assert "insufficient_temporal_anchor_points" in fused.rejection_reasons
