import numpy as np
import pytest

from liquid_depth.training.dataset import _complex_scene_augment


def _sample():
    rgb = np.full((64, 64, 3), 160, dtype=np.uint8)
    depth = np.ones((64, 64), dtype=np.float32)
    mask = np.ones((64, 64), dtype=np.float32)
    return rgb, depth, mask


def test_specialist_augmentations_preserve_their_contracts():
    rgb, depth, mask = _sample()
    np.random.seed(4)
    low_light, low_depth = _complex_scene_augment(
        rgb,
        depth.copy(),
        mask,
        "low_light",
    )
    assert float(low_light.mean()) < float(rgb.mean())
    assert np.array_equal(low_depth, depth)

    np.random.seed(4)
    _, failed_depth = _complex_scene_augment(
        rgb,
        depth.copy(),
        mask,
        "depth_failure",
    )
    assert 0 < int((failed_depth == 0).sum()) < depth.size

    np.random.seed(4)
    glare, glare_depth = _complex_scene_augment(
        rgb,
        depth.copy(),
        mask,
        "glare",
    )
    assert glare.max() > rgb.max()
    assert int((glare_depth == 0).sum()) > 0


def test_unknown_specialist_augmentation_is_rejected():
    rgb, depth, mask = _sample()
    with pytest.raises(ValueError):
        _complex_scene_augment(
            rgb,
            depth,
            mask,
            "unsupported",
        )
