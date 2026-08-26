import numpy as np
import torch

from liquid_depth.cleargrasp_refiner import ClearGraspRefiner


def test_cleargrasp_normal_rotation_matches_depth2depth_axes() -> None:
    normals = np.asarray([[[1.0]], [[2.0]], [[3.0]]], dtype=np.float32)
    rotated = ClearGraspRefiner._rotate_normals(normals)
    np.testing.assert_array_equal(
        rotated,
        np.asarray([[[1.0]], [[-3.0]], [[2.0]]], dtype=np.float32),
    )


def test_cleargrasp_occlusion_weight_matches_released_formula() -> None:
    logits = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    weight = ClearGraspRefiner._occlusion_weight(logits)
    expected = int(((1.0 - 1.0 / 3.0) ** 3) * 1000.0)
    np.testing.assert_array_equal(weight, np.full((2, 2), expected, dtype=np.uint16))
