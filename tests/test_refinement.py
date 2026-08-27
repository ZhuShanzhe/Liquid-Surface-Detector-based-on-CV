import numpy as np

from liquid_depth.geometry import Plane, plane_angle_degrees
from liquid_depth.refinement import (
    IdentityDepthRefiner,
    make_complex_depth_refiners,
)


def test_identity_refiner_converts_millimeters_and_validity():
    depth = np.asarray([[1000, 0], [2000, 1500]], dtype=np.uint16)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)

    result = IdentityDepthRefiner().predict(rgb, depth)

    np.testing.assert_allclose(result.depth_m, [[1.0, 0.0], [2.0, 1.5]])
    np.testing.assert_array_equal(result.confidence, [[1.0, 0.0], [1.0, 1.0]])


def test_plane_angle_is_sign_invariant():
    first = Plane(np.asarray([0.0, 0.0, -1.0]), 1.0, np.zeros(3))
    second = Plane(np.asarray([0.0, 0.0, 1.0]), -1.0, np.zeros(3))

    assert plane_angle_degrees(first, second) == 0.0


def test_identical_complex_refiners_share_one_loaded_instance():
    config = {
        "complex_scene": {
            "models": {
                "glare": {"backend": "identity"},
                "depth_failure": {"backend": "identity"},
                "disabled": {"enabled": False},
            }
        }
    }
    refiners = make_complex_depth_refiners(config)
    assert set(refiners) == {"glare", "depth_failure"}
    assert refiners["glare"] is refiners["depth_failure"]
