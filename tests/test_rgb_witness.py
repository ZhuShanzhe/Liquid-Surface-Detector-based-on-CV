import numpy as np
import pytest

from liquid_depth.rgb_witness import RGBContourWitness


def reference():
    k = np.array([[120.0, 0.0, 60.0], [0.0, 120.0, 45.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[2, 3] = 1.0
    w = RGBContourWitness()
    w.rx = 0.25
    w.ry = 0.18
    w.bottom = -0.3
    mask = w._project_mask(0.3, k, pose, (90, 120))
    image = np.full((90, 120, 3), 230, np.uint8)
    image[mask > 0] = [70, 140, 80]
    w.calibrate(image, mask, 0.3, k, pose, -0.3, 0.25, 0.18)
    return w, image, k, pose


def test_rgb_calibration_roundtrip_and_depth_independence():
    witness, image, k, pose = reference()
    result = witness.estimate(image, k, pose)
    assert result["available"] and not result["depth_input_used"]
    assert abs(result["level_m"] - 0.3) < 0.001
    restored = RGBContourWitness.from_dict(witness.to_dict())
    assert restored.estimate(image, k, pose) == result


def test_unobservable_dark_image_cannot_be_used_as_metric_evidence():
    witness, image, k, pose = reference()
    assert not witness.estimate(np.zeros_like(image), k, pose)["available"]


def test_invalid_calibration_is_rejected():
    witness, _, _, _ = reference()
    payload = witness.to_dict()
    payload["spread"] = [1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        RGBContourWitness.from_dict(payload)


def test_resolution_bound_includes_annotation_error_and_roundtrips():
    witness, image, k, pose = reference()
    measured = witness.estimate(image, k, pose, resolution_checks=True)
    assert measured["available"] and measured["resolution_checked"]
    assert measured["error_bound_proxy_m"] >= 2 * measured["uncertainty_proxy_m"]
    assert not measured["bound_is_statistically_calibrated"]
    witness.calibration_error_m = 0.01
    restored = RGBContourWitness.from_dict(witness.to_dict())
    larger = restored.estimate(image, k, pose, resolution_checks=True)
    assert abs(larger["error_bound_proxy_m"] - measured["error_bound_proxy_m"] - 0.009) < 1e-9


def test_invalid_annotation_error_is_rejected():
    with pytest.raises(ValueError):
        RGBContourWitness(calibration_error_m=-1)
