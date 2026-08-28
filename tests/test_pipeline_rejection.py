import json

import cv2
import numpy as np

from liquid_depth.pipeline import infer_frame
from liquid_depth.refinement import RefinedDepth


class _Segmenter:
    def predict(self, rgb):
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        mask[8:24, 8:24] = 255
        return mask, np.ones(rgb.shape[:2], dtype=np.float32)


class _EmptyRefiner:
    def predict(self, rgb, depth):
        del rgb
        return RefinedDepth(
            np.zeros(depth.shape, dtype=np.float32),
            np.zeros(depth.shape, dtype=np.float32),
            "empty_test",
        )


def test_infer_rejects_missing_liquid_depth_without_aborting(tmp_path):
    frame = tmp_path / "frame"
    frame.mkdir()
    cv2.imwrite(str(frame / "rgb.png"), np.full((32, 32, 3), 80, np.uint8))
    np.save(frame / "depth.npy", np.ones((32, 32), dtype=np.float32))
    (frame / "depth_info.json").write_text(
        json.dumps(
            {
                "K": [
                    100.0,
                    0.0,
                    16.0,
                    0.0,
                    100.0,
                    16.0,
                    0.0,
                    0.0,
                    1.0,
                ]
            }
        )
    )
    config = {
        "segmentation": {"backend": "test"},
        "illumination": {"enabled": False},
        "complex_scene": {
            "mode": "off",
            "latency_budget_ms": 500.0,
            "enforce_latency_budget": True,
            "missing_model_policy": "reject",
        },
        "geometry": {
            "liquid_erode_px": 0,
            "meniscus_width_px": 0,
            "ransac_threshold_m": 0.006,
            "max_points": 1000,
            "seed": 7,
            "min_depth_confidence": 0.25,
        },
        "surface_support": {"enabled": True},
        "output": {
            "depth_unit": "cm",
            "calibration_scale_per_meter": 100.0,
        },
    }
    result = infer_frame(
        frame,
        tmp_path / "unused_bottom.json",
        tmp_path / "output",
        config,
        segmenter=_Segmenter(),
        depth_refiner=_EmptyRefiner(),
        complex_depth_refiners={},
    )
    assert not result["accepted"]
    assert result["liquid_depth"] is None
    assert "insufficient_liquid_depth_support" in result["rejection_reasons"]
    assert (tmp_path / "output" / "depth_result.json").is_file()
