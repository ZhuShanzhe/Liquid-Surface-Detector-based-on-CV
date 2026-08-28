import csv

import cv2
import numpy as np

from liquid_depth.depth_evaluation import evaluate_depth_manifest


def test_manifest_evaluation_normalizes_rgba_mask_resolution(tmp_path):
    np.save(tmp_path / "target.npy", np.ones((4, 4), dtype=np.float32))
    np.save(tmp_path / "prediction.npy", np.ones((4, 4), dtype=np.float32))
    np.save(tmp_path / "confidence.npy", np.ones((4, 4), dtype=np.float32))
    mask = np.zeros((2, 2, 4), dtype=np.uint8)
    mask[..., :3] = 255
    mask[..., 3] = 255
    cv2.imwrite(str(tmp_path / "mask.png"), mask)

    fields = (
        "target_depth_path",
        "prediction_path",
        "mask_path",
        "confidence_path",
        "scenario",
        "difficulty_tags",
        "depth_scale_to_m",
    )
    with (tmp_path / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "target_depth_path": "target.npy",
                "prediction_path": "prediction.npy",
                "mask_path": "mask.png",
                "confidence_path": "confidence.npy",
                "scenario": "format_contract",
                "difficulty_tags": "transparent",
                "depth_scale_to_m": "",
            }
        )

    result = evaluate_depth_manifest(tmp_path / "manifest.csv")
    assert result["overall"]["prediction_coverage"] == 1.0
    assert result["overall"]["depth_mae_m"] == 0.0
