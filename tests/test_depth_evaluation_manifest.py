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


def test_manifest_evaluation_uses_independent_target_and_prediction_scales(tmp_path):
    np.save(tmp_path / "target_mm.npy", np.full((2, 2), 1000, dtype=np.uint16))
    np.save(tmp_path / "prediction_m.npy", np.ones((2, 2), dtype=np.float32))
    np.save(tmp_path / "mask.npy", np.ones((2, 2), dtype=np.uint8))
    fields = (
        "target_depth_path",
        "prediction_path",
        "mask_path",
        "scenario",
        "dataset",
        "target_depth_scale_to_m",
        "prediction_depth_scale_to_m",
    )
    with (tmp_path / "scaled.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "target_depth_path": "target_mm.npy",
            "prediction_path": "prediction_m.npy",
            "mask_path": "mask.npy",
            "scenario": "scale_contract",
            "dataset": "synthetic",
            "target_depth_scale_to_m": "0.001",
            "prediction_depth_scale_to_m": "1.0",
        })
    result = evaluate_depth_manifest(tmp_path / "scaled.csv")
    assert result["overall"]["depth_mae_m"] == 0.0
    assert result["dataset:synthetic"]["prediction_coverage"] == 1.0
