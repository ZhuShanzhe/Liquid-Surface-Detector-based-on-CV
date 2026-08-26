import csv
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.training.dataset import MultiTaskDataset


def test_multitask_dataset_handles_research_contract(tmp_path: Path) -> None:
    rgb = np.full((6, 8, 3), 127, dtype=np.uint8)
    raw = np.zeros((6, 8, 3), dtype=np.uint16)
    raw[..., 2] = 1000
    target = np.full((6, 8, 3), 1200, dtype=np.uint16)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:4, 3:5] = 1
    cv2.imwrite(str(tmp_path / "rgb.png"), rgb)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    np.save(tmp_path / "raw.npy", raw)
    np.save(tmp_path / "target.npy", target)

    fields = [
        "rgb_path",
        "raw_depth_path",
        "target_depth_path",
        "mask_path",
        "normal_path",
        "split",
        "sequence_id",
        "difficulty_tags",
        "depth_scale_to_m",
        "corrupt_depth_in_mask",
    ]
    with (tmp_path / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "rgb_path": "rgb.png",
                "raw_depth_path": "raw.npy",
                "target_depth_path": "target.npy",
                "mask_path": "mask.png",
                "normal_path": "",
                "split": "train",
                "sequence_id": "sequence-a",
                "difficulty_tags": "transparent;glare",
                "depth_scale_to_m": "0.001",
                "corrupt_depth_in_mask": "1",
            }
        )

    dataset = MultiTaskDataset(tmp_path / "manifest.csv", "train", (8, 6), 3.0)
    inputs, target_values = dataset[0]
    assert inputs.shape == (5, 6, 8)
    assert target_values["normal"].shape == (3, 6, 8)
    assert target_values["mask"][0, 2, 3] == 1
    assert inputs[3, 2, 3] == 0
    assert inputs[4, 2, 3] == 0
    assert np.isclose(float(inputs[3, 0, 0]), 1.0 / 3.0)
    assert np.isfinite(target_values["normal"].numpy()).all()
