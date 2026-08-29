from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.io import load_frame
from liquid_depth.virtual_camera import prepare_universal_camera_input, replay_manifest


def _write_manifest_fixture(root: Path) -> Path:
    sample = root / "sample"
    sample.mkdir()
    rgb = np.full((4, 6, 3), (20, 80, 160), dtype=np.uint8)
    assert cv2.imwrite(str(sample / "rgb.png"), rgb)
    raw_depth = np.full((4, 6), 1.25, dtype=np.float32)
    raw_depth[0, 0] = 0.0
    target_depth = np.full((4, 6), 1.20, dtype=np.float32)
    mask = np.ones((4, 6), dtype=np.uint8)
    np.save(sample / "raw_depth_m.npy", raw_depth)
    np.save(sample / "target_depth_m.npy", target_depth)
    np.save(sample / "mask.npy", mask)
    (sample / "metadata.json").write_text(
        json.dumps(
            {
                "camera_intrinsics": [
                    [120.0, 0.0, 2.5],
                    [0.0, 121.0, 1.5],
                    [0.0, 0.0, 1.0],
                ],
                "sensor_model": "test_sensor",
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "manifest.csv"
    fieldnames = [
        "split",
        "scenario",
        "difficulty_tags",
        "sensor_model",
        "sequence_id",
        "rgb_path",
        "raw_depth_path",
        "target_depth_path",
        "mask_path",
        "metadata_path",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "split": "test",
                "scenario": "transparent",
                "difficulty_tags": "invalid_depth",
                "sensor_model": "test_sensor",
                "sequence_id": "sequence_1",
                "rgb_path": "sample/rgb.png",
                "raw_depth_path": "sample/raw_depth_m.npy",
                "target_depth_path": "sample/target_depth_m.npy",
                "mask_path": "sample/mask.npy",
                "metadata_path": "sample/metadata.json",
            }
        )
    return manifest


def test_virtual_camera_replays_capture_contract(tmp_path: Path) -> None:
    manifest = _write_manifest_fixture(tmp_path)
    captures = replay_manifest(manifest, tmp_path / "captures", split="test")

    assert len(captures) == 1
    frame = load_frame(captures[0])
    assert frame.depth.dtype == np.uint16
    assert frame.depth.shape == (4, 6)
    assert frame.depth[0, 0] == 0
    assert np.all(frame.depth[1:, :] == 1250)
    assert frame.camera_matrix[0, 0] == 120.0
    provenance = json.loads(
        (captures[0] / "virtual_camera.json").read_text(encoding="utf-8")
    )
    assert provenance["hardware_validated"] is False
    assert provenance["quantization"] == "uint16_millimeter"


def test_virtual_camera_prepares_deployment_tensor(tmp_path: Path) -> None:
    manifest = _write_manifest_fixture(tmp_path)
    capture = replay_manifest(manifest, tmp_path / "captures", split="test")[0]

    inputs = prepare_universal_camera_input(
        capture,
        image_size=(6, 4),
        min_depth_m=0.1,
        max_depth_m=10.0,
    )

    assert inputs.shape == (5, 4, 6)
    assert inputs.dtype == np.float32
    assert inputs[3, 0, 0] == 0.0
    assert inputs[4, 0, 0] == 0.0
    assert inputs[3, 1, 1] > 0.0
    assert inputs[4, 1, 1] == 1.0
