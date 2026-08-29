from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .io import RGBDFrame, load_frame, write_json


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _depth_meters(depth: np.ndarray, declared_scale: float | None) -> np.ndarray:
    value = np.asarray(depth, dtype=np.float32)
    if value.ndim == 3:
        valid_counts = [
            int(np.count_nonzero(np.isfinite(value[..., index]) & (value[..., index] > 0)))
            for index in range(value.shape[2])
        ]
        value = value[..., int(np.argmax(valid_counts))]
    if value.ndim != 2:
        raise ValueError(f"Virtual camera depth must be HxW, got {value.shape}")
    if declared_scale is not None:
        value *= declared_scale
    else:
        valid = np.isfinite(value) & (value > 0)
        if np.any(valid) and float(np.median(value[valid])) > 10.0:
            value /= 1000.0
    return np.where(np.isfinite(value) & (value > 0), value, 0.0).astype(np.float32)


def load_manifest_rows(
    manifest: str | Path,
    split: str,
    *,
    scenarios: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    path = Path(manifest).expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    allowed = {value.strip() for value in scenarios or () if value.strip()}
    selected = [
        row
        for row in rows
        if row.get("split", "").strip() == split
        and (not allowed or row.get("scenario", "").strip() in allowed)
    ]
    if limit is not None:
        if limit < 1:
            raise ValueError("Virtual camera frame limit must be positive")
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No virtual camera rows found for split={split!r}")
    return selected


def export_virtual_capture(
    row: Mapping[str, str],
    *,
    manifest_root: str | Path,
    output_root: str | Path,
    index: int,
) -> Path:
    root = Path(manifest_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    rgb_path = _resolve(root, row["rgb_path"])
    raw_depth_path = _resolve(root, row["raw_depth_path"])
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(rgb_path)
    raw_depth = np.load(raw_depth_path, allow_pickle=False)
    declared_scale = (
        float(row["depth_scale_to_m"]) if row.get("depth_scale_to_m", "").strip() else None
    )
    depth_m = _depth_meters(raw_depth, declared_scale)
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(f"Virtual RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth_m.shape}")

    metadata_path_value = row.get("metadata_path", "").strip()
    metadata: dict[str, Any] = {}
    metadata_path = None
    if metadata_path_value:
        metadata_path = _resolve(root, metadata_path_value)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(
        metadata.get(
            "camera_intrinsics",
            [
                [max(rgb.shape[:2]), 0.0, (rgb.shape[1] - 1) / 2.0],
                [0.0, max(rgb.shape[:2]), (rgb.shape[0] - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
        ),
        dtype=np.float64,
    ).reshape(3, 3)

    frame_id = f"virtual_{index:06d}"
    target = output / frame_id
    target.mkdir(parents=True, exist_ok=False)
    if not cv2.imwrite(str(target / "rgb.png"), rgb):
        raise OSError(target / "rgb.png")
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = (depth_m > 0) & (depth_m < np.iinfo(np.uint16).max / 1000.0)
    depth_mm[valid] = np.rint(depth_m[valid] * 1000.0).astype(np.uint16)
    np.save(target / "depth.npy", depth_mm, allow_pickle=False)

    camera_info = {
        "width": rgb.shape[1],
        "height": rgb.shape[0],
        "distortion_model": "plumb_bob",
        "D": [0.0] * 5,
        "K": intrinsics.reshape(-1).tolist(),
        "R": np.eye(3).reshape(-1).tolist(),
        "P": [
            intrinsics[0, 0],
            0.0,
            intrinsics[0, 2],
            0.0,
            0.0,
            intrinsics[1, 1],
            intrinsics[1, 2],
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
        "depth_scale_to_m": 0.001,
        "source": "virtual_rgbd_camera",
    }
    write_json(target / "depth_info.json", camera_info)
    write_json(target / "color_info.json", camera_info)
    write_json(
        target / "scene_context.json",
        {
            "scenario": row.get("scenario", "unknown"),
            "difficulty_tags": row.get("difficulty_tags", ""),
            "sensor_model": row.get("sensor_model", metadata.get("sensor_model", "unknown")),
            "virtual_camera": True,
        },
    )
    write_json(
        target / "virtual_camera.json",
        {
            "frame_id": frame_id,
            "source_manifest_root": root.as_posix(),
            "source_rgb_path": rgb_path.as_posix(),
            "source_raw_depth_path": raw_depth_path.as_posix(),
            "source_target_depth_path": _resolve(root, row["target_depth_path"]).as_posix(),
            "source_mask_path": _resolve(root, row["mask_path"]).as_posix(),
            "source_metadata_path": metadata_path.as_posix() if metadata_path else None,
            "sequence_id": row.get("sequence_id", frame_id),
            "scenario": row.get("scenario", "unknown"),
            "difficulty_tags": row.get("difficulty_tags", ""),
            "depth_scale_to_m": 0.001,
            "quantization": "uint16_millimeter",
            "hardware_validated": False,
        },
    )
    write_json(output / "latest.json", {"frame_id": frame_id, "path": target.as_posix()})
    return target


def replay_manifest(
    manifest: str | Path,
    output_root: str | Path,
    *,
    split: str,
    scenarios: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    manifest_path = Path(manifest).expanduser().resolve()
    rows = load_manifest_rows(manifest_path, split, scenarios=scenarios, limit=limit)
    return [
        export_virtual_capture(
            row,
            manifest_root=manifest_path.parent,
            output_root=output_root,
            index=index,
        )
        for index, row in enumerate(rows)
    ]


def prepare_universal_camera_input(
    frame: RGBDFrame | str | Path,
    *,
    image_size: tuple[int, int],
    min_depth_m: float,
    max_depth_m: float,
    depth_scale_to_m: float | None = None,
) -> np.ndarray:
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError("Expected 0 < min_depth_m < max_depth_m")
    value = load_frame(frame) if not isinstance(frame, RGBDFrame) else frame
    if depth_scale_to_m is None:
        info = json.loads((value.source_dir / "depth_info.json").read_text(encoding="utf-8"))
        depth_scale_to_m = float(info.get("depth_scale_to_m", 0.001))
    width, height = image_size
    rgb = cv2.resize(value.rgb_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(value.depth, (width, height), interpolation=cv2.INTER_NEAREST)
    depth_m = _depth_meters(depth, float(depth_scale_to_m))
    valid = ((depth_m >= min_depth_m) & (depth_m <= max_depth_m)).astype(np.float32)
    encoded = np.log(
        np.clip(depth_m, min_depth_m, max_depth_m) / min_depth_m
    ) / np.log(max_depth_m / min_depth_m)
    encoded = np.where(valid > 0, encoded, 0.0).astype(np.float32)
    rgb_unit = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb_normalized = (
        rgb_unit - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return np.concatenate(
        (rgb_normalized, encoded[..., None], valid[..., None]),
        axis=2,
    ).transpose(2, 0, 1).astype(np.float32)
