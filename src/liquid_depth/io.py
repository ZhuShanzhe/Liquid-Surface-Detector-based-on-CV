from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    frame_id: str
    rgb_bgr: np.ndarray
    depth: np.ndarray
    camera_matrix: np.ndarray
    source_dir: Path


def load_frame(frame_dir: str | Path) -> RGBDFrame:
    source = Path(frame_dir).expanduser().resolve()
    rgb_path = source / "rgb.png"
    depth_path = source / "depth.npy"
    info_path = source / "depth_info.json"
    missing = [str(path) for path in (rgb_path, depth_path, info_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete frame; missing: " + ", ".join(missing))

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(f"OpenCV could not decode {rgb_path}")
    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D depth image, got {depth.shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape[:2]} vs {depth.shape}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    matrix = np.asarray(info["K"], dtype=np.float64).reshape(3, 3)
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"Invalid camera intrinsics in {info_path}")
    return RGBDFrame(source.name, rgb, depth, matrix, source)


def write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

