from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


class SegmentationDataset:
    """CSV-backed dataset with columns image_path, mask_path, and split."""

    def __init__(self, manifest: str | Path, split: str, image_size: tuple[int, int], augment: bool = False):
        import torch

        self.torch = torch
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.image_size = image_size
        self.augment = augment
        with self.manifest.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.rows = [row for row in rows if row["split"].strip() == split]
        if not self.rows:
            raise ValueError(f"No rows for split '{split}' in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not read dataset row {row}")
        width, height = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if self.augment and np.random.random() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = self.torch.from_numpy(image.transpose(2, 0, 1)).float()
        target = self.torch.from_numpy((mask >= 128).astype(np.int64))
        return image, target


def _read_array(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    value = cv2.imread(str(path), flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _depth_meters(depth: np.ndarray, declared_scale: float | None) -> np.ndarray:
    depth = depth.astype(np.float32)
    if declared_scale is not None:
        depth *= float(declared_scale)
    else:
        valid = np.isfinite(depth) & (depth > 0)
        if np.any(valid) and float(np.median(depth[valid])) > 10.0:
            depth /= 1000.0
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)


class MultiTaskDataset:
    """Canonical RGB-D manifest for mask, metric depth, normal, and uncertainty training."""

    REQUIRED_COLUMNS = {  # noqa: RUF012
        "rgb_path",
        "raw_depth_path",
        "target_depth_path",
        "mask_path",
        "normal_path",
        "split",
        "sequence_id",
        "difficulty_tags",
    }

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: tuple[int, int],
        max_depth_m: float,
        augment: bool = False,
    ) -> None:
        import torch

        self.torch = torch
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.image_size = image_size
        self.max_depth_m = float(max_depth_m)
        self.augment = augment
        with self.manifest.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = self.REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
            rows = list(reader)
        self.rows = [row for row in rows if row["split"].strip() == split]
        if not self.rows:
            raise ValueError(f"No rows for split '{split}' in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def __getitem__(self, index: int):
        row = self.rows[index]
        rgb = _read_array(self._path(row["rgb_path"]), cv2.IMREAD_COLOR)
        raw_depth = _read_array(self._path(row["raw_depth_path"]))
        target_depth = _read_array(self._path(row["target_depth_path"]))
        mask = _read_array(self._path(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        normal = _read_array(self._path(row["normal_path"]))
        if normal.ndim != 3 or normal.shape[2] != 3:
            raise ValueError(f"Normal map must be HxWx3: {self._path(row['normal_path'])}")

        declared_scale = float(row["depth_scale_to_m"]) if row.get("depth_scale_to_m") else None
        raw_depth = _depth_meters(raw_depth, declared_scale)
        target_depth = _depth_meters(target_depth, declared_scale)
        if normal.dtype == np.uint8:
            normal = normal.astype(np.float32) / 127.5 - 1.0
        else:
            normal = normal.astype(np.float32)

        width, height = self.image_size
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        raw_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        target_depth = cv2.resize(target_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        normal = cv2.resize(normal, (width, height), interpolation=cv2.INTER_LINEAR)
        if self.augment and np.random.random() < 0.5:
            rgb = rgb[:, ::-1].copy()
            raw_depth = raw_depth[:, ::-1].copy()
            target_depth = target_depth[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            normal = normal[:, ::-1].copy()
            normal[..., 0] *= -1.0

        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
        mask = (mask >= 128).astype(np.float32)
        valid = ((target_depth > 0) & (target_depth <= self.max_depth_m)).astype(np.float32) * mask
        raw_valid = ((raw_depth > 0) & (raw_depth <= self.max_depth_m)).astype(np.float32)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        inputs = np.concatenate(
            (rgb, np.clip(raw_depth / self.max_depth_m, 0.0, 1.0)[..., None], raw_valid[..., None]),
            axis=2,
        )
        target = {
            "mask": self.torch.from_numpy(mask[None]).float(),
            "depth_m": self.torch.from_numpy(target_depth[None]).float(),
            "normal": self.torch.from_numpy(normal.transpose(2, 0, 1)).float(),
            "valid": self.torch.from_numpy(valid[None]).float(),
            "normal_valid": self.torch.from_numpy(valid[None]).float(),
        }
        return self.torch.from_numpy(inputs.transpose(2, 0, 1)).float(), target
