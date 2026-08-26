from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

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


def _depth_channel(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        valid_counts = [
            int((np.isfinite(depth[..., index]) & (depth[..., index] > 0)).sum())
            for index in range(depth.shape[2])
        ]
        depth = depth[..., int(np.argmax(valid_counts))]
    if depth.ndim != 2:
        raise ValueError(f"Depth map must be HxW or HxWxC, got {depth.shape}")
    return depth


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        # Ignore alpha: TODD stores instance ids in RGB with an opaque alpha channel.
        mask = np.any(mask[..., :3] != 0, axis=2)
    return (mask != 0).astype(np.float32)


def _normal_from_depth(depth_m: np.ndarray) -> np.ndarray:
    horizontal = cv2.Sobel(depth_m, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(depth_m, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.stack((-horizontal, -vertical, np.ones_like(depth_m)), axis=2)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    normal[depth_m <= 0] = 0.0
    return normal.astype(np.float32)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _complex_scene_augment(
    rgb: np.ndarray, raw_depth: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    image = rgb.astype(np.float32)
    if np.random.random() < 0.8:
        gain = np.random.uniform(0.65, 1.35)
        bias = np.random.uniform(-20.0, 20.0)
        image = image * gain + bias
    if np.random.random() < 0.45:
        height, width = mask.shape
        locations = np.argwhere(mask > 0)
        if locations.size:
            center_y, center_x = locations[np.random.randint(len(locations))]
        else:
            center_y, center_x = np.random.randint(height), np.random.randint(width)
        axes = (
            max(2, int(width * np.random.uniform(0.02, 0.10))),
            max(2, int(height * np.random.uniform(0.02, 0.10))),
        )
        highlight = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(
            highlight,
            (int(center_x), int(center_y)),
            axes,
            float(np.random.uniform(0.0, 180.0)),
            0.0,
            360.0,
            255,
            -1,
        )
        softness = cv2.GaussianBlur(highlight, (0, 0), sigmaX=max(1.0, min(axes) / 2.0))
        alpha = softness.astype(np.float32)[..., None] / 255.0 * np.random.uniform(0.7, 1.0)
        image = image * (1.0 - alpha) + 255.0 * alpha
        if np.random.random() < 0.8:
            raw_depth[highlight > 0] = 0.0
    if np.random.random() < 0.3:
        image += np.random.normal(0.0, np.random.uniform(2.0, 10.0), image.shape)
    return np.clip(image, 0.0, 255.0).astype(np.uint8), raw_depth


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
        raw_depth = _depth_channel(_read_array(self._path(row["raw_depth_path"])))
        target_depth = _depth_channel(_read_array(self._path(row["target_depth_path"])))
        mask = _binary_mask(_read_array(self._path(row["mask_path"])))
        normal_value = row.get("normal_path", "").strip()
        normal = _read_array(self._path(normal_value)) if normal_value else None
        if normal is not None:
            if normal.ndim != 3 or normal.shape[2] < 3:
                raise ValueError(f"Normal map must be HxWx3: {self._path(normal_value)}")
            normal = normal[..., :3]
            if row.get("normal_channel_order", "").strip().lower() == "bgr":
                normal = normal[..., ::-1]

        declared_scale = float(row["depth_scale_to_m"]) if row.get("depth_scale_to_m") else None
        raw_depth = _depth_meters(raw_depth, declared_scale)
        target_depth = _depth_meters(target_depth, declared_scale)
        if normal is not None:
            if normal.dtype == np.uint8:
                normal = normal.astype(np.float32) / 127.5 - 1.0
            else:
                normal = normal.astype(np.float32)

        width, height = self.image_size
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        raw_depth = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        target_depth = cv2.resize(target_depth, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        normal = (
            cv2.resize(normal, (width, height), interpolation=cv2.INTER_LINEAR)
            if normal is not None
            else _normal_from_depth(target_depth)
        )
        if _truthy(row.get("corrupt_depth_in_mask", "")):
            raw_depth[mask > 0] = 0.0
        if self.augment and np.random.random() < 0.5:
            rgb = rgb[:, ::-1].copy()
            raw_depth = raw_depth[:, ::-1].copy()
            target_depth = target_depth[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            normal = normal[:, ::-1].copy()
            normal[..., 0] *= -1.0
        if self.augment:
            rgb, raw_depth = _complex_scene_augment(rgb, raw_depth, mask)

        normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
        mask = (mask > 0).astype(np.float32)
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
        normal_valid = valid * (np.linalg.norm(normal, axis=2) > 1e-6).astype(np.float32)
        target = {
            "mask": self.torch.from_numpy(mask[None]).float(),
            "depth_m": self.torch.from_numpy(target_depth[None]).float(),
            "normal": self.torch.from_numpy(normal.transpose(2, 0, 1)).float(),
            "valid": self.torch.from_numpy(valid[None]).float(),
            "normal_valid": self.torch.from_numpy(normal_valid[None]).float(),
        }
        return self.torch.from_numpy(inputs.transpose(2, 0, 1)).float(), target
