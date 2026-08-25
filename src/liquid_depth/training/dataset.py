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

