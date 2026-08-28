from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

from .sampling import parse_tags


@dataclass
class DepthMetricAccumulator:
    target_count: int = 0
    prediction_count: int = 0
    error_count: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    boundary_error_count: int = 0
    boundary_squared_error_sum: float = 0.0
    confidence_count: int = 0
    confidence_sum: float = 0.0
    error_sum_for_confidence: float = 0.0
    error_squared_sum_for_confidence: float = 0.0
    confidence_squared_sum: float = 0.0
    confidence_error_product_sum: float = 0.0
    within_tolerance_count: int = 0
    relative_tolerance: float = 0.01
    absolute_tolerance_floor_m: float = 0.003

    def update(
        self,
        target_m: np.ndarray,
        prediction_m: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray | None = None,
    ) -> None:
        if target_m.shape != prediction_m.shape or target_m.shape != mask.shape:
            raise ValueError("Target, prediction, and mask shapes must match")
        target_valid = np.isfinite(target_m) & (target_m > 0) & (mask > 0)
        prediction_valid = np.isfinite(prediction_m) & (prediction_m > 0)
        evaluated = target_valid & prediction_valid
        self.target_count += int(target_valid.sum())
        self.prediction_count += int(evaluated.sum())
        errors = prediction_m[evaluated] - target_m[evaluated]
        absolute_errors = np.abs(errors)
        tolerance = np.maximum(
            self.absolute_tolerance_floor_m,
            self.relative_tolerance * target_m[evaluated],
        )
        self.error_count += int(errors.size)
        self.within_tolerance_count += int(
            (absolute_errors <= tolerance).sum()
        )
        self.absolute_error_sum += float(absolute_errors.sum())
        self.squared_error_sum += float((errors**2).sum())

        boundary = cv2.morphologyEx(
            (mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)
        )
        boundary_errors = prediction_m[evaluated & (boundary > 0)] - target_m[evaluated & (boundary > 0)]
        self.boundary_error_count += int(boundary_errors.size)
        self.boundary_squared_error_sum += float((boundary_errors**2).sum())

        if confidence is not None:
            if confidence.shape != target_m.shape:
                raise ValueError("Confidence shape must match target depth")
            conf = np.clip(confidence[evaluated].astype(np.float64), 0.0, 1.0)
            absolute = absolute_errors.astype(np.float64)
            self.confidence_count += int(conf.size)
            self.confidence_sum += float(conf.sum())
            self.error_sum_for_confidence += float(absolute.sum())
            self.error_squared_sum_for_confidence += float((absolute**2).sum())
            self.confidence_squared_sum += float((conf**2).sum())
            self.confidence_error_product_sum += float((conf * absolute).sum())

    def finalize(self) -> dict[str, float | int | None]:
        count = max(self.error_count, 1)
        correlation = None
        if self.confidence_count > 1:
            n = self.confidence_count
            covariance = (
                self.confidence_error_product_sum - self.confidence_sum * self.error_sum_for_confidence / n
            )
            confidence_variance = self.confidence_squared_sum - self.confidence_sum**2 / n
            error_variance = self.error_squared_sum_for_confidence - self.error_sum_for_confidence**2 / n
            denominator = math.sqrt(max(confidence_variance * error_variance, 0.0))
            if denominator > 0:
                correlation = covariance / denominator
        return {
            "target_pixels": self.target_count,
            "prediction_coverage": self.prediction_count / max(self.target_count, 1),
            "depth_mae_m": self.absolute_error_sum / count,
            "depth_rmse_m": math.sqrt(self.squared_error_sum / count),
            "within_tolerance_rate": self.within_tolerance_count / count,
            "within_tolerance_coverage": (
                self.within_tolerance_count / max(self.target_count, 1)
            ),
            "relative_tolerance": self.relative_tolerance,
            "absolute_tolerance_floor_m": (
                self.absolute_tolerance_floor_m
            ),
            "boundary_depth_rmse_m": math.sqrt(
                self.boundary_squared_error_sum / max(self.boundary_error_count, 1)
            ),
            "confidence_absolute_error_correlation": correlation,
        }


def _array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".exr" and value.ndim == 3:
        value = value[..., 0]
    return value


def _meters(value: np.ndarray, scale: float | None) -> np.ndarray:
    value = value.astype(np.float32)
    if scale is not None:
        return value * scale
    valid = np.isfinite(value) & (value > 0)
    return value / 1000.0 if np.any(valid) and np.median(value[valid]) > 10 else value


def _resolve(root: Path, row: dict[str, str], key: str) -> Path:
    value = Path(row[key]).expanduser()
    return value if value.is_absolute() else root / value


def evaluate_depth_manifest(manifest: str | Path) -> dict:
    manifest = Path(manifest).resolve()
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"target_depth_path", "prediction_path", "mask_path"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Manifest requires columns: {', '.join(sorted(required))}")
    root = manifest.parent
    groups: dict[str, DepthMetricAccumulator] = defaultdict(DepthMetricAccumulator)
    groups["overall"] = DepthMetricAccumulator()
    for row in rows:
        scale = float(row["depth_scale_to_m"]) if row.get("depth_scale_to_m") else None
        target = _meters(_array(_resolve(root, row, "target_depth_path")), scale)
        prediction = _meters(_array(_resolve(root, row, "prediction_path")), scale)
        mask = _array(_resolve(root, row, "mask_path"))
        if mask.ndim == 3:
            mask = np.any(mask[..., :3] != 0, axis=2).astype(np.uint8)
        if mask.shape != target.shape:
            mask = cv2.resize(
                mask,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        confidence = _array(_resolve(root, row, "confidence_path")) if row.get("confidence_path") else None
        names = {"overall", f"scenario:{row.get('scenario', 'unspecified') or 'unspecified'}"}
        names.update(f"difficulty:{tag}" for tag in parse_tags(row.get("difficulty_tags", "ordinary")))
        for name in names:
            groups[name].update(target, prediction, mask, confidence)
    return {name: accumulator.finalize() for name, accumulator in sorted(groups.items())}
