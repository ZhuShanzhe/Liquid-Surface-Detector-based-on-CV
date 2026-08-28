from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import resolve_config_path


class Segmenter(Protocol):
    def predict(self, rgb_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return uint8 binary mask and float confidence map."""


def _largest_component(mask: np.ndarray, min_area: float) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if not contours:
        return result
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) >= min_area:
        cv2.drawContours(result, [contour], -1, 255, -1)
    return result


class ClassicalSegmenter:
    def __init__(self, roi: list[int], options: dict):
        self.roi = tuple(map(int, roi))
        self.options = options

    def predict(self, rgb_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= rgb_bgr.shape[1] and 0 <= y1 < y2 <= rgb_bgr.shape[0]):
            raise ValueError(f"ROI {self.roi} is outside image shape {rgb_bgr.shape[:2]}")
        crop = rgb_bgr[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        dx, dy = self.options["ellipse_center_offset"]
        margin_x, margin_y = self.options["ellipse_axis_margin"]
        inner = np.zeros((h, w), np.uint8)
        cv2.ellipse(
            inner,
            (w // 2 + int(dx), h // 2 + int(dy)),
            (max(1, w // 2 - int(margin_x)), max(1, h // 2 - int(margin_y))),
            0,
            0,
            360,
            255,
            -1,
        )
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[..., 1], hsv[..., 2]
        candidate = (
            (saturation < self.options["saturation_max"])
            & (value > self.options["value_min"])
            & (value < self.options["value_max"])
            & (inner > 0)
        )
        mask = np.where(candidate, 255, 0).astype(np.uint8)
        open_size, close_size = self.options["open_kernel"], self.options["close_kernel"]
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = _largest_component(mask, self.options["min_area"])
        full = np.zeros(rgb_bgr.shape[:2], np.uint8)
        full[y1:y2, x1:x2] = mask
        return full, (full.astype(np.float32) / 255.0)


class TorchScriptSegmenter:
    def __init__(self, model_path: Path, input_size: list[int], threshold: float):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the 'train' extra to use the TorchScript backend") from exc
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.input_size = tuple(map(int, input_size))
        self.threshold = float(threshold)

    def predict(self, rgb_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        height, width = rgb_bgr.shape[:2]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size[::-1], interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            output = self.model((tensor - mean) / std)
        if isinstance(output, dict):
            output = output.get("out", next(iter(output.values())))
        if isinstance(output, (tuple, list)):
            output = output[0]
        probability = output[0]
        if probability.ndim == 3:
            probability = probability[0] if probability.shape[0] == 1 else probability.softmax(0)[1]
        if probability.min() < 0 or probability.max() > 1:
            probability = probability.sigmoid()
        probability = cv2.resize(probability.float().cpu().numpy(), (width, height))
        mask = np.where(probability >= self.threshold, 255, 0).astype(np.uint8)
        return mask, probability.astype(np.float32)


def make_segmenter(config: dict) -> Segmenter:
    segmentation = config["segmentation"]
    backend = segmentation["backend"].lower()
    if backend == "classical":
        return ClassicalSegmenter(segmentation["roi"], segmentation["classical"])
    if backend == "torchscript":
        options = segmentation["torchscript"]
        path = resolve_config_path(config, options["model_path"])
        return TorchScriptSegmenter(path, options["input_size"], options["threshold"])
    raise ValueError(f"Unknown segmentation backend: {backend}")


def make_bottom_mask(shape: tuple[int, ...], config: dict) -> np.ndarray:
    options = config["bottom"]
    x1, y1, x2, y2 = map(int, options["roi"])
    mask = np.zeros(shape[:2], np.uint8)
    dx, dy = options["ellipse_center_offset"]
    axes = tuple(map(int, options["ellipse_axes"]))
    cv2.ellipse(mask, ((x1 + x2) // 2 + int(dx), (y1 + y2) // 2 + int(dy)), axes, 0, 0, 360, 255, -1)
    return mask


def overlay_mask(rgb_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = rgb_bgr.copy()
    blue = np.zeros_like(result)
    blue[:] = (255, 0, 0)
    active = mask > 0
    if np.any(active):
        result[active] = cv2.addWeighted(
            result[active], 0.6, blue[active], 0.4, 0
        )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    return result

