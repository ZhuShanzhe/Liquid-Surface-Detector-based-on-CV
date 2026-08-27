from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from liquid_depth.models.multitask import ConvBlock

DTLD_OBJECT_INDEX = {"15": 0, "16": 1, "17": 2, "19": 3}


def _read_image(path: str, flags: int) -> np.ndarray:
    value = cv2.imread(path, flags)
    if value is None:
        raise FileNotFoundError(path)
    return value


def _augment_non_lambertian(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize glare, saturation, haze, and correlated active-depth failure."""
    image = rgb.astype(np.float32)
    height, width = image.shape[:2]
    if np.random.random() < 0.45:
        mask = np.zeros((height, width), dtype=np.float32)
        center = (
            int(np.random.uniform(0.15, 0.85) * width),
            int(np.random.uniform(0.1, 0.9) * height),
        )
        axes = (
            max(2, int(np.random.uniform(0.03, 0.18) * width)),
            max(2, int(np.random.uniform(0.02, 0.12) * height)),
        )
        cv2.ellipse(
            mask,
            center,
            axes,
            float(np.random.uniform(0.0, 180.0)),
            0.0,
            360.0,
            1.0,
            -1,
        )
        sigma = max(1.0, np.random.uniform(0.01, 0.04) * max(height, width))
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        alpha = np.clip(mask * np.random.uniform(0.65, 1.0), 0.0, 1.0)[..., None]
        tint = np.random.uniform(235.0, 255.0, size=(1, 1, 3))
        image = image * (1.0 - alpha) + tint * alpha
        if np.random.random() < 0.7:
            depth[mask > np.random.uniform(0.2, 0.55)] = 0.0
    if np.random.random() < 0.3:
        mean_color = image.mean(axis=(0, 1), keepdims=True)
        haze = np.random.uniform(0.15, 0.45)
        image = image * (1.0 - haze) + mean_color * haze
    if np.random.random() < 0.45:
        # Broad illumination gradients cover shadowed vessels and uneven factory lighting.
        direction = np.random.choice(("horizontal", "vertical"))
        if direction == "horizontal":
            ramp = np.linspace(
                np.random.uniform(0.22, 0.65),
                np.random.uniform(0.75, 1.15),
                width,
                dtype=np.float32,
            )[None, :, None]
        else:
            ramp = np.linspace(
                np.random.uniform(0.22, 0.65),
                np.random.uniform(0.75, 1.15),
                height,
                dtype=np.float32,
            )[:, None, None]
        if np.random.random() < 0.5:
            ramp = np.flip(ramp, axis=1 if direction == "horizontal" else 0)
        image *= ramp
    if np.random.random() < 0.25:
        # Floating material is treated as an occluder: the network must recover
        # the representative interface while lowering confidence locally.
        occluder = np.zeros((height, width), dtype=np.uint8)
        center = (np.random.randint(width), np.random.randint(height))
        axes = (
            max(3, int(width * np.random.uniform(0.03, 0.12))),
            max(3, int(height * np.random.uniform(0.03, 0.12))),
        )
        cv2.ellipse(occluder, center, axes, np.random.uniform(0, 180), 0, 360, 255, -1)
        color = np.random.uniform(15.0, 220.0, size=(1, 1, 3))
        alpha = np.random.uniform(0.65, 1.0)
        selected = occluder > 0
        image[selected] = image[selected] * (1.0 - alpha) + color.reshape(3) * alpha
        depth[selected] = 0.0
    if np.random.random() < 0.8:
        image = image * np.random.uniform(0.35, 1.4) + np.random.uniform(-35.0, 15.0)
    if np.random.random() < 0.45:
        image *= np.random.uniform(0.75, 1.25, size=(1, 1, 3))
    if np.random.random() < 0.5:
        gamma = np.random.uniform(0.55, 2.8)
        image = 255.0 * np.power(np.clip(image / 255.0, 0.0, 1.0), gamma)
    if np.random.random() < 0.25:
        signal = np.clip(image, 0.0, 255.0) / 255.0
        sigma = np.random.uniform(1.0, 6.0) + (1.0 - signal) * np.random.uniform(2.0, 14.0)
        image += np.random.normal(0.0, sigma, size=image.shape)
    if np.random.random() < 0.15:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=np.random.uniform(0.3, 1.2))
    if np.random.random() < 0.35:
        dropout = np.random.random(depth.shape) < np.random.uniform(0.01, 0.08)
        depth[dropout] = 0.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8), depth


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    locations = cv2.findNonZero((mask > 0).astype(np.uint8))
    if locations is None:
        return None
    x, y, width, height = cv2.boundingRect(locations)
    return x, y, x + width, y + height


def _clip_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    margin: float = 0.18,
) -> tuple[int, int, int, int]:
    image_height, image_width = shape
    x0, y0, x1, y1 = bbox
    width, height = max(x1 - x0, 2), max(y1 - y0, 2)
    x0 = max(0, int(x0 - width * margin))
    x1 = min(image_width, int(x1 + width * margin))
    y0 = max(0, int(y0 - height * margin))
    y1 = min(image_height, int(y1 + height * margin))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Degenerate DTLD crop: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def _jitter_crop(
    crop: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    width = max(x1 - x0, 2)
    height = max(y1 - y0, 2)
    scale = float(np.random.uniform(0.85, 1.2))
    center_x = (x0 + x1) / 2.0 + np.random.uniform(-0.08, 0.08) * width
    center_y = (y0 + y1) / 2.0 + np.random.uniform(-0.08, 0.08) * height
    half_width = width * scale / 2.0
    half_height = height * scale / 2.0
    return _clip_bbox(
        (
            int(center_x - half_width),
            int(center_y - half_height),
            int(center_x + half_width),
            int(center_y + half_height),
        ),
        shape,
        margin=0.0,
    )


def _pose_bbox(row: dict[str, str], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    camera = json.loads(row["camera_json"])
    pose = json.loads(row["pose_json"])
    matrix = np.asarray(camera["cam_K"], dtype=np.float32).reshape(3, 3)
    translation = np.asarray(pose["cam_t_m2c"], dtype=np.float32)
    if float(np.linalg.norm(translation)) > 10.0:
        translation /= 1000.0
    z = max(float(translation[2]), 1e-3)
    center_x = float(matrix[0, 0] * translation[0] / z + matrix[0, 2])
    center_y = float(matrix[1, 1] * translation[1] / z + matrix[1, 2])
    object_id = row["object_id"]
    width_fraction = 0.22 if object_id != "19" else 0.32
    height_fraction = 0.62 if object_id != "19" else 0.48
    half_width = width * width_fraction / 2.0
    half_height = height * height_fraction / 2.0
    return _clip_bbox(
        (
            int(center_x - half_width),
            int(center_y - half_height),
            int(center_x + half_width),
            int(center_y + half_height),
        ),
        shape,
        margin=0.0,
    )


def _instance_bbox(
    row: dict[str, str],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    info = json.loads(row.get("object_info_json") or "{}")
    for name in ("bbox_visib", "bbox_obj"):
        value = info.get(name)
        if isinstance(value, list) and len(value) == 4 and value[2] > 1 and value[3] > 1:
            x, y, width, height = map(int, value)
            return _clip_bbox((x, y, x + width, y + height), shape)
    mask_path = row.get("mask_path", "")
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            bbox = _bbox_from_mask(mask)
            if bbox is not None:
                return _clip_bbox(bbox, shape)
    return _pose_bbox(row, shape)


def _liquid_lane(label: dict, object_id: str) -> np.ndarray:
    if object_id == "19":
        points = np.asarray(list(label.values()), dtype=np.float32)
        lower_half = points[np.argsort(points[:, 1])[::-1]][: len(points) // 2]
        return lower_half[np.argsort(lower_half[:, 0])]
    a = np.asarray(label["A"], dtype=np.float32)
    b = np.asarray(label["B"], dtype=np.float32)
    c = np.asarray(label["C"], dtype=np.float32)
    d = np.asarray(label["D"], dtype=np.float32)
    candidates = ((a, c), (a, d), (b, c), (b, d))
    start, end = max(
        candidates,
        key=lambda pair: float(np.linalg.norm(pair[0] - pair[1])),
    )
    return np.linspace(start, end, num=20, dtype=np.float32)


def _contact_target(
    points: np.ndarray,
    crop: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = crop
    width, height = image_size
    target = np.zeros((height, width), dtype=np.float32)
    scale_x = width / max(x1 - x0, 1)
    scale_y = height / max(y1 - y0, 1)
    transformed = np.column_stack(((points[:, 0] - x0) * scale_x, (points[:, 1] - y0) * scale_y))
    rendered = np.rint(transformed).astype(np.int32)
    inside = rendered[
        (rendered[:, 0] >= 0) & (rendered[:, 0] < width) & (rendered[:, 1] >= 0) & (rendered[:, 1] < height)
    ]
    if len(inside) >= 2:
        cv2.polylines(target, [inside], False, 1.0, 3)
    for x, y in inside:
        cv2.circle(target, (int(x), int(y)), 3, 1.0, -1)
    target = cv2.GaussianBlur(target, (0, 0), sigmaX=1.5)
    if target.max() > 0:
        target /= target.max()
    return target


def _bezier_target(
    points: np.ndarray,
    crop: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    """Fit four normalized cubic Bezier control points to an annotated contact line."""
    x0, y0, x1, y1 = crop
    transformed = np.column_stack(
        (
            (points[:, 0] - x0) / max(x1 - x0, 1),
            (points[:, 1] - y0) / max(y1 - y0, 1),
        )
    )
    transformed = np.clip(transformed, 0.0, 1.0)
    if transformed[0, 0] > transformed[-1, 0]:
        transformed = transformed[::-1]
    segment = np.linalg.norm(np.diff(transformed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    if cumulative[-1] < 1e-8:
        parameter = np.linspace(0.0, 1.0, len(transformed))
    else:
        parameter = cumulative / cumulative[-1]
    basis = np.column_stack(
        (
            (1.0 - parameter) ** 3,
            3.0 * (1.0 - parameter) ** 2 * parameter,
            3.0 * (1.0 - parameter) * parameter**2,
            parameter**3,
        )
    )
    control, _, _, _ = np.linalg.lstsq(basis, transformed, rcond=None)
    control[0] = transformed[0]
    control[-1] = transformed[-1]
    control[:, 0] = np.maximum.accumulate(control[:, 0])
    return np.clip(control, 0.0, 1.0).astype(np.float32)


def _pose_features(row: dict[str, str]) -> np.ndarray:
    pose = json.loads(row["pose_json"])
    rotation = np.asarray(pose["cam_R_m2c"], dtype=np.float32)
    translation = np.asarray(pose["cam_t_m2c"], dtype=np.float32)
    if float(np.linalg.norm(translation)) > 10.0:
        translation /= 1000.0
    return np.concatenate((rotation, translation / 2.0)).astype(np.float32)


class DTLDContactHeightDataset:
    """Instance crops for contact-line perception and metric liquid-height regression."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: tuple[int, int] = (320, 180),
        max_depth_m: float = 3.0,
        augment: bool = False,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.image_size = image_size
        self.max_depth_m = float(max_depth_m)
        self.augment = augment
        with self.manifest.open("r", encoding="utf-8", newline="") as stream:
            self.rows = [row for row in csv.DictReader(stream) if row["split"].strip() == split]
        if not self.rows:
            raise ValueError(f"No DTLD rows for split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        rgb = _read_image(row["rgb_path"], cv2.IMREAD_COLOR)
        depth = _read_image(row["raw_depth_path"], cv2.IMREAD_UNCHANGED)
        crop = _instance_bbox(row, rgb.shape[:2])
        if self.augment:
            crop = _jitter_crop(crop, rgb.shape[:2])
        x0, y0, x1, y1 = crop
        width, height = self.image_size
        rgb = cv2.resize(rgb[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(
            depth[y0:y1, x0:x1],
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)
        depth *= float(row["depth_scale_to_m"])
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
        if self.augment:
            rgb, depth = _augment_non_lambertian(rgb, depth)
        valid = ((depth > 0) & (depth <= self.max_depth_m)).astype(np.float32)
        rgb_unit = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb_unit - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        inputs = np.concatenate(
            (
                rgb,
                np.clip(depth / self.max_depth_m, 0.0, 1.0)[..., None],
                valid[..., None],
            ),
            axis=2,
        )
        label = json.loads(row["liquid_label_json"])
        points = _liquid_lane(label, row["object_id"])
        contact = _contact_target(points, crop, self.image_size)
        bezier = _bezier_target(points, crop, self.image_size)
        support = contact > 0.05
        color_residual = np.zeros_like(rgb_unit)
        if np.any(support):
            mean_color = rgb_unit[support].mean(axis=0)
            color_residual[support] = mean_color - rgb_unit[support]
        target = {
            "contact": torch.from_numpy(contact[None]).float(),
            "bezier_control_points": torch.from_numpy(bezier).float(),
            "color_residual": torch.from_numpy(color_residual.transpose(2, 0, 1)).float(),
            "height_mm": torch.tensor(float(row["liquid_height_mm"]), dtype=torch.float32),
            "object_index": torch.tensor(
                DTLD_OBJECT_INDEX[row["object_id"]],
                dtype=torch.long,
            ),
            "pose": torch.from_numpy(_pose_features(row)).float(),
            "row_index": torch.tensor(index, dtype=torch.long),
        }
        return torch.from_numpy(inputs.transpose(2, 0, 1)).float(), target


class DTLDContactHeightNet(nn.Module):
    """RGB-D contact-line network with pose-conditioned metric-height output."""

    def __init__(
        self,
        base_channels: int = 24,
        max_height_mm: float = 120.0,
    ) -> None:
        super().__init__()
        channels = base_channels
        self.max_height_mm = float(max_height_mm)
        self.enc1 = ConvBlock(5, channels)
        self.enc2 = ConvBlock(channels, channels * 2)
        self.enc3 = ConvBlock(channels * 2, channels * 4)
        self.bottleneck = ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(channels * 12, channels * 4)
        self.dec2 = ConvBlock(channels * 6, channels * 2)
        self.dec1 = ConvBlock(channels * 3, channels)
        self.contact_head = nn.Conv2d(channels, 1, 1)
        self.object_embedding = nn.Embedding(4, 8)
        regression_inputs = channels * 8 + 4 + 12 + 8
        self.height_head = nn.Sequential(
            nn.Linear(regression_inputs, channels * 4),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(channels * 4, 2),
        )

    @staticmethod
    def _up(inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _contact_moments(logits: torch.Tensor) -> torch.Tensor:
        probability = logits.sigmoid()
        batch, _, height, width = probability.shape
        x = torch.linspace(-1.0, 1.0, width, device=logits.device, dtype=logits.dtype)
        y = torch.linspace(-1.0, 1.0, height, device=logits.device, dtype=logits.dtype)
        mass = probability.sum(dim=(2, 3)).clamp_min(1e-6)
        mean_x = (probability * x.view(1, 1, 1, width)).sum(dim=(2, 3)) / mass
        mean_y = (probability * y.view(1, 1, height, 1)).sum(dim=(2, 3)) / mass
        var_x = (probability * (x.view(1, 1, 1, width) - mean_x[:, :, None, None]).square()).sum(
            dim=(2, 3)
        ) / mass
        var_y = (probability * (y.view(1, 1, height, 1) - mean_y[:, :, None, None]).square()).sum(
            dim=(2, 3)
        ) / mass
        return torch.cat((mean_x, mean_y, var_x.sqrt(), var_y.sqrt()), dim=1).reshape(batch, 4)

    def forward(
        self,
        inputs: torch.Tensor,
        object_index: torch.Tensor,
        pose: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        features = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat((self._up(features, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        contact_logits = self.contact_head(d1)
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        regression = torch.cat(
            (
                pooled,
                self._contact_moments(contact_logits),
                pose,
                self.object_embedding(object_index),
            ),
            dim=1,
        )
        height_output = self.height_head(regression)
        return {
            "contact_logits": contact_logits,
            "height_mm": height_output[:, 0].sigmoid() * self.max_height_mm,
            "height_log_variance": height_output[:, 1].clamp(-7.0, 5.0),
            "height_confidence": torch.sigmoid(-height_output[:, 1]),
        }


class DTLDContactHeightLoss(nn.Module):
    def __init__(
        self,
        max_height_mm: float = 120.0,
        contact_weight: float = 1.0,
        height_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.max_height_mm = float(max_height_mm)
        self.contact_weight = float(contact_weight)
        self.height_weight = float(height_weight)

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        contact = target["contact"]
        logits = prediction["contact_logits"]
        positive_weight = torch.as_tensor(20.0, device=logits.device, dtype=logits.dtype)
        contact_bce = F.binary_cross_entropy_with_logits(
            logits,
            contact,
            pos_weight=positive_weight,
        )
        probability = logits.sigmoid()
        contact_dice = 1.0 - (2.0 * (probability * contact).sum() + 1.0) / (
            probability.sum() + contact.sum() + 1.0
        )
        contact_loss = contact_bce + contact_dice
        denominator = target["height_mm"].abs().clamp_min(1.0)
        error = (prediction["height_mm"] - target["height_mm"]).abs() / denominator
        log_variance = prediction["height_log_variance"]
        height_loss = (error * torch.exp(-log_variance) + 0.5 * log_variance).mean()
        total = self.contact_weight * contact_loss + self.height_weight * height_loss
        return {
            "total": total,
            "contact": contact_loss,
            "height": height_loss,
        }
