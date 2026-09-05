from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .container_geometry import ContainerModel, project_model_points


@dataclass(frozen=True)
class AnchorMemoryFusion:
    points_px: np.ndarray
    confidences: np.ndarray
    observed_current: np.ndarray
    accepted: bool
    rejection_reasons: tuple[str, ...]
    current_points: int
    total_reliable_points: int
    recovered_points: int
    occupied_bins: int
    memory_fraction: float
    mean_rgb_similarity: float | None
    mean_alignment_error_px: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "current_reliable_points": self.current_points,
            "total_reliable_points": self.total_reliable_points,
            "recovered_points": self.recovered_points,
            "occupied_horizontal_bins": self.occupied_bins,
            "memory_fraction": self.memory_fraction,
            "mean_rgb_similarity": self.mean_rgb_similarity,
            "mean_alignment_error_px": self.mean_alignment_error_px,
        }


@dataclass
class _AnchorFrame:
    model_points_m: np.ndarray
    image_points_px: np.ndarray
    colors_lab: np.ndarray
    confidence: np.ndarray
    gray: np.ndarray
    age: int = 0


class TemporalAnchorMemory:
    """Guarded long-term memory for directly observed liquid contact points."""

    def __init__(
        self,
        *,
        max_age_frames: int = 45,
        max_history_frames: int = 12,
        min_confidence: float = 0.55,
        confidence_decay: float = 0.94,
        memory_confidence_cap: float = 0.82,
        spatial_match_px: float = 10.0,
        max_model_match_px: float = 8.0,
        flow_window_px: int = 21,
        max_flow_error_px: float = 12.0,
        max_pose_flow_disagreement_px: float = 8.0,
        rgb_sigma_lab: float = 32.0,
        min_rgb_similarity: float = 0.18,
        min_current_points: int = 2,
        min_total_points: int = 6,
        horizontal_bins: int = 8,
        min_occupied_bins: int = 3,
        max_memory_fraction: float = 0.80,
    ) -> None:
        self.max_age_frames = max(1, int(max_age_frames))
        self.max_history_frames = max(1, int(max_history_frames))
        self.min_confidence = float(min_confidence)
        self.confidence_decay = float(confidence_decay)
        self.memory_confidence_cap = float(memory_confidence_cap)
        self.spatial_match_px = float(spatial_match_px)
        self.max_model_match_px = float(max_model_match_px)
        self.flow_window_px = max(5, int(flow_window_px) | 1)
        self.max_flow_error_px = float(max_flow_error_px)
        self.max_pose_flow_disagreement_px = float(max_pose_flow_disagreement_px)
        self.rgb_sigma_lab = float(rgb_sigma_lab)
        self.min_rgb_similarity = float(min_rgb_similarity)
        self.min_current_points = max(0, int(min_current_points))
        self.min_total_points = max(2, int(min_total_points))
        self.horizontal_bins = max(1, int(horizontal_bins))
        self.min_occupied_bins = max(1, int(min_occupied_bins))
        self.max_memory_fraction = float(max_memory_fraction)
        self._frames: list[_AnchorFrame] = []
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0, 1]")
        if not 0.0 < self.confidence_decay <= 1.0:
            raise ValueError("confidence_decay must be within (0, 1]")
        if not 0.0 <= self.max_memory_fraction <= 1.0:
            raise ValueError("max_memory_fraction must be within [0, 1]")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> TemporalAnchorMemory:
        options = dict(config or {})
        options.pop("enabled", None)
        options.pop("activation_valid_ratio_below", None)
        return cls(**options)

    @property
    def history_frames(self) -> int:
        return len(self._frames)

    def reset(self) -> None:
        self._frames.clear()

    def advance(self) -> None:
        for frame in self._frames:
            frame.age += 1
        self._frames = [f for f in self._frames if f.age <= self.max_age_frames]

    @staticmethod
    def _images(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must have shape (H, W, 3)")
        return (
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32),
        )

    @staticmethod
    def _sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        xy = np.rint(points).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        return image[xy[:, 1], xy[:, 0]]

    @staticmethod
    def _project(points_m, matrix, rotation, translation):
        camera = points_m @ rotation.T + translation
        valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-6)
        homogeneous = camera @ matrix.T
        pixels = np.full((len(camera), 2), np.nan, dtype=np.float64)
        pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
        return pixels, valid & np.isfinite(pixels).all(axis=1)

    def _flow(self, record: _AnchorFrame, gray: np.ndarray):
        count = len(record.image_points_px)
        empty = (
            np.full((count, 2), np.nan, np.float32),
            np.zeros(count, dtype=bool),
            np.full(count, np.inf, np.float32),
        )
        if record.gray.shape != gray.shape or count == 0:
            return empty
        target, status, error = cv2.calcOpticalFlowPyrLK(
            record.gray,
            gray,
            record.image_points_px.astype(np.float32).reshape(-1, 1, 2),
            None,
            winSize=(self.flow_window_px, self.flow_window_px),
            maxLevel=3,
        )
        if target is None or status is None:
            return empty
        target = target.reshape(-1, 2)
        error = np.full(count, np.inf) if error is None else error.reshape(-1)
        valid = status.reshape(-1).astype(bool)
        valid &= np.isfinite(target).all(axis=1) & (error <= self.max_flow_error_px)
        return target, valid, error

    def _history(self, image, matrix, rotation, translation):
        gray, lab = self._images(image)
        all_points, all_scores, all_similarities, all_errors = [], [], [], []
        h, w = gray.shape
        for record in self._frames:
            projected, pose_valid = self._project(record.model_points_m, matrix, rotation, translation)
            flowed, flow_valid, _ = self._flow(record, gray)
            chosen = projected.copy()
            error = np.zeros(len(projected), dtype=np.float64)
            disagreement = np.linalg.norm(projected - flowed, axis=1)
            both = pose_valid & flow_valid
            agreed = both & (disagreement <= self.max_pose_flow_disagreement_px)
            conflicted = both & ~agreed
            chosen[agreed] = 0.65 * projected[agreed] + 0.35 * flowed[agreed]
            error[agreed] = disagreement[agreed]
            valid = pose_valid & ~conflicted
            valid &= (chosen[:, 0] >= 0) & (chosen[:, 0] < w)
            valid &= (chosen[:, 1] >= 0) & (chosen[:, 1] < h)
            current_color = self._sample(lab, chosen)
            delta = np.linalg.norm(current_color - record.colors_lab, axis=1)
            similarity = np.exp(-(delta**2) / (2.0 * self.rgb_sigma_lab**2))
            alignment = np.exp(-error / max(self.max_pose_flow_disagreement_px, 1e-6))
            score = np.minimum(
                record.confidence * self.confidence_decay**record.age * similarity * alignment,
                self.memory_confidence_cap,
            )
            valid &= similarity >= self.min_rgb_similarity
            valid &= score >= 0.5 * self.min_confidence
            all_points.append(chosen[valid])
            all_scores.append(score[valid])
            all_similarities.append(similarity[valid])
            all_errors.append(error[valid])
        if not any(len(item) for item in all_points):
            return (np.empty((0, 2)),) + (np.empty(0),) * 3
        return tuple(
            np.concatenate(items, axis=0) for items in (all_points, all_scores, all_similarities, all_errors)
        )

    def _deduplicate_history(self, points, scores, similarities, errors):
        """Keep only the strongest historical vote in each image neighborhood."""
        if len(points) < 2:
            return points, scores, similarities, errors
        order = np.argsort(-scores, kind="stable")
        kept: list[int] = []
        radius = max(1.0, 0.5 * self.spatial_match_px)
        for index in order:
            if kept and np.min(np.linalg.norm(points[np.asarray(kept)] - points[index], axis=1)) < radius:
                continue
            kept.append(int(index))
        indices = np.asarray(kept, dtype=np.int64)
        return (
            points[indices],
            scores[indices],
            similarities[indices],
            errors[indices],
        )

    def fuse(
        self,
        image_bgr: np.ndarray,
        contact_curve_pixels: np.ndarray,
        point_confidences: np.ndarray,
        camera_matrix: np.ndarray,
        rotation_m2c: np.ndarray,
        translation_m2c_m: np.ndarray,
        *,
        roi_xyxy: tuple[int, int, int, int] | None = None,
    ) -> AnchorMemoryFusion:
        self.advance()
        curve = np.asarray(contact_curve_pixels, dtype=np.float64).copy()
        confidence = np.clip(np.asarray(point_confidences).reshape(-1), 0.0, 1.0)
        if curve.ndim != 2 or curve.shape != (len(confidence), 2):
            raise ValueError("contact points/confidences have incompatible shapes")
        observed = confidence >= self.min_confidence
        current_points = int(observed.sum())
        points, scores, similarities, errors = self._history(
            image_bgr,
            np.asarray(camera_matrix, np.float64),
            np.asarray(rotation_m2c, np.float64),
            np.asarray(translation_m2c_m, np.float64),
        )
        points, scores, similarities, errors = self._deduplicate_history(points, scores, similarities, errors)
        for point, score in zip(points, scores, strict=True):
            distance = np.linalg.norm(curve - point, axis=1)
            nearest = int(np.argmin(distance)) if len(distance) else -1
            if nearest >= 0 and distance[nearest] <= self.spatial_match_px:
                total = max(confidence[nearest] + score, 1e-6)
                curve[nearest] = (confidence[nearest] * curve[nearest] + score * point) / total
                confidence[nearest] = 1.0 - (1.0 - confidence[nearest]) * (1.0 - score)
            else:
                curve = np.vstack((curve, point))
                confidence = np.append(confidence, score)
                observed = np.append(observed, False)
        reliable = confidence >= self.min_confidence
        total = int(reliable.sum())
        recovered = int(np.count_nonzero(reliable & ~observed))
        memory_fraction = recovered / max(total, 1)
        x0, x1 = (0.0, float(image_bgr.shape[1]))
        if roi_xyxy is not None:
            x0, _, x1, _ = map(float, roi_xyxy)
        occupied = 0
        if total:
            nx = np.clip((curve[reliable, 0] - x0) / max(x1 - x0, 1.0), 0, 1)
            bins = np.minimum((nx * self.horizontal_bins).astype(int), self.horizontal_bins - 1)
            occupied = len(np.unique(bins))
        reasons = []
        if total < self.min_total_points:
            reasons.append("insufficient_temporal_anchor_points")
        if current_points < self.min_current_points:
            reasons.append("insufficient_current_anchor_observability")
        if occupied < self.min_occupied_bins:
            reasons.append("insufficient_temporal_anchor_spread")
        if memory_fraction > self.max_memory_fraction:
            reasons.append("temporal_memory_fraction_too_high")
        return AnchorMemoryFusion(
            curve,
            confidence,
            observed,
            not reasons,
            tuple(reasons),
            current_points,
            total,
            recovered,
            occupied,
            float(memory_fraction),
            float(similarities.mean()) if len(similarities) else None,
            float(errors.mean()) if len(errors) else None,
        )

    def commit(
        self,
        image_bgr: np.ndarray,
        contact_curve_pixels: np.ndarray,
        point_confidences: np.ndarray,
        model: ContainerModel,
        camera_matrix: np.ndarray,
        rotation_m2c: np.ndarray,
        translation_m2c_m: np.ndarray,
    ) -> int:
        curve = np.asarray(contact_curve_pixels, dtype=np.float64)
        confidence = np.asarray(point_confidences).reshape(-1)
        keep = np.isfinite(curve).all(axis=1) & (confidence >= self.min_confidence)
        curve, confidence = curve[keep], confidence[keep]
        if len(curve) < 2:
            return 0
        projected, _, camera_points = project_model_points(
            model, camera_matrix, rotation_m2c, translation_m2c_m
        )
        squared = np.sum((curve[:, None] - projected[None, :]) ** 2, axis=2)
        nearest = np.argmin(squared, axis=1)
        keep = np.sqrt(squared[np.arange(len(curve)), nearest]) <= self.max_model_match_px
        if np.count_nonzero(keep) < 2:
            return 0
        curve, confidence, nearest = curve[keep], confidence[keep], nearest[keep]
        model_points = (camera_points[nearest] - translation_m2c_m) @ rotation_m2c
        gray, lab = self._images(image_bgr)
        self._frames.append(
            _AnchorFrame(
                model_points,
                curve.copy(),
                self._sample(lab, curve),
                confidence.copy(),
                gray.copy(),
            )
        )
        self._frames = self._frames[-self.max_history_frames :]
        return len(curve)
