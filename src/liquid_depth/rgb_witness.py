"""Depth-independent RGB silhouette witness for calibrated elliptical vessels."""

from __future__ import annotations

import cv2
import numpy as np


class RGBContourWitness:
    def __init__(self, *, calibration_error_m=0.001):
        if not np.isfinite(calibration_error_m) or calibration_error_m < 0:
            raise ValueError("Invalid metric annotation error allowance")
        self.calibration_error_m = float(calibration_error_m)
        self.ready = False

    def _project_mask(self, level, matrix, pose, shape):
        angle = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        world = np.column_stack(
            (self.rx * np.cos(angle), self.ry * np.sin(angle), np.full(len(angle), self.bottom + level))
        )
        camera = (world - pose[:3, 3]) @ pose[:3, :3]
        if np.any(camera[:, 2] <= 0):
            return np.zeros(shape, np.uint8)
        uv = camera[:, :2] / camera[:, 2:3]
        uv = uv * np.array([matrix[0, 0], matrix[1, 1]]) + matrix[:2, 2]
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(uv).astype(np.int32)], 1)
        return mask

    def _segment(self, rgb, matrix, pose):
        lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(float)
        distance = np.sum(((lab - self.color) / self.spread) ** 2, axis=2)
        mask = ((distance <= self.color_cutoff) & (lab[..., 0] >= self.min_luminance)).astype(np.uint8)
        # This ROI comes from calibrated container geometry, not current depth.
        roi = self._project_mask(self.reference + self.search_span, matrix, pose, mask.shape)
        roi = cv2.dilate(roi, np.ones((9, 9), np.uint8))
        mask &= roi
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return np.zeros_like(mask)
        chosen = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return (labels == chosen).astype(np.uint8)

    def _fit(self, mask, matrix, pose):
        levels = self.reference + np.linspace(-self.search_span, self.search_span, 101)
        values = []
        for level in levels:
            candidate = self._project_mask(level, matrix, pose, mask.shape)
            intersection = np.count_nonzero(candidate & mask)
            union = np.count_nonzero(candidate | mask)
            values.append(intersection / max(union, 1))
        values = np.asarray(values)
        index = int(np.argmax(values))
        near = levels[values >= values[index] - 0.005]
        sigma = max(self.search_span / 100.0, float(np.ptp(near)) / 2.0)
        return float(levels[index]), float(values[index]), sigma, index in (0, len(levels) - 1)

    def calibrate(
        self,
        rgb_bgr,
        annotated_surface_mask,
        known_level_m,
        matrix,
        camera_to_world_cv,
        bottom_world_m,
        radius_x_m,
        radius_y_m,
    ):
        """The mask and height are operator-provided calibration annotations."""
        self.rx, self.ry = radius_x_m, radius_y_m
        self.bottom = bottom_world_m
        self.reference = known_level_m
        center = np.array([0.0, 0.0, bottom_world_m + known_level_m])
        distance = np.linalg.norm(camera_to_world_cv[:3, 3] - center)
        self.search_span = 0.15 * distance
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB).astype(float)
        distance_inside = cv2.distanceTransform(annotated_surface_mask.astype(np.uint8), cv2.DIST_L2, 3)
        interior = distance_inside >= 0.5 * distance_inside.max()
        if interior.sum() < 64:
            raise ValueError("Insufficient annotated surface for RGB calibration")
        values = lab[interior]
        self.color = np.median(values, axis=0)
        self.min_luminance = float(np.percentile(values[:, 0], 5) - 15)
        self.spread = np.maximum(np.std(values, axis=0), [12.0, 3.0, 3.0])
        self.color_cutoff = float(
            np.percentile(np.sum(((values - self.color) / self.spread) ** 2, axis=1), 99)
        )
        self.color_cutoff = max(self.color_cutoff, 3.0)
        segmented = self._segment(rgb_bgr, matrix, camera_to_world_cv)
        measured, score, _, boundary = self._fit(segmented, matrix, camera_to_world_cv)
        if score < 0.65 or boundary:
            raise ValueError("RGB silhouette calibration is unobservable")
        self.bias = measured - known_level_m
        self.minimum_iou = max(0.65, score - 0.12)
        self.ready = True

    def estimate(self, rgb_bgr, matrix, camera_to_world_cv, *, resolution_checks=False, source_pixel_scale=1):
        if not np.isfinite(source_pixel_scale) or source_pixel_scale < 1 or source_pixel_scale > 8:
            raise ValueError("Source pixel scale must be in [1, 8]")
        if not self.ready:
            return {"available": False, "reason": "rgb_reference_not_calibrated"}
        mask = self._segment(rgb_bgr, matrix, camera_to_world_cv)
        value, score, sigma, boundary = self._fit(mask, matrix, camera_to_world_cv)
        available = score >= self.minimum_iou and not boundary
        error_bound = None
        resolution_boundary = False
        if resolution_checks and available:
            # One-pixel contour ambiguity, plus fit plateau and a 1 mm
            # calibration annotation allowance. This is not a statistical CI.
            bounds = [2 * sigma]
            for operation in (cv2.erode, cv2.dilate):
                radius = int(np.ceil(source_pixel_scale))
                # SR interpolation must retain the original sampling ambiguity.
                perturbed = operation(mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
                alternative, _, _, edge = self._fit(perturbed, matrix, camera_to_world_cv)
                bounds.append(abs(alternative - value))
                resolution_boundary |= edge
            error_bound = float(max(bounds) + self.calibration_error_m)
            available = available and not resolution_boundary
        return {
            "available": available,
            "level_m": value - self.bias if available else None,
            "uncertainty_proxy_m": sigma,
            "resolution_checked": resolution_checks,
            "error_bound_proxy_m": error_bound,
            "bound_is_statistically_calibrated": False,
            "iou": score,
            "reason": None if available else "rgb_contour_unobservable",
            "source": "rgb_contour_and_calibrated_vessel",
            "depth_input_used": False,
        }

    def to_dict(self):
        if not self.ready:
            raise ValueError("RGB witness is not calibrated")
        keys = (
            "rx",
            "ry",
            "bottom",
            "reference",
            "search_span",
            "color_cutoff",
            "min_luminance",
            "bias",
            "minimum_iou",
        )
        return {
            "schema_version": 1,
            "calibration_error_m": self.calibration_error_m,
            **{key: float(getattr(self, key)) for key in keys},
            "color": self.color.tolist(),
            "spread": self.spread.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported RGB witness calibration")
        witness = cls(calibration_error_m=payload.get("calibration_error_m", 0.001))
        keys = (
            "rx",
            "ry",
            "bottom",
            "reference",
            "search_span",
            "color_cutoff",
            "min_luminance",
            "bias",
            "minimum_iou",
        )
        for key in keys:
            value = float(payload[key])
            if not np.isfinite(value):
                raise ValueError("Nonfinite RGB calibration")
            setattr(witness, key, value)
        witness.color = np.asarray(payload["color"], dtype=float)
        witness.spread = np.asarray(payload["spread"], dtype=float)
        if (
            witness.color.shape != (3,)
            or witness.spread.shape != (3,)
            or not np.isfinite(witness.color).all()
            or not np.isfinite(witness.spread).all()
        ):
            raise ValueError("Invalid RGB color model")
        if min(
            witness.rx, witness.ry, witness.search_span, witness.reference, witness.color_cutoff
        ) <= 0 or np.any(witness.spread <= 0):
            raise ValueError("Invalid RGB calibration dimensions")
        if not 0 <= witness.minimum_iou <= 1:
            raise ValueError("Invalid RGB calibration overlap")
        witness.ready = True
        return witness
