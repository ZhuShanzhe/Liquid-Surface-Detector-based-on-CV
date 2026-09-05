"""Metric sparse plane memory for top-mounted, gravity-calibrated RGB-D cameras.

Coordinates use OpenCV camera axes and a persistent world frame with +Z gravity
opposite. History is evidence about geometry, never independent fresh evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _SurfaceRecord:
    world: np.ndarray
    pixels: np.ndarray
    colors: np.ndarray
    gray: np.ndarray
    frame: int


def world_points(depth, pixels, matrix, camera_to_world):
    rays = np.column_stack(
        (
            (pixels[:, 0] - matrix[0, 2]) / matrix[0, 0],
            (pixels[:, 1] - matrix[1, 2]) / matrix[1, 1],
            np.ones(len(pixels)),
        )
    )
    return (rays * depth[:, None]) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]


def robust_plane(points):
    """Robust affine surface z=ax+by+c, with residual/normal diagnostics."""
    if len(points) < 6 or not np.isfinite(points).all():
        return None
    center = np.median(points[:, :2], axis=0)
    design = np.column_stack((points[:, :2] - center, np.ones(len(points))))
    keep = np.ones(len(points), dtype=bool)
    for _ in range(4):
        if keep.sum() < 6:
            return None
        coef, _, rank, _ = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)
        if rank < 3:
            return None
        residual = points[:, 2] - design @ coef
        scale = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        keep = np.abs(residual) <= max(0.004, 3 * scale)
    return {
        "coef": coef,
        "center": center,
        "keep": keep,
        "residual_m": float(np.median(np.abs(residual[keep]))),
        "tilt_deg": float(np.degrees(np.arctan(np.linalg.norm(coef[:2])))),
        "level_world_m": float(coef[2]),
    }


class MetricSurfaceMemory:
    def __init__(
        self,
        *,
        max_age_frames=45,
        max_history_frames=4,
        min_current_points=24,
        min_current_tiles=4,
        max_memory_fraction=0.75,
        max_shift_m=0.03,
        max_plane_residual_m=0.012,
        min_model_confidence=0.3,
        max_accepted_step_m=0.015,
        range_calibration=None,
    ):
        if not 0.0 <= max_memory_fraction < 1.0:
            raise ValueError("max_memory_fraction must be in [0, 1)")
        if (
            min_current_points < 6
            or min_current_tiles < 2
            or max_age_frames < 1
            or max_history_frames < 1
            or max_accepted_step_m <= 0
        ):
            raise ValueError("Invalid support, age or step constraints")
        self.max_age_frames = max_age_frames
        self.max_history_frames = max_history_frames
        self.min_current_points = min_current_points
        self.min_current_tiles = min_current_tiles
        self.max_memory_fraction = max_memory_fraction
        self.max_shift_m = max_shift_m
        self.max_plane_residual_m = max_plane_residual_m
        self.min_model_confidence = min_model_confidence
        self.records = []
        self.last_level_m = None
        self.max_accepted_step_m = max_accepted_step_m
        self.range_calibration = range_calibration
        self.frame = 0

    def reset(self, *, clear_reference=True):
        self.records.clear()
        if clear_reference:
            self.last_level_m = None

    def estimate(
        self,
        rgb_bgr,
        raw_depth_m,
        prediction,
        matrix,
        camera_to_world_cv,
        bottom_world_m,
        *,
        use_memory=True,
        pose_valid=True,
        guard_jumps=True,
    ):
        self.frame += 1
        self.records = [r for r in self.records if self.frame - r.frame <= self.max_age_frames]
        diag = {
            "accepted": False,
            "level_m": None,
            "reasons": [],
            "fresh_points": 0,
            "history_points": 0,
            "memory_fraction": 0.0,
            "reset": False,
        }
        if not pose_valid or not np.isfinite(camera_to_world_cv).all():
            self.reset()
            diag["reasons"] = ["invalid_pose"]
            return diag
        mask = np.asarray(prediction["mask"], bool)
        conf = np.asarray(prediction["confidence"])
        # Erosion suppresses meniscus/wall leakage. No true masks are consulted.
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        valid = interior & (conf >= self.min_model_confidence) & np.isfinite(raw_depth_m) & (raw_depth_m > 0)
        plane_gate = self.max_plane_residual_m
        if self.range_calibration is not None:
            valid, range_diag = self.range_calibration.select(raw_depth_m, interior, conf)
            diag.update(range_diag)
            plane_gate = range_diag["plane_gate_m"]
            if not range_diag["range_calibration_available"]:
                diag["reasons"] = ["range_calibration_unavailable"]
                return diag
        yy, xx = np.nonzero(valid)
        if len(xx) > 512:
            take = np.linspace(0, len(xx) - 1, 512).astype(int)
            yy, xx = yy[take], xx[take]
        pixels = np.column_stack((xx, yy)).astype(np.float32)
        fresh = world_points(raw_depth_m[yy, xx], pixels, matrix, camera_to_world_cv)
        diag["fresh_points"] = len(fresh)
        if len(fresh) < self.min_current_points:
            diag["reasons"] = ["insufficient_fresh_metric_anchors"]
            return diag
        # Bin within predicted surface bounds instead of the whole image.
        my, mx = np.nonzero(interior)
        bins = np.clip(((xx - mx.min()) * 4 / max(np.ptp(mx) + 1, 1)).astype(int), 0, 3)
        bins += 4 * np.clip(((yy - my.min()) * 4 / max(np.ptp(my) + 1, 1)).astype(int), 0, 3)
        diag["fresh_tiles"] = len(np.unique(bins))
        required_tiles = 2 if use_memory and self.records else self.min_current_tiles
        if diag["fresh_tiles"] < required_tiles:
            diag["reasons"] = ["insufficient_fresh_spatial_support"]
            return diag
        plane = robust_plane(fresh)
        if plane is None:
            diag["reasons"] = ["degenerate_surface_fit"]
            return diag
        keep = plane["keep"]
        diag["plane_residual_m"] = plane["residual_m"]
        diag["plane_gate_m"] = plane_gate
        diag["normalized_plane_residual"] = plane["residual_m"] / plane_gate
        diag["tilt_deg"] = plane["tilt_deg"]
        inlier_fraction = float(np.mean(keep))
        fresh, pixels = fresh[keep], pixels[keep]
        diag["fresh_inliers"] = len(fresh)
        if (
            len(fresh) < self.min_current_points
            or plane["residual_m"] > plane_gate
            or (self.range_calibration is not None and inlier_fraction < 0.60)
            or plane["tilt_deg"] > 12
        ):
            diag["reasons"] = ["inconsistent_metric_surface"]
            return diag
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB).astype(float)
        history = []
        shifts = []
        needs_history = use_memory and diag["fresh_tiles"] < self.min_current_tiles
        diag["memory_activated"] = needs_history
        for record in self.records if needs_history else []:
            # Evaluate old surface at current XY. Current observations anchor
            # liquid-height change; memory must not retain the old height.
            old = robust_plane(record.world)
            if old is None:
                continue
            old_z = (fresh[:, :2] - old["center"]) @ old["coef"][:2] + old["coef"][2]
            delta = float(np.median(fresh[:, 2] - old_z))
            shifts.append(delta)
            if abs(delta) > self.max_shift_m:
                continue
            moved = record.world.copy()
            moved[:, 2] += delta
            cam = (moved - camera_to_world_cv[:3, 3]) @ camera_to_world_cv[:3, :3]
            z = cam[:, 2]
            uv = cam[:, :2] / np.maximum(z[:, None], 1e-6)
            uv = uv * np.array([matrix[0, 0], matrix[1, 1]]) + matrix[:2, 2]
            inside = (z > 0) & np.isfinite(uv).all(axis=1)
            inside &= (
                (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
            )
            ix = np.clip(np.rint(uv[:, 0]).astype(int), 0, mask.shape[1] - 1)
            iy = np.clip(np.rint(uv[:, 1]).astype(int), 0, mask.shape[0] - 1)
            similarity = np.exp(-np.sum((lab[iy, ix] - record.colors) ** 2, axis=1) / (2 * 32.0**2))
            inside &= interior[iy, ix] & (similarity > 0.5)
            if record.gray.shape == gray.shape:
                flow, status, _ = cv2.calcOpticalFlowPyrLK(
                    record.gray, gray, record.pixels.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3
                )
                if flow is not None:
                    agreement = np.linalg.norm(flow.reshape(-1, 2) - uv, axis=1)
                    inside &= ~status.ravel().astype(bool) | (agreement < 6.0)
            history.extend(moved[inside])
        if shifts and min(abs(x) for x in shifts) > self.max_shift_m:
            self.reset(clear_reference=False)
            diag["reset"] = True
        merged = fresh
        if history:
            historical = np.asarray(history)
            # Keep one vote per XY cell; history length cannot inflate support.
            cells = np.rint(historical[:, :2] / 0.015).astype(int)
            _, unique = np.unique(cells, axis=0, return_index=True)
            historical = historical[np.sort(unique)]
            cap = int(len(fresh) * self.max_memory_fraction / (1 - self.max_memory_fraction))
            historical = historical[:cap]
            merged = np.vstack((fresh, historical))
            diag["history_points"] = len(historical)
            diag["memory_fraction"] = len(historical) / len(merged)
        final = robust_plane(merged)
        if final is None:
            diag["reasons"] = ["degenerate_fused_plane"]
            return diag
        if diag["fresh_tiles"] < self.min_current_tiles and diag["history_points"] < 64:
            diag["reasons"] = ["insufficient_independent_history_support"]
            return diag
        # Evaluate at the current support center, so shifting spatial coverage
        # does not change the reference location on a tilted candidate.
        z = float((plane["center"] - final["center"]) @ final["coef"][:2] + final["coef"][2])
        level = z - bottom_world_m
        if (
            guard_jumps
            and self.last_level_m is not None
            and abs(level - self.last_level_m) > self.max_accepted_step_m
        ):
            diag["reasons"] = ["unverified_liquid_level_jump"]
            diag["candidate_level_m"] = level
            return diag
        diag.update(
            level_m=level,
            accepted=bool(level > 0),
            plane_residual_m=final["residual_m"],
            tilt_deg=final["tilt_deg"],
            level_shift_m=float(np.median(shifts)) if shifts else None,
        )
        if level <= 0:
            diag["reasons"] = ["nonpositive_liquid_depth"]
        if diag["accepted"]:
            self.last_level_m = level
        if diag["accepted"] and diag["fresh_tiles"] >= self.min_current_tiles:
            idx = np.rint(pixels).astype(int)
            self.records.append(
                _SurfaceRecord(
                    fresh.copy(), pixels.copy(), lab[idx[:, 1], idx[:, 0]].copy(), gray.copy(), self.frame
                )
            )
            self.records = self.records[-self.max_history_frames :]
        return diag
