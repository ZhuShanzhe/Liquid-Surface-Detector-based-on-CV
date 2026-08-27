from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    source: str
    rms_reprojection_px: float | None = None
    views_used: int | None = None

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.reshape(-1).tolist(),
            "image_size": list(self.image_size),
            "source": self.source,
            "rms_reprojection_px": self.rms_reprojection_px,
            "views_used": self.views_used,
        }

    @classmethod
    def from_dict(cls, value: dict) -> CameraCalibration:
        image_size = tuple(int(item) for item in value["image_size"])
        if len(image_size) != 2:
            raise ValueError("image_size must contain width and height")
        return cls(
            np.asarray(value["camera_matrix"], dtype=np.float64).reshape(3, 3),
            np.asarray(value.get("distortion", []), dtype=np.float64).reshape(-1),
            image_size,
            str(value.get("source", "unknown")),
            value.get("rms_reprojection_px"),
            value.get("views_used"),
        )


@dataclass(frozen=True)
class PoseEstimate:
    rotation_m2c: np.ndarray
    translation_m2c_m: np.ndarray
    reprojection_rmse_px: float
    source: str
    marker_corners_px: np.ndarray | None = None

    def to_dict(self) -> dict:
        payload = {
            "rotation_m2c": self.rotation_m2c.tolist(),
            "translation_m2c_m": self.translation_m2c_m.tolist(),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "source": self.source,
        }
        if self.marker_corners_px is not None:
            payload["marker_corners_px"] = self.marker_corners_px.tolist()
        return payload


def save_camera_calibration(path: str | Path, calibration: CameraCalibration) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(calibration.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    return CameraCalibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def import_factory_calibration(frame_dir: str | Path) -> CameraCalibration:
    source = Path(frame_dir).expanduser().resolve()
    rgb = cv2.imread(str(source / "rgb.png"), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(source / "rgb.png")
    for filename in ("depth_info.json", "color_info.json"):
        path = source / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        width = int(payload.get("width") or rgb.shape[1])
        height = int(payload.get("height") or rgb.shape[0])
        if (height, width) != rgb.shape[:2]:
            continue
        matrix = np.asarray(payload["K"], dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            continue
        return CameraCalibration(
            matrix,
            np.asarray(payload.get("D", []), dtype=np.float64),
            (width, height),
            f"factory_{path.stem}",
        )
    raise ValueError("No camera-info JSON matches the RGB frame. Enable depth registration and recapture.")


def calibrate_checkerboard(
    image_paths: Iterable[str | Path],
    pattern_size: tuple[int, int],
    square_size_m: float,
    *,
    min_views: int = 8,
) -> tuple[CameraCalibration, list[dict]]:
    if min(pattern_size) < 2 or square_size_m <= 0:
        raise ValueError("Invalid checkerboard dimensions or square size")
    template = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    template[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2) * square_size_m
    objects, images, records = [], [], []
    image_size = None
    for raw_path in image_paths:
        path = Path(raw_path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            records.append({"path": str(path), "accepted": False, "reason": "decode_failed"})
            continue
        current_size = (image.shape[1], image.shape[0])
        image_size = image_size or current_size
        if current_size != image_size:
            records.append({"path": str(path), "accepted": False, "reason": "size_mismatch"})
            continue
        found, corners = cv2.findChessboardCornersSB(
            image,
            pattern_size,
            flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        if not found:
            records.append({"path": str(path), "accepted": False, "reason": "pattern_not_found"})
            continue
        objects.append(template.copy())
        images.append(corners.astype(np.float32))
        records.append({"path": str(path), "accepted": True})
    if image_size is None or len(objects) < min_views:
        raise ValueError(f"Checkerboard calibration needs {min_views} accepted views; got {len(objects)}")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        objects,
        images,
        image_size,
        None,
        None,
    )
    accepted_records = (item for item in records if item["accepted"])
    for record, object_points, image_points, rotation, translation in zip(
        accepted_records,
        objects,
        images,
        rotations,
        translations,
        strict=True,
    ):
        projected, _ = cv2.projectPoints(
            object_points,
            rotation,
            translation,
            matrix,
            distortion,
        )
        residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        record["reprojection_rmse_px"] = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return (
        CameraCalibration(
            matrix,
            distortion.reshape(-1),
            image_size,
            "checkerboard",
            float(rms),
            len(objects),
        ),
        records,
    )


def euler_xyz_degrees_to_matrix(values: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (math.radians(float(item)) for item in values)
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def detect_aruco_marker_pose(
    image_bgr: np.ndarray,
    calibration: CameraCalibration,
    *,
    marker_id: int,
    marker_size_m: float,
    dictionary_name: str = "DICT_4X4_50",
) -> PoseEstimate:
    if marker_size_m <= 0:
        raise ValueError("marker_size_m must be positive")
    dictionary_code = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_code is None:
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_code)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, identifiers, _ = detector.detectMarkers(image_bgr)
    if identifiers is None:
        raise ValueError("aruco_marker_not_found")
    matches = np.flatnonzero(identifiers.reshape(-1) == int(marker_id))
    if len(matches) != 1:
        raise ValueError(f"aruco_marker_{marker_id}_not_found")
    image_points = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
    half = marker_size_m / 2.0
    object_points = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float64,
    )
    success, rvec, translation = cv2.solvePnP(
        object_points,
        image_points,
        calibration.camera_matrix,
        calibration.distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        raise ValueError("aruco_pose_failed")
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        translation,
        calibration.camera_matrix,
        calibration.distortion,
    )
    residual = projected.reshape(-1, 2) - image_points
    return PoseEstimate(
        rotation,
        translation.reshape(3),
        float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        f"aruco:{dictionary_name}:{marker_id}",
        image_points,
    )


def compose_container_pose(
    marker_pose: PoseEstimate,
    container_to_marker: np.ndarray,
) -> PoseEstimate:
    marker_to_camera = make_transform(
        marker_pose.rotation_m2c,
        marker_pose.translation_m2c_m,
    )
    container_to_camera = marker_to_camera @ np.asarray(
        container_to_marker,
        dtype=np.float64,
    ).reshape(4, 4)
    return PoseEstimate(
        container_to_camera[:3, :3],
        container_to_camera[:3, 3],
        marker_pose.reprojection_rmse_px,
        marker_pose.source + "+container_fixture",
        marker_pose.marker_corners_px,
    )


def solve_container_pose_from_correspondences(
    model_points_m: np.ndarray,
    image_points_px: np.ndarray,
    calibration: CameraCalibration,
) -> PoseEstimate:
    model_points = np.asarray(model_points_m, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    if len(model_points) != len(image_points) or len(model_points) < 6:
        raise ValueError("PnP calibration requires at least six 3-D/2-D pairs")
    success, rvec, translation, inliers = cv2.solvePnPRansac(
        model_points,
        image_points,
        calibration.camera_matrix,
        calibration.distortion,
        iterationsCount=1000,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 6:
        raise ValueError("container_pnp_failed")
    selected = inliers[:, 0]
    success, rvec, translation = cv2.solvePnP(
        model_points[selected],
        image_points[selected],
        calibration.camera_matrix,
        calibration.distortion,
        rvec,
        translation,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("container_pnp_refinement_failed")
    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(
        model_points[selected],
        rvec,
        translation,
        calibration.camera_matrix,
        calibration.distortion,
    )
    residual = projected.reshape(-1, 2) - image_points[selected]
    return PoseEstimate(
        rotation,
        translation.reshape(3),
        float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        f"pnp:{len(selected)}/{len(model_points)}",
    )


def fit_output_calibration(
    predicted_depth_m: Iterable[float],
    known_depth_m: Iterable[float],
) -> dict:
    predicted = np.asarray(tuple(predicted_depth_m), dtype=np.float64)
    known = np.asarray(tuple(known_depth_m), dtype=np.float64)
    if predicted.shape != known.shape or predicted.ndim != 1 or len(predicted) < 3:
        raise ValueError("Output calibration requires at least three predicted/known pairs")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(known)):
        raise ValueError("Output calibration contains non-finite values")
    if float(np.ptp(known)) < 0.02:
        raise ValueError("Known calibration depths must span at least 20 mm")
    design = np.column_stack((predicted, np.ones_like(predicted)))
    scale, offset = np.linalg.lstsq(design, known, rcond=None)[0]
    error = scale * predicted + offset - known
    return {
        "scale": float(scale),
        "offset_m": float(offset),
        "samples": len(predicted),
        "known_span_m": float(np.ptp(known)),
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_m": float(np.max(np.abs(error))),
    }
