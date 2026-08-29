from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import (
    CameraCalibration,
    calibrate_checkerboard,
    compose_container_pose,
    detect_aruco_marker_pose,
    euler_xyz_degrees_to_matrix,
    fit_output_calibration,
    import_factory_calibration,
    make_transform,
    solve_container_pose_from_correspondences,
)
from .container_geometry import load_container_model, project_model_points
from .io import write_json
from .rail_runtime import make_product_system
from .system_runtime import load_system_profile, save_system_profile


def _csv_floats(value: str, count: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if len(result) != count:
        raise argparse.ArgumentTypeError(f"{name} requires {count} comma-separated values")
    return result


def _triple(value: str) -> tuple[float, float, float]:
    return _csv_floats(value, 3, "three-vector")


def _quad(value: str) -> tuple[int, int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 4:
        raise argparse.ArgumentTypeError("crop requires x0,y0,x1,y1")
    return result


def _pattern(value: str) -> tuple[int, int]:
    values = value.lower().split("x")
    if len(values) != 2:
        raise argparse.ArgumentTypeError("pattern requires COLSxROWS")
    return int(values[0]), int(values[1])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="liquid-depth-system",
        description="Deploy, calibrate, and run the packaged RGB-D liquid-depth system",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    marker = commands.add_parser("make-marker", help="Generate a printable ArUco marker")
    marker.add_argument("--id", type=int, default=0)
    marker.add_argument("--dictionary", default="DICT_4X4_50")
    marker.add_argument("--pixels", type=int, default=1200)
    marker.add_argument("--physical-size-mm", type=float, default=60.0)
    marker.add_argument("--output", type=Path, required=True)

    factory = commands.add_parser(
        "camera-from-frame",
        help="Import factory intrinsics from one synchronized RGB-D frame",
    )
    factory.add_argument("--frame", type=Path, required=True)
    factory.add_argument("--depth-scale-to-m", type=float, default=0.001)
    factory.add_argument("--output", type=Path, required=True)

    checkerboard = commands.add_parser(
        "camera-checkerboard",
        help="Recalibrate RGB intrinsics from checkerboard images",
    )
    checkerboard.add_argument("--images", type=Path, required=True)
    checkerboard.add_argument("--glob", default="*.png")
    checkerboard.add_argument("--pattern", type=_pattern, default=(9, 6))
    checkerboard.add_argument("--square-size-mm", type=float, required=True)
    checkerboard.add_argument("--min-views", type=int, default=8)
    checkerboard.add_argument("--depth-scale-to-m", type=float, default=0.001)
    checkerboard.add_argument("--output", type=Path, required=True)

    setup = commands.add_parser(
        "setup",
        help="Create a complete fixed or marker-tracked deployment profile",
    )
    setup.add_argument("--frame", type=Path, required=True)
    setup.add_argument("--camera-json", type=Path)
    setup.add_argument("--container-model", type=Path, required=True)
    setup.add_argument("--checkpoint", type=Path, required=True)
    setup.add_argument("--object-index", type=int, choices=range(4), required=True)
    setup.add_argument("--mode", choices=("fixed", "marker_tracking"), default="fixed")
    setup.add_argument("--marker-id", type=int, default=0)
    setup.add_argument("--marker-size-mm", type=float, default=60.0)
    setup.add_argument("--marker-dictionary", default="DICT_4X4_50")
    setup.add_argument(
        "--container-origin-in-marker-mm",
        type=_triple,
        default=(0.0, 0.0, 0.0),
    )
    setup.add_argument(
        "--container-rpy-in-marker-deg",
        type=_triple,
        default=(0.0, 0.0, 0.0),
    )
    setup.add_argument(
        "--correspondences-json",
        type=Path,
        help="Alternative to ArUco: JSON with model_points_m and image_points_px",
    )
    setup.add_argument("--level-axis", type=_triple, required=True)
    setup.add_argument("--level-origin-m", type=_triple, required=True)
    setup.add_argument("--crop-xyxy", type=_quad)
    setup.add_argument("--depth-scale-to-m", type=float, default=0.001)
    setup.add_argument("--max-pose-rmse-px", type=float, default=2.5)
    setup.add_argument("--output", type=Path, required=True)

    measure = commands.add_parser("measure", help="Measure one captured RGB-D frame")
    measure.add_argument("--profile", type=Path, required=True)
    measure.add_argument("--frame", type=Path, required=True)
    measure.add_argument("--output-dir", type=Path, required=True)
    measure.add_argument("--device")

    calibrate_output = commands.add_parser(
        "calibrate-output",
        help="Fit final scale/offset from at least three known liquid depths",
    )
    calibrate_output.add_argument("--profile", type=Path, required=True)
    calibrate_output.add_argument("--samples", type=Path, required=True)
    calibrate_output.add_argument("--device")
    calibrate_output.add_argument("--report", type=Path)

    watch = commands.add_parser(
        "watch",
        help="Continuously measure complete frame directories from the camera node",
    )
    watch.add_argument("--profile", type=Path, required=True)
    watch.add_argument("--input-dir", type=Path, required=True)
    watch.add_argument("--output-dir", type=Path, required=True)
    watch.add_argument("--device")
    watch.add_argument("--poll-seconds", type=float, default=0.25)
    watch.add_argument("--once", action="store_true")
    return parser


def _make_marker(args) -> None:
    code = getattr(cv2.aruco, args.dictionary, None)
    if code is None:
        raise ValueError(f"Unsupported ArUco dictionary: {args.dictionary}")
    if args.pixels < 200 or args.physical_size_mm <= 0:
        raise ValueError("Marker must be at least 200 pixels and have positive size")
    dictionary = cv2.aruco.getPredefinedDictionary(code)
    marker = cv2.aruco.generateImageMarker(dictionary, args.id, args.pixels, borderBits=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), marker):
        raise OSError(f"Could not write {args.output}")
    metadata = {
        "dictionary": args.dictionary,
        "id": args.id,
        "physical_size_mm": args.physical_size_mm,
        "print_instruction": (
            "Print without page scaling; the outer black square must measure exactly "
            f"{args.physical_size_mm:.3f} mm."
        ),
    }
    write_json(args.output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def _camera_from_frame(args) -> None:
    if args.depth_scale_to_m <= 0:
        raise ValueError("--depth-scale-to-m must be positive")
    calibration = import_factory_calibration(args.frame)
    payload = calibration.to_dict()
    payload["depth_scale_to_m"] = args.depth_scale_to_m
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _camera_checkerboard(args) -> None:
    images = sorted(args.images.glob(args.glob))
    calibration, views = calibrate_checkerboard(
        images,
        args.pattern,
        args.square_size_mm / 1000.0,
        min_views=args.min_views,
    )
    payload = calibration.to_dict()
    payload["depth_scale_to_m"] = args.depth_scale_to_m
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_json(args.output.with_name(args.output.stem + "_views.json"), {"views": views})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_rgb(frame_dir: Path) -> np.ndarray:
    image = cv2.imread(str(frame_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(frame_dir / "rgb.png")
    return image


def _setup(args) -> None:
    if args.depth_scale_to_m <= 0:
        raise ValueError("--depth-scale-to-m must be positive")
    if args.camera_json:
        camera_payload = json.loads(args.camera_json.read_text(encoding="utf-8"))
        calibration = CameraCalibration.from_dict(camera_payload)
        depth_scale = float(camera_payload.get("depth_scale_to_m", args.depth_scale_to_m))
    else:
        calibration = import_factory_calibration(args.frame)
        depth_scale = args.depth_scale_to_m
    rgb = _load_rgb(args.frame)
    marker_config = None
    if args.correspondences_json:
        if args.mode == "marker_tracking":
            raise ValueError("marker_tracking requires an ArUco marker")
        pairs = json.loads(args.correspondences_json.read_text(encoding="utf-8"))
        pose = solve_container_pose_from_correspondences(
            pairs["model_points_m"],
            pairs["image_points_px"],
            calibration,
        )
    else:
        marker_pose = detect_aruco_marker_pose(
            rgb,
            calibration,
            marker_id=args.marker_id,
            marker_size_m=args.marker_size_mm / 1000.0,
            dictionary_name=args.marker_dictionary,
        )
        container_rotation = euler_xyz_degrees_to_matrix(args.container_rpy_in_marker_deg)
        container_translation = np.asarray(args.container_origin_in_marker_mm, dtype=np.float64) / 1000.0
        container_to_marker = make_transform(
            container_rotation,
            container_translation,
        )
        pose = compose_container_pose(marker_pose, container_to_marker)
        marker_config = {
            "dictionary": args.marker_dictionary,
            "id": args.marker_id,
            "size_m": args.marker_size_mm / 1000.0,
            "container_to_marker": container_to_marker.tolist(),
        }
    if pose.reprojection_rmse_px > args.max_pose_rmse_px:
        raise ValueError(
            f"Pose reprojection RMSE {pose.reprojection_rmse_px:.3f} px exceeds "
            f"{args.max_pose_rmse_px:.3f} px"
        )
    model = load_container_model(
        args.container_model,
        args.level_axis,
        args.level_origin_m,
    )
    projected, _, _ = project_model_points(
        model,
        calibration.camera_matrix,
        pose.rotation_m2c,
        pose.translation_m2c_m,
    )
    visible = projected[
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < rgb.shape[1])
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < rgb.shape[0])
    ]
    if len(visible) < 10:
        raise ValueError("Container CAD does not project into the calibration image")
    overlay = rgb.copy()
    for x, y in visible[:: max(1, len(visible) // 3000)]:
        cv2.circle(overlay, (round(x), round(y)), 1, (0, 255, 0), -1)
    if pose.marker_corners_px is not None:
        cv2.polylines(
            overlay,
            [np.rint(pose.marker_corners_px).astype(np.int32)],
            True,
            (0, 0, 255),
            2,
        )
    profile = {
        "schema_version": 1,
        "name": args.output.stem,
        "measurement": {"mode": "cad"},
        "camera": {
            **calibration.to_dict(),
            "depth_scale_to_m": depth_scale,
            "depth_registered_to_color": True,
            "depth_correction": {"scale": 1.0, "offset_m": 0.0, "status": "not_verified"},
        },
        "container": {
            "model_path": str(args.container_model.expanduser().resolve()),
            "level_axis": list(args.level_axis),
            "level_origin_m": list(args.level_origin_m),
        },
        "perception": {
            "checkpoint_path": str(args.checkpoint.expanduser().resolve()),
            "object_index": args.object_index,
            "device": "cuda",
            "crop_xyxy": list(args.crop_xyxy) if args.crop_xyxy else None,
            "crop_margin_ratio": 0.18,
        },
        "pose": {
            "mode": args.mode,
            "rotation_m2c": pose.rotation_m2c.tolist(),
            "translation_m2c_m": pose.translation_m2c_m.tolist(),
            "calibration_reprojection_rmse_px": pose.reprojection_rmse_px,
            "max_reprojection_rmse_px": args.max_pose_rmse_px,
            "marker": marker_config,
        },
        "selection": {
            "min_point_confidence": 0.5,
            "max_selected_points": 24,
            "horizontal_bins": 8,
            "min_reliable_points": 6,
            "min_horizontal_span_ratio": 0.5,
            "min_occupied_bins": 3,
        },
        "geometry": {
            "neighbors": 8,
            "max_reprojection_px": 6.0,
            "max_local_ambiguity_m": 0.015,
            "max_global_spread_m": 0.01,
        },
        "output_calibration": {
            "scale": 1.0,
            "offset_m": 0.0,
            "status": "not_verified",
        },
        "temporal": {
            "process_variance_m2": 1e-6,
            "measurement_variance_m2": 2.5e-5,
            "gate_sigma": 3.5,
            "max_jump_m": 0.02,
            "min_confidence": 0.2,
        },
    }
    save_system_profile(args.output, profile)
    overlay_path = args.output.with_name(args.output.stem + "_pose_overlay.png")
    cv2.imwrite(str(overlay_path), overlay)
    result = {
        "profile": str(args.output.resolve()),
        "pose_overlay": str(overlay_path.resolve()),
        "visible_cad_points": len(visible),
        "pose": pose.to_dict(),
        "next": "Inspect the overlay, then run calibrate-output with 3+ known depths.",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _measure(args) -> None:
    system = make_product_system(args.profile, device=args.device)
    result = system.measure(args.frame, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _calibrate_output(args) -> None:
    with args.samples.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {"frame_dir", "known_depth_mm"} - set(rows[0]):
        raise ValueError("Samples CSV requires frame_dir,known_depth_mm")
    system = make_product_system(args.profile, device=args.device)
    predicted, known, records = [], [], []
    calibration_groups: dict[str, list[tuple[float, float]]] = {}
    root = args.samples.parent
    holdout_roles = {"validation", "holdout", "test"}
    for row in rows:
        frame = Path(row["frame_dir"]).expanduser()
        if not frame.is_absolute():
            frame = root / frame
        role = row.get("role", "calibration").strip().lower() or "calibration"
        known_depth_mm = float(row["known_depth_mm"])
        level_id = row.get("level_id", "").strip() or f"{known_depth_mm:.6f}"
        result = system.measure(frame, apply_output_calibration=False)
        record = {
            "frame_dir": str(frame.resolve()),
            "known_depth_mm": known_depth_mm,
            "level_id": level_id,
            "role": role,
            "accepted": result["accepted"],
            "predicted_depth_m": result["raw_geometry_level_m"],
            "rejection_reasons": result["rejection_reasons"],
        }
        records.append(record)
        if result["accepted"] and result["raw_geometry_level_m"] is not None and role not in holdout_roles:
            calibration_groups.setdefault(level_id, []).append(
                (
                    float(result["raw_geometry_level_m"]),
                    known_depth_mm / 1000.0,
                )
            )
    calibration_levels = []
    minimum_accepted_frames_per_level = 2
    excluded_levels = []
    for level_id, samples in sorted(calibration_groups.items()):
        if len(samples) < minimum_accepted_frames_per_level:
            excluded_levels.append(
                {
                    "level_id": level_id,
                    "accepted_frames": len(samples),
                    "reason": "insufficient_accepted_frames",
                }
            )
            continue
        sample_array = np.asarray(samples, dtype=np.float64)
        predicted_level = float(np.median(sample_array[:, 0]))
        known_level = float(np.median(sample_array[:, 1]))
        predicted.append(predicted_level)
        known.append(known_level)
        calibration_levels.append(
            {
                "level_id": level_id,
                "accepted_frames": len(samples),
                "predicted_median_m": predicted_level,
                "known_depth_m": known_level,
            }
        )
    calibration = fit_output_calibration(predicted, known)
    calibration["levels"] = calibration_levels
    calibration["minimum_accepted_frames_per_level"] = minimum_accepted_frames_per_level
    calibration["excluded_levels"] = excluded_levels

    for record in records:
        raw_depth = record["predicted_depth_m"]
        if not record["accepted"] or raw_depth is None:
            continue
        known_depth_m = float(record["known_depth_mm"]) / 1000.0
        calibrated = calibration["scale"] * float(raw_depth) + calibration["offset_m"]
        absolute_error = abs(calibrated - known_depth_m)
        tolerance = max(0.003, 0.01 * known_depth_m)
        record.update(
            {
                "calibrated_depth_m": calibrated,
                "absolute_error_m": absolute_error,
                "tolerance_m": tolerance,
                "within_tolerance": absolute_error <= tolerance,
            }
        )

    holdout = [
        record for record in records if record["role"] in holdout_roles and "absolute_error_m" in record
    ]
    if holdout:
        errors = np.asarray(
            [record["absolute_error_m"] for record in holdout],
            dtype=np.float64,
        )
        holdout_report = {
            "samples": len(holdout),
            "mae_m": float(errors.mean()),
            "rmse_m": float(np.sqrt(np.mean(errors**2))),
            "max_abs_error_m": float(errors.max()),
            "within_tolerance_rate": float(np.mean([record["within_tolerance"] for record in holdout])),
        }
    else:
        holdout_report = {
            "samples": 0,
            "status": "not_provided",
            "minimum_recommended": 2,
        }
    calibration["holdout_validation"] = holdout_report
    calibration["validation_status"] = "holdout_validated" if len(holdout) >= 2 else "loocv_only"

    profile = load_system_profile(args.profile)
    profile["output_calibration"] = {**calibration, "status": "verified"}
    save_system_profile(args.profile, profile)
    report = {
        "profile": str(args.profile.resolve()),
        "calibration": calibration,
        "records": records,
    }
    report_path = args.report or args.profile.with_name(args.profile.stem + "_output_calibration.json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _complete_frame(path: Path) -> bool:
    return all((path / name).is_file() for name in ("rgb.png", "depth.npy", "depth_info.json"))


def _watch(args) -> None:
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system = make_product_system(args.profile, device=args.device, temporal=True)
    while True:
        processed = 0
        for frame in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
            target = args.output_dir / frame.name
            if (target / "depth_result.json").is_file() or not _complete_frame(frame):
                continue
            try:
                result = system.measure(frame, target)
                status = "accepted" if result["accepted"] else "rejected"
                value = result["liquid_depth_m"]
                shown = "null" if value is None else f"{value * 1000.0:.2f} mm"
                print(
                    f"{frame.name}: {shown}, confidence={result['confidence_uncalibrated']:.3f}, {status}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - keep long-running hardware loop alive
                target.mkdir(parents=True, exist_ok=True)
                write_json(
                    target / "depth_result.json",
                    {
                        "schema_version": 1,
                        "frame_id": frame.name,
                        "accepted": False,
                        "liquid_depth_m": None,
                        "rejection_reasons": ["runtime_error"],
                        "error": str(exc),
                    },
                )
                print(f"{frame.name}: ERROR {exc}", flush=True)
            processed += 1
        if args.once:
            return
        if processed == 0:
            time.sleep(args.poll_seconds)


def main() -> None:
    args = _parser().parse_args()
    actions = {
        "make-marker": _make_marker,
        "camera-from-frame": _camera_from_frame,
        "camera-checkerboard": _camera_checkerboard,
        "setup": _setup,
        "measure": _measure,
        "calibrate-output": _calibrate_output,
        "watch": _watch,
    }
    actions[args.command](args)


if __name__ == "__main__":
    main()
