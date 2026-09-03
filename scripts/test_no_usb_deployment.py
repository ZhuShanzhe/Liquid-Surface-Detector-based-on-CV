#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from liquid_depth.confidence_policy import depth_failure_confidence_group
from liquid_depth.io import load_frame
from liquid_depth.virtual_camera import prepare_universal_camera_input, replay_manifest


def _depth_meters(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32)
    valid = np.isfinite(value) & (value > 0)
    if np.any(valid) and float(np.median(value[valid])) > 10.0:
        value /= 1000.0
    return np.where(np.isfinite(value) & (value > 0), value, 0.0)


def _mask(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim == 3:
        value = np.any(value[..., :3] != 0, axis=2)
    return value != 0


def _confidence_entry(
    scenario: str,
    raw_valid_ratio: float,
    default_threshold: float,
    policy: dict | None,
) -> tuple[str, float, bool]:
    group = depth_failure_confidence_group(scenario, raw_valid_ratio)
    if policy is None:
        return group, default_threshold, True
    entry = policy.get(group)
    if entry is None and ":" in group:
        entry = policy.get(group.split(":", maxsplit=1)[0])
    if isinstance(entry, (int, float)):
        return group, float(entry), True
    if isinstance(entry, dict):
        return group, float(entry.get("threshold", default_threshold)), bool(
            entry.get("qualified", False)
        )
    return group, default_threshold, False


def _summary(
    errors: list[float],
    truths: list[float],
    *,
    relative_tolerance: float,
    absolute_floor_m: float,
) -> dict:
    if not errors:
        return {
            "accepted_frames": 0,
            "mae_m": None,
            "rmse_m": None,
            "abs_rel": None,
            "within_tolerance_rate": 0.0,
        }
    error = np.asarray(errors, dtype=np.float64)
    truth = np.asarray(truths, dtype=np.float64)
    absolute = np.abs(error)
    tolerance = np.maximum(absolute_floor_m, relative_tolerance * truth)
    return {
        "accepted_frames": len(errors),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "abs_rel": float(np.mean(absolute / np.maximum(truth, 1e-6))),
        "within_tolerance_rate": float(np.mean(absolute <= tolerance)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the deployment model through a virtual RGB-D camera without USB"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--frames", type=int, default=84)
    parser.add_argument("--capture-output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--confidence-report", type=Path)
    parser.add_argument("--relative-tolerance", type=float)
    parser.add_argument("--absolute-floor-m", type=float)
    parser.add_argument("--minimum-coverage", type=float)
    parser.add_argument("--maximum-abs-rel", type=float)
    parser.add_argument("--minimum-within-tolerance-rate", type=float)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    calibration_report: dict = {}
    if args.confidence_report is not None:
        calibration_report = json.loads(
            args.confidence_report.read_text(encoding="utf-8")
        )
    confidence_policy = calibration_report.get("policy")
    if confidence_policy is not None and not isinstance(confidence_policy, dict):
        raise TypeError("confidence report policy must be an object")
    profile = calibration_report.get("quality_profile", {})
    relative_tolerance = float(
        args.relative_tolerance
        if args.relative_tolerance is not None
        else profile.get("relative_tolerance", 0.02)
    )
    absolute_floor_m = float(
        args.absolute_floor_m
        if args.absolute_floor_m is not None
        else profile.get("absolute_floor_m", 0.005)
    )
    minimum_coverage = float(
        args.minimum_coverage
        if args.minimum_coverage is not None
        else profile.get("minimum_coverage", 0.30)
    )
    maximum_abs_rel = float(
        args.maximum_abs_rel
        if args.maximum_abs_rel is not None
        else profile.get("maximum_abs_rel", 0.03)
    )
    minimum_within_tolerance_rate = float(
        args.minimum_within_tolerance_rate
        if args.minimum_within_tolerance_rate is not None
        else profile.get("minimum_within_tolerance_rate", 0.50)
    )

    import torch

    captures = replay_manifest(
        args.manifest,
        args.capture_output,
        split=args.split,
        scenarios=args.scenarios,
        limit=args.frames,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = tuple(map(int, checkpoint["image_size"]))
    min_depth_m = float(checkpoint.get("min_depth_m", 0.1))
    max_depth_m = float(checkpoint.get("max_depth_m", 10.0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(args.model), map_location=device).eval()
    warmup = torch.zeros(
        1,
        5,
        image_size[1],
        image_size[0],
        device=device,
    )
    with torch.inference_mode():
        for _ in range(3):
            model(warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()

    records = []
    latencies = []
    scenario_errors: defaultdict[str, list[float]] = defaultdict(list)
    scenario_truths: defaultdict[str, list[float]] = defaultdict(list)
    raw_errors, raw_truths = [], []
    model_errors, model_truths = [], []
    for capture in captures:
        provenance = json.loads(
            (capture / "virtual_camera.json").read_text(encoding="utf-8")
        )
        frame = load_frame(capture)
        inputs = prepare_universal_camera_input(
            frame,
            image_size=image_size,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )
        tensor = torch.from_numpy(inputs)[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        width, height = image_size
        target = cv2.resize(
            _depth_meters(Path(provenance["source_target_depth_path"])),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        truth_mask = cv2.resize(
            _mask(Path(provenance["source_mask_path"])).astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        truth_valid = (
            truth_mask
            & np.isfinite(target)
            & (target >= min_depth_m)
            & (target <= max_depth_m)
        )
        raw = cv2.resize(
            frame.depth.astype(np.float32) * float(provenance["depth_scale_to_m"]),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        raw_selected = truth_valid & (raw > 0)
        raw_error = None
        if int(raw_selected.sum()) >= 64:
            raw_estimate = float(np.median(raw[raw_selected]))
            raw_truth = float(np.median(target[raw_selected]))
            raw_error = raw_estimate - raw_truth
            raw_errors.append(raw_error)
            raw_truths.append(raw_truth)
        raw_valid_ratio = float(raw_selected.sum()) / max(int(truth_valid.sum()), 1)
        scenario = str(provenance["scenario"])
        confidence_group, confidence_threshold, confidence_qualified = (
            _confidence_entry(
                scenario,
                raw_valid_ratio,
                args.confidence_threshold,
                confidence_policy,
            )
        )

        mask = prediction["mask_logits"].sigmoid()[0, 0].cpu().numpy() >= 0.5
        confidence = prediction["confidence"][0, 0].cpu().numpy()
        predicted_depth = prediction["depth_m"][0, 0].cpu().numpy()
        selected = (
            mask
            & (confidence >= confidence_threshold)
            & np.isfinite(predicted_depth)
            & (predicted_depth >= min_depth_m)
            & (predicted_depth <= max_depth_m)
        )
        minimum_points = max(64, int(0.01 * width * height))
        deployment_accepted = confidence_qualified and int(selected.sum()) >= minimum_points
        evaluation_selected = selected & truth_valid
        error = None
        truth_level = None
        if deployment_accepted and int(evaluation_selected.sum()) >= minimum_points:
            estimate = float(np.median(predicted_depth[evaluation_selected]))
            truth_level = float(np.median(target[evaluation_selected]))
            error = estimate - truth_level
            model_errors.append(error)
            model_truths.append(truth_level)
            scenario_errors[scenario].append(error)
            scenario_truths[scenario].append(truth_level)

        records.append(
            {
                "frame_id": frame.frame_id,
                "scenario": provenance["scenario"],
                "confidence_scenario": confidence_group,
                "confidence_threshold": confidence_threshold,
                "confidence_qualified": confidence_qualified,
                "raw_depth_valid_ratio": raw_valid_ratio,
                "rejection_reason": (
                    None
                    if deployment_accepted
                    else (
                        "scenario_confidence_not_qualified"
                        if not confidence_qualified
                        else "scenario_confidence_support_insufficient"
                    )
                ),
                "deployment_accepted": deployment_accepted,
                "selected_points": int(selected.sum()),
                "evaluation_points": int(evaluation_selected.sum()),
                "mean_selected_confidence": (
                    float(confidence[selected].mean()) if np.any(selected) else 0.0
                ),
                "truth_level_m": truth_level,
                "signed_error_m": error,
                "raw_signed_error_m": raw_error,
                "latency_ms": latency_ms,
                "capture_contract": {
                    "rgb_depth_shape_match": frame.rgb_bgr.shape[:2] == frame.depth.shape,
                    "intrinsics_valid": bool(
                        frame.camera_matrix[0, 0] > 0 and frame.camera_matrix[1, 1] > 0
                    ),
                    "millimeter_uint16": str(frame.depth.dtype) == "uint16",
                },
            }
        )

    latency = np.asarray(latencies, dtype=np.float64)
    summary_kwargs = {
        "relative_tolerance": relative_tolerance,
        "absolute_floor_m": absolute_floor_m,
    }
    restored_summary = _summary(model_errors, model_truths, **summary_kwargs)
    capture_contract_pass = all(
        all(record["capture_contract"].values()) for record in records
    )
    latency_pass = bool(np.percentile(latency, 95) <= 500.0)
    coverage_pass = len(model_errors) / max(len(captures), 1) >= minimum_coverage
    accuracy_pass = bool(
        restored_summary["abs_rel"] is not None
        and restored_summary["abs_rel"] <= maximum_abs_rel
        and restored_summary["within_tolerance_rate"] >= minimum_within_tolerance_rate
    )
    software_path_pass = capture_contract_pass and latency_pass
    report = {
        "mode": "virtual_rgbd_no_usb",
        "hardware_validated": False,
        "interfaces_tested": [
            "rgb.png",
            "depth.npy_uint16_mm",
            "depth_info.json",
            "camera_intrinsics",
            "universal_torchscript_inference",
            "confidence_rejection",
        ],
        "interfaces_not_tested": [
            "usb_transport",
            "ros2_driver",
            "hardware_rgb_depth_timestamp_sync",
            "sensor_temperature_drift",
            "real_optical_error",
        ],
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "model": args.model.resolve().as_posix(),
        "manifest": args.manifest.resolve().as_posix(),
        "confidence_report": (
            args.confidence_report.resolve().as_posix()
            if args.confidence_report is not None
            else None
        ),
        "quality_profile": {
            "relative_tolerance": relative_tolerance,
            "absolute_floor_m": absolute_floor_m,
            "minimum_coverage": minimum_coverage,
            "maximum_abs_rel": maximum_abs_rel,
            "minimum_within_tolerance_rate": minimum_within_tolerance_rate,
        },
        "frames": len(captures),
        "accepted_coverage": len(model_errors) / max(len(captures), 1),
        "raw_sensor_surface_oracle_mask": _summary(
            raw_errors,
            raw_truths,
            **summary_kwargs,
        ),
        "restored_surface": restored_summary,
        "qualification": {
            "capture_contract_pass": capture_contract_pass,
            "latency_p95_under_500ms_pass": latency_pass,
            "accepted_coverage_profile_pass": coverage_pass,
            "accuracy_profile_pass": accuracy_pass,
            "software_path_pass": software_path_pass,
            "deployment_ready": False,
        },
        "by_scenario": {
            scenario: _summary(
                errors, scenario_truths[scenario], **summary_kwargs
            )
            for scenario, errors in sorted(scenario_errors.items())
        },
        "latency_ms": {
            "mean": float(latency.mean()),
            "p95": float(np.percentile(latency, 95)),
            "max": float(latency.max()),
            "within_500ms_rate": float(np.mean(latency <= 500.0)),
        },
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": args.report.resolve().as_posix(),
                "frames": report["frames"],
                "accepted_coverage": report["accepted_coverage"],
                "restored_surface": report["restored_surface"],
                "latency_ms": report["latency_ms"],
                "qualification": report["qualification"],
                "hardware_validated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
