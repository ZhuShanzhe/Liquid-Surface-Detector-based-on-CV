#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from liquid_depth.container_geometry import load_container_model
from liquid_depth.sparse_contact import (
    analyze_contact_pixel_sensitivity,
    estimate_level_from_sparse_contact,
)


def _vector(value: str) -> np.ndarray:
    result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if result.shape != (3,):
        raise argparse.ArgumentTypeError("Expected three comma-separated values")
    return result


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _frame(payload, frame_key: str | None):
    if frame_key is None:
        return payload
    candidates = (frame_key, str(int(frame_key)))
    for key in candidates:
        if isinstance(payload, dict) and key in payload:
            return payload[key]
    raise KeyError(f"Frame {frame_key!r} is absent")


def _camera_matrix(payload) -> np.ndarray:
    if isinstance(payload, list):
        value = payload
    else:
        value = payload.get("camera_matrix", payload.get("cam_K"))
    if value is None:
        raise KeyError("Camera JSON requires camera_matrix or cam_K")
    return np.asarray(value, dtype=np.float64).reshape(3, 3)


def _pose(payload, instance_index: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(payload, list):
        payload = payload[instance_index]
    rotation = payload.get("rotation_m2c", payload.get("cam_R_m2c"))
    translation = payload.get("translation_m2c_m", payload.get("cam_t_m2c"))
    if rotation is None or translation is None:
        raise KeyError("Pose JSON requires rotation_m2c/cam_R_m2c and translation_m2c_m/cam_t_m2c")
    return (
        np.asarray(rotation, dtype=np.float64).reshape(3, 3),
        np.asarray(translation, dtype=np.float64).reshape(3),
    )


def _curve(payload) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(payload, list):
        points = payload
        confidence = None
    else:
        points = payload.get(
            "contact_curve_pixels",
            payload.get("points", payload.get("contact_curve")),
        )
        confidence = payload.get(
            "point_confidences",
            payload.get("contact_curve_confidences", payload.get("curve_confidence")),
        )
    if points is None:
        raise KeyError("Curve JSON requires contact_curve_pixels, contact_curve, or points")
    curve = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if confidence is None:
        return curve, None
    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if len(values) == 1:
        values = np.full(len(curve), values.item(), dtype=np.float64)
    return curve, values


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate metric liquid level from a 2D contact curve and calibrated container geometry"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--curve-json", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--level-axis", type=_vector, required=True)
    parser.add_argument("--level-origin-m", type=_vector, required=True)
    parser.add_argument("--frame-key")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument(
        "--translation-scale-to-m",
        type=float,
        default=0.0,
        help="0 selects automatic mm-to-m detection; otherwise multiply pose translation by this value",
    )
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--max-reprojection-px", type=float, default=6.0)
    parser.add_argument("--min-point-confidence", type=float, default=0.5)
    parser.add_argument("--max-selected-points", type=int, default=24)
    parser.add_argument("--horizontal-bins", type=int, default=8)
    parser.add_argument("--min-reliable-points", type=int, default=6)
    parser.add_argument("--min-horizontal-span-ratio", type=float, default=0.5)
    parser.add_argument("--min-occupied-bins", type=int, default=3)
    parser.add_argument(
        "--sensitivity-trials",
        type=int,
        default=0,
        help="Run pixel-to-metric perturbation analysis when positive",
    )
    parser.add_argument("--sensitivity-jitter-px", type=_float_tuple, default=(0.25, 0.5, 1.0, 2.0, 4.0))
    parser.add_argument(
        "--sensitivity-vertical-offset-px",
        type=_float_tuple,
        default=(-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = load_container_model(args.model, args.level_axis, args.level_origin_m)
    curve, point_confidences = _curve(_frame(_read_json(args.curve_json), args.frame_key))
    camera = _camera_matrix(_frame(_read_json(args.camera_json), args.frame_key))
    rotation, translation = _pose(
        _frame(_read_json(args.pose_json), args.frame_key),
        args.instance_index,
    )
    if args.translation_scale_to_m > 0:
        translation *= args.translation_scale_to_m
    elif np.linalg.norm(translation) > 10.0:
        translation *= 0.001

    sparse_options = {
        "min_point_confidence": args.min_point_confidence,
        "max_selected_points": args.max_selected_points,
        "horizontal_bins": args.horizontal_bins,
        "min_reliable_points": args.min_reliable_points,
        "min_horizontal_span_ratio": args.min_horizontal_span_ratio,
        "min_occupied_bins": args.min_occupied_bins,
        "geometry_options": {
            "neighbors": args.neighbors,
            "max_reprojection_px": args.max_reprojection_px,
        },
    }
    estimate = estimate_level_from_sparse_contact(
        model,
        curve,
        camera,
        rotation,
        translation,
        point_confidences=point_confidences,
        **sparse_options,
    )
    result = {
        **estimate.to_dict(),
        "model": str(args.model.resolve()),
        "model_level_range_m": list(model.level_range_m),
        "translation_m2c_m": translation.tolist(),
        "algorithm": "sparse_reliable_projected_container_geometry_v2",
    }
    if args.sensitivity_trials > 0:
        result["pixel_sensitivity"] = analyze_contact_pixel_sensitivity(
            model,
            curve,
            camera,
            rotation,
            translation,
            point_confidences=point_confidences,
            jitter_sigmas_px=args.sensitivity_jitter_px,
            vertical_offsets_px=args.sensitivity_vertical_offset_px,
            trials=args.sensitivity_trials,
            seed=args.seed,
            estimate_options=sparse_options,
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
