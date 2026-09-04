#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _predicted_liquid_level(
    prediction: dict,
    row: dict,
    dataset_root: Path,
    threshold: float,
    width: int,
    height: int,
) -> tuple[float | None, float, int]:
    metadata_path = row.get("metadata_path", "").strip()
    if not metadata_path:
        return None, 0.0, 0
    path = Path(metadata_path)
    if not path.is_absolute():
        path = dataset_root / path
    metadata = json.loads(path.read_text(encoding="utf-8"))
    mask = prediction["mask_logits"].sigmoid()[0, 0].cpu().numpy() >= 0.5
    confidence = prediction["confidence"][0, 0].cpu().numpy()
    depth_m = prediction["depth_m"][0, 0].cpu().numpy()
    selected = mask & (confidence >= threshold) & np.isfinite(depth_m) & (depth_m > 0)
    minimum_points = max(64, int(0.01 * width * height))
    selected_points = int(selected.sum())
    mean_confidence = float(confidence[selected].mean()) if selected_points else 0.0
    if selected_points < minimum_points:
        return None, mean_confidence, selected_points

    intrinsics = np.asarray(metadata["camera_intrinsics"], dtype=np.float64)
    original_width = int(metadata["width"])
    original_height = int(metadata["height"])
    scale_x = width / original_width
    scale_y = height / original_height
    fx = intrinsics[0, 0] * scale_x
    fy = intrinsics[1, 1] * scale_y
    cx = intrinsics[0, 2] * scale_x
    cy = intrinsics[1, 2] * scale_y
    rows, columns = np.nonzero(selected)
    depth = depth_m[selected].astype(np.float64)
    camera_points = np.column_stack(
        (
            (columns - cx) / fx * depth,
            -(rows - cy) / fy * depth,
            -depth,
        )
    )
    camera_to_world = np.asarray(metadata["camera_to_world"], dtype=np.float64)
    world_z = camera_points @ camera_to_world[2, :3] + camera_to_world[2, 3]
    inside_bottom_m = float(metadata["container_bottom_z_m"]) + float(metadata["wall_thickness_m"])
    return (
        float(np.median(world_z) - inside_bottom_m),
        mean_confidence,
        selected_points,
    )


def _extract_records(args) -> tuple[list[dict], dict]:
    import torch
    from torch.utils.data import DataLoader

    from liquid_depth.models.universal import UniversalLiquidSurfaceNet
    from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = tuple(map(int, checkpoint["image_size"]))
    minimum = float(checkpoint.get("min_depth_m", 0.1))
    maximum = float(checkpoint.get("max_depth_m", 10.0))
    dataset = UniversalMultiTaskDataset(
        args.manifest,
        args.split,
        image_size,
        augment=False,
        min_depth_m=minimum,
        max_depth_m=maximum,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        pin_memory=True,
    )
    model = UniversalLiquidSurfaceNet(
        int(checkpoint["base_channels"]),
        minimum,
        maximum,
        rgb_prior_enabled=bool(checkpoint.get("rgb_prior_enabled", False)),
        separate_confidence_head=bool(checkpoint.get("separate_confidence_head", False)),
        level_calibration_enabled=bool(checkpoint.get("level_calibration_enabled", False)),
        calibration_scale_limit=float(checkpoint.get("calibration_scale_limit", 0.05)),
        calibration_bias_limit_m=float(checkpoint.get("calibration_bias_limit_m", 0.02)),
        robust_depth_anchor_enabled=bool(checkpoint.get("robust_depth_anchor_enabled", False)),
        robust_anchor_mask_threshold=float(checkpoint.get("robust_anchor_mask_threshold", 0.5)),
        robust_anchor_bias_limit_m=float(checkpoint.get("robust_anchor_bias_limit_m", 0.25)),
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    records = []
    with torch.inference_mode():
        for index, (inputs, _) in enumerate(loader):
            row = dataset.rows[index]
            prediction = model(inputs.to(device, non_blocking=True))
            level, confidence, points = _predicted_liquid_level(
                {key: value for key, value in prediction.items()},
                row,
                dataset.root,
                args.confidence_threshold,
                image_size[0],
                image_size[1],
            )
            metadata_path = Path(row["metadata_path"])
            if not metadata_path.is_absolute():
                metadata_path = dataset.root / metadata_path
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            truth_m = -(float(metadata["container_bottom_z_m"]) + float(metadata["wall_thickness_m"]))
            accepted = level is not None and truth_m > 0
            records.append(
                {
                    "sample_id": row.get("sequence_id", str(index)) + f":{index}",
                    "scenario": row.get("scenario", "unknown"),
                    "sensor_model": row.get("sensor_model", "unknown"),
                    "truth_m": truth_m,
                    "predicted_m": level,
                    "error_m": level - truth_m if accepted else None,
                    "relative_error": (level - truth_m) / truth_m if accepted else None,
                    "accepted": accepted,
                    "mean_confidence": confidence,
                    "selected_points": points,
                }
            )
    empirical = {}
    for scenario in sorted({str(item["scenario"]) for item in records}):
        values = [item for item in records if item["scenario"] == scenario]
        accepted = [item for item in values if item["accepted"]]
        errors = np.asarray([item["error_m"] for item in accepted], dtype=np.float64)
        relative = np.asarray(
            [item["relative_error"] for item in accepted],
            dtype=np.float64,
        )
        empirical[scenario] = {
            "frames": len(values),
            "accepted_frames": len(accepted),
            "frame_acceptance": len(accepted) / max(len(values), 1),
            "signed_bias_m": float(errors.mean()) if len(errors) else None,
            "mae_m": float(np.abs(errors).mean()) if len(errors) else None,
            "median_relative_error": float(np.median(relative)) if len(relative) else None,
        }
    return records, {
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "manifest": Path(args.manifest).resolve().as_posix(),
        "split": args.split,
        "confidence_threshold": args.confidence_threshold,
        "frames": len(records),
        "by_scenario": empirical,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate few-shot site calibration under market RGB-D error envelopes"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from liquid_depth.site_calibration_simulation import (
        CAMERA_ERROR_PROFILES,
        simulate_site_calibration,
    )

    records, empirical = _extract_records(args)
    strategies = {
        "current_5_levels_3_frames": {
            "calibration_level_count": 5,
            "calibration_frames": 3,
            "validation_frames": 3,
            "minimum_accepted_frames": 1,
            "design": "linear",
            "scenario_bias_correction": False,
        },
        "median_5_levels_5_frames": {
            "calibration_level_count": 5,
            "calibration_frames": 5,
            "validation_frames": 5,
            "minimum_accepted_frames": 2,
            "design": "hybrid",
            "scenario_bias_correction": False,
        },
        "robust_7_levels_5_frames": {
            "calibration_level_count": 7,
            "calibration_frames": 5,
            "validation_frames": 5,
            "minimum_accepted_frames": 2,
            "design": "hybrid",
            "scenario_bias_correction": False,
        },
        "research_scenario_bias_correction": {
            "calibration_level_count": 7,
            "calibration_frames": 5,
            "validation_frames": 5,
            "minimum_accepted_frames": 2,
            "design": "hybrid",
            "scenario_bias_correction": True,
        },
    }
    simulations = {}
    for profile_name, profile in CAMERA_ERROR_PROFILES.items():
        simulations[profile_name] = {}
        for strategy_index, (strategy_name, strategy) in enumerate(strategies.items()):
            simulations[profile_name][strategy_name] = simulate_site_calibration(
                records,
                profile,
                trials=args.trials,
                seed=args.seed + strategy_index,
                **strategy,
            )
    report = {
        "method": (
            "Model residuals come from the independent near-vertical synthetic test. "
            "Each scenario is split by stable order: one half estimates optional bias "
            "and the other half drives Monte Carlo validation. Camera errors are fixed "
            "per virtual site plus frame-random noise."
        ),
        "empirical_model_records": empirical,
        "strategies": strategies,
        "simulations": simulations,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        profile: {
            strategy: {
                "site_success_rate": result["site_success_rate"],
                "mae_m": result["global"]["mae_m"],
                "abs_rel": result["global"]["abs_rel"],
                "within_tolerance_rate": result["global"]["within_tolerance_rate"],
                "level_coverage": result["global"]["level_coverage"],
            }
            for strategy, result in values.items()
        }
        for profile, values in simulations.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
