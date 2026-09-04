#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def select_operating_point(rows: list[dict[str, float]]) -> dict[str, float] | None:
    qualified = [
        row
        for row in rows
        if row["val_selected_liquid_abs_rel"] <= 0.03
        and row["val_selected_liquid_within_tolerance"] >= 0.50
        and row["val_selected_liquid_frame_coverage"] >= 0.30
        and row["val_evaluable_output_rate"] >= 0.90
    ]
    if not qualified:
        return None
    return max(
        qualified,
        key=lambda row: (
            row["val_selected_liquid_within_tolerance"],
            -row["val_selected_liquid_abs_rel"],
            row["val_selected_liquid_frame_coverage"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and calibrate a transparent multi-layer checkpoint"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.005)
    parser.add_argument("--selector-relative-tolerance", type=float, default=0.008)
    parser.add_argument("--selector-absolute-floor-m", type=float, default=0.02)
    parser.add_argument("--selector-rejection-multiplier", type=float, default=3.0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from train_transparent_multilayer import evaluate, filter_scenarios

    from liquid_depth.models.universal import UniversalLiquidSurfaceNet
    from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_keys = state["model"]
    model = UniversalLiquidSurfaceNet(
        base_channels=int(state.get("base_channels", 32)),
        min_depth_m=float(state.get("min_depth_m", 0.1)),
        max_depth_m=float(state.get("max_depth_m", 10.0)),
        rgb_prior_enabled=bool(
            state.get(
                "rgb_prior_enabled",
                any(name.startswith("rgb_prior.") for name in model_keys),
            )
        ),
        separate_confidence_head=bool(state.get("separate_confidence_head", True)),
        level_calibration_enabled=bool(
            state.get(
                "level_calibration_enabled",
                any(name.startswith("level_calibration_head.") for name in model_keys),
            )
        ),
        calibration_scale_limit=float(state.get("calibration_scale_limit", 0.05)),
        calibration_bias_limit_m=float(
            state.get("calibration_bias_limit_m", 0.02)
        ),
        robust_depth_anchor_enabled=bool(
            state.get("robust_depth_anchor_enabled", False)
        ),
        num_ray_layers=int(state.get("num_ray_layers", 4)),
        ray_layer_hidden_channels=int(state.get("ray_layer_hidden_channels", 32)),
    ).cuda()
    model.load_state_dict(model_keys, strict=True)

    image_size = tuple(state.get("image_size", (320, 180)))
    dataset = UniversalMultiTaskDataset(
        args.manifest,
        args.split,
        image_size,
        augment=False,
        min_depth_m=float(state.get("min_depth_m", 0.1)),
        max_depth_m=float(state.get("max_depth_m", 10.0)),
    )
    scenarios = set(state.get("route_scope", ()))
    if scenarios:
        filter_scenarios(dataset, scenarios)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    rows = []
    for threshold in (float(value) for value in args.thresholds.split(",")):
        options = SimpleNamespace(
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance_m=args.absolute_tolerance_m,
            selector_relative_tolerance=args.selector_relative_tolerance,
            selector_absolute_floor_m=args.selector_absolute_floor_m,
            selector_rejection_multiplier=args.selector_rejection_multiplier,
            confidence_threshold=threshold,
        )
        metrics = evaluate(model, loader, torch.device("cuda"), options)
        row = {"confidence_threshold": threshold, **metrics}
        rows.append(row)
        print(json.dumps(row), flush=True)

    recommended = select_operating_point(rows)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "sample_count": len(dataset),
        "selector_rejection_multiplier": args.selector_rejection_multiplier,
        "promotion_gates": {
            "selected_liquid_abs_rel_max": 0.03,
            "tolerance_pass_rate_min": 0.50,
            "accepted_coverage_min": 0.30,
            "evaluable_output_rate_min": 0.90,
        },
        "operating_points": rows,
        "recommended_operating_point": recommended,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"recommended_operating_point": recommended}), flush=True)


if __name__ == "__main__":
    main()
