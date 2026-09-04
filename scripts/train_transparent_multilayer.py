#!/usr/bin/env python3
"""Train the transparent/semtransparent multi-layer RGB-D specialist."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def move_target(target: dict, device):
    return {name: value.to(device, non_blocking=True) for name, value in target.items()}


def filter_scenarios(dataset, scenarios: set[str]) -> None:
    dataset.rows = [
        row
        for row in dataset.rows
        if row.get("scenario", "").strip().lower() in scenarios and row.get("layer_depths_path", "").strip()
    ]
    if not dataset.rows:
        raise ValueError(f"No layered rows for scenarios {sorted(scenarios)}")


def evaluate(model, loader, device, args):
    import torch

    from liquid_depth.models.layered import select_liquid_interface

    totals = {
        "layer_abs": 0.0,
        "layer_rel": 0.0,
        "layer_tol": 0,
        "layer_n": 0,
        "tuple_ok": 0,
        "tuple_n": 0,
        "count_abs": 0.0,
        "count_n": 0,
        "base_abs": 0.0,
        "base_rel": 0.0,
        "base_tol": 0,
        "base_n": 0,
        "selected_abs": 0.0,
        "selected_rel": 0.0,
        "selected_tol": 0,
        "selected_n": 0,
        "liquid_possible": 0,
        "confidence_sum": 0.0,
        "evaluable_frames": 0,
        "output_frames": 0,
        "total_frames": 0,
        "surface_abs": 0.0,
        "surface_rel": 0.0,
        "surface_tol": 0,
        "surface_n": 0,
    }
    model.eval()
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = move_target(target, device)
            prediction = model(inputs)
            target_layers = target["layer_depths_m"]
            target_valid = target["layer_valid"] > 0
            errors = (target_layers[:, :, None] - prediction["layer_depths_m"][:, None]).abs()
            nearest = errors.min(dim=2).values
            tolerance = torch.maximum(
                target_layers * args.relative_tolerance,
                torch.full_like(
                    target_layers,
                    args.absolute_tolerance_m,
                ),
            )
            totals["layer_abs"] += float(nearest[target_valid].sum())
            totals["layer_rel"] += float(
                (nearest[target_valid] / target_layers[target_valid].clamp_min(1e-6)).sum()
            )
            totals["layer_tol"] += int(((nearest <= tolerance) & target_valid).sum())
            totals["layer_n"] += int(target_valid.sum())

            count = target_valid.sum(dim=1)
            multilayer = count >= 2
            tuple_ok = ((nearest <= tolerance) | ~target_valid).all(dim=1)
            totals["tuple_ok"] += int((tuple_ok & multilayer).sum())
            totals["tuple_n"] += int(multilayer.sum())
            predicted_count = prediction["layer_presence_logits"].sigmoid().sum(dim=1)
            any_layer = count > 0
            totals["count_abs"] += float((predicted_count[any_layer] - count[any_layer]).abs().sum())
            totals["count_n"] += int(any_layer.sum())

            liquid_valid = target["valid"] > 0
            liquid_truth = target["depth_m"]
            base_error = (prediction["depth_m"] - liquid_truth).abs()
            liquid_tolerance = torch.maximum(
                liquid_truth * args.relative_tolerance,
                torch.full_like(
                    liquid_truth,
                    args.absolute_tolerance_m,
                ),
            )
            totals["base_abs"] += float(base_error[liquid_valid].sum())
            totals["base_rel"] += float(
                (base_error[liquid_valid] / liquid_truth[liquid_valid].clamp_min(1e-6)).sum()
            )
            totals["base_tol"] += int(((base_error <= liquid_tolerance) & liquid_valid).sum())
            totals["base_n"] += int(liquid_valid.sum())

            selected = select_liquid_interface(
                prediction,
                prediction["depth_m"],
                relative_tolerance=args.selector_relative_tolerance,
                absolute_floor_m=args.selector_absolute_floor_m,
                rejection_multiplier=args.selector_rejection_multiplier,
                confidence_threshold=args.confidence_threshold,
            )
            predicted_liquid = prediction["mask_logits"].sigmoid() >= 0.5
            output_pixels = selected["accepted"] & predicted_liquid
            accepted = output_pixels & liquid_valid
            selected_error = (selected["depth_m"] - liquid_truth).abs()
            totals["selected_abs"] += float(selected_error[accepted].sum())
            totals["selected_rel"] += float(
                (selected_error[accepted] / liquid_truth[accepted].clamp_min(1e-6)).sum()
            )
            totals["selected_tol"] += int(((selected_error <= liquid_tolerance) & accepted).sum())
            totals["selected_n"] += int(accepted.sum())
            totals["liquid_possible"] += int(liquid_valid.sum())
            totals["confidence_sum"] += float(selected["confidence"][accepted].sum())
            output_frames = output_pixels.flatten(1).any(dim=1)
            evaluable_frames = accepted.flatten(1).any(dim=1)
            totals["total_frames"] += inputs.shape[0]
            totals["output_frames"] += int(output_frames.sum())
            totals["evaluable_frames"] += int(evaluable_frames.sum())
            for sample_index in range(inputs.shape[0]):
                sample_valid = liquid_valid[sample_index]
                sample_accepted = accepted[sample_index]
                minimum_points = max(64, int(0.01 * int(sample_valid.sum())))
                if int(sample_accepted.sum()) < minimum_points:
                    continue
                signed_error = (
                    selected["depth_m"][sample_index]
                    - liquid_truth[sample_index]
                )[sample_accepted]
                reference = float(liquid_truth[sample_index][sample_accepted].median())
                surface_error = float(signed_error.median().abs())
                surface_tolerance = max(
                    args.absolute_tolerance_m,
                    args.relative_tolerance * reference,
                )
                totals["surface_abs"] += surface_error
                totals["surface_rel"] += surface_error / max(reference, 1e-6)
                totals["surface_tol"] += int(surface_error <= surface_tolerance)
                totals["surface_n"] += 1

    layer_n = max(totals["layer_n"], 1)
    base_n = max(totals["base_n"], 1)
    selected_n = max(totals["selected_n"], 1)
    return {
        "val_layer_set_mae_m": totals["layer_abs"] / layer_n,
        "val_layer_set_abs_rel": totals["layer_rel"] / layer_n,
        "val_layer_recall_at_tolerance": totals["layer_tol"] / layer_n,
        "val_multilayer_tuple_accuracy": (totals["tuple_ok"] / max(totals["tuple_n"], 1)),
        "val_multilayer_pixels": totals["tuple_n"],
        "val_layer_count_mae": (totals["count_abs"] / max(totals["count_n"], 1)),
        "val_base_liquid_mae_m": totals["base_abs"] / base_n,
        "val_base_liquid_abs_rel": totals["base_rel"] / base_n,
        "val_base_liquid_within_tolerance": totals["base_tol"] / base_n,
        "val_selected_liquid_mae_m": (totals["selected_abs"] / selected_n),
        "val_selected_liquid_abs_rel": (totals["selected_rel"] / selected_n),
        "val_selected_liquid_within_tolerance": (totals["selected_tol"] / selected_n),
        "val_selected_liquid_coverage": (totals["selected_n"] / max(totals["liquid_possible"], 1)),
        "val_selected_liquid_mean_confidence": (totals["confidence_sum"] / selected_n),
        "val_selected_liquid_frame_coverage": (
            totals["output_frames"] / max(totals["total_frames"], 1)
        ),
        "val_evaluable_output_rate": (
            totals["evaluable_frames"] / max(totals["output_frames"], 1)
        ),
        "val_surface_level_mae_m": (
            totals["surface_abs"] / max(totals["surface_n"], 1)
        ),
        "val_surface_level_abs_rel": (
            totals["surface_rel"] / max(totals["surface_n"], 1)
        ),
        "val_surface_level_within_tolerance": (
            totals["surface_tol"] / max(totals["surface_n"], 1)
        ),
        "val_surface_level_coverage": (
            totals["surface_n"] / max(totals["total_frames"], 1)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SeeGroup-inspired metric ray decomposition")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--head-only-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate-scale", type=float, default=0.05)
    parser.add_argument("--image-size", default="320,180")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--num-ray-layers", type=int, default=4)
    parser.add_argument("--ray-layer-hidden-channels", type=int, default=32)
    parser.add_argument(
        "--scenarios",
        default="transparent,translucent,multilayer,compound",
    )
    parser.add_argument("--base-loss-weight", type=float, default=0.25)
    parser.add_argument("--layer-loss-weight", type=float, default=1.0)
    parser.add_argument("--multilayer-pixel-boost", type=float, default=4.0)
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.005)
    parser.add_argument("--selector-relative-tolerance", type=float, default=0.08)
    parser.add_argument("--selector-absolute-floor-m", type=float, default=0.02)
    parser.add_argument("--selector-rejection-multiplier", type=float, default=3.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from liquid_depth.models.layered import PermutationInvariantLayerLoss
    from liquid_depth.models.universal import (
        UniversalLiquidSurfaceNet,
        UniversalMultiTaskLoss,
    )
    from liquid_depth.training.universal_dataset import (
        UniversalMultiTaskDataset,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    scenarios = {value.strip().lower() for value in args.scenarios.split(",") if value.strip()}
    image_size = tuple(map(int, args.image_size.split(",")))
    state = torch.load(
        args.initialize_from,
        map_location="cpu",
        weights_only=False,
    )
    min_depth_m = float(state.get("min_depth_m", 0.1))
    max_depth_m = float(state.get("max_depth_m", 10.0))
    train_set = UniversalMultiTaskDataset(
        args.manifest,
        "train",
        image_size,
        augment=True,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    val_set = UniversalMultiTaskDataset(
        args.manifest,
        "val",
        image_size,
        augment=False,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
    )
    filter_scenarios(train_set, scenarios)
    filter_scenarios(val_set, scenarios)
    if args.max_train_samples is not None:
        train_set.rows = train_set.rows[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_set.rows = val_set.rows[: args.max_val_samples]
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    model = UniversalLiquidSurfaceNet(
        base_channels=int(state.get("base_channels", 32)),
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        rgb_prior_enabled=bool(state.get("rgb_prior_enabled", False)),
        separate_confidence_head=True,
        level_calibration_enabled=bool(state.get("level_calibration_enabled", False)),
        calibration_scale_limit=float(state.get("calibration_scale_limit", 0.05)),
        calibration_bias_limit_m=float(state.get("calibration_bias_limit_m", 0.02)),
        robust_depth_anchor_enabled=False,
        num_ray_layers=args.num_ray_layers,
        ray_layer_hidden_channels=args.ray_layer_hidden_channels,
    ).to(device)
    incompatible = model.load_state_dict(state["model"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [name for name in incompatible.missing_keys if not name.startswith("ray_layer_head.")]
    if unexpected or invalid_missing:
        raise RuntimeError(f"Initialization mismatch: missing={invalid_missing}, unexpected={unexpected}")

    base_criterion = UniversalMultiTaskLoss(
        relative_weight=1.0,
        tolerance_weight=0.5,
        uncertainty_weight=0.2,
        surface_level_weight=0.5,
        confidence_relative_tolerance=args.relative_tolerance,
        confidence_absolute_floor_m=args.absolute_tolerance_m,
    )
    layer_criterion = PermutationInvariantLayerLoss(
        gamma=0.8,
        multilayer_pixel_boost=args.multilayer_pixel_boost,
    )
    head_parameters = list(model.ray_layer_head.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.learning_rate},
            {
                "params": backbone_parameters,
                "lr": args.learning_rate * args.backbone_learning_rate_scale,
            },
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        head_only = epoch <= args.head_only_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("ray_layer_head.") or not head_only)
        model.train()
        running = {
            "total": 0.0,
            "base": 0.0,
            "layer": 0.0,
        }
        for step, (inputs, target) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            target = move_target(target, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs)
                base_losses = base_criterion(prediction, target)
                layer_losses = layer_criterion(
                    prediction,
                    target["layer_depths_m"],
                    target["layer_valid"],
                    target["depth_m"],
                    target["valid"],
                )
                base_weight = 0.0 if head_only else args.base_loss_weight
                loss = base_weight * base_losses["total"] + args.layer_loss_weight * layer_losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running["total"] += float(loss.detach())
            running["base"] += float(base_losses["total"].detach())
            running["layer"] += float(layer_losses["total"].detach())
            if args.log_every and step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(train_loader),
                            "head_only": head_only,
                            "mean_total_loss": running["total"] / step,
                            "mean_layer_loss": running["layer"] / step,
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        metrics = evaluate(model, val_loader, device, args)
        metrics.update(
            {
                "epoch": epoch,
                "head_only": head_only,
                "train_total_loss": running["total"] / len(train_loader),
                "train_base_loss": running["base"] / len(train_loader),
                "train_layer_loss": running["layer"] / len(train_loader),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
            }
        )
        selection_score = (
            metrics["val_multilayer_tuple_accuracy"]
            + metrics["val_selected_liquid_within_tolerance"]
            + 0.5 * metrics["val_selected_liquid_coverage"]
            - metrics["val_selected_liquid_abs_rel"]
        )
        metrics["selection_score"] = selection_score
        print(json.dumps(metrics), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "image_size": image_size,
            "base_channels": int(state.get("base_channels", 32)),
            "min_depth_m": min_depth_m,
            "max_depth_m": max_depth_m,
            "num_ray_layers": args.num_ray_layers,
            "ray_layer_hidden_channels": args.ray_layer_hidden_channels,
            "model_family": "universal_rgbd_self_grouping_multilayer_v9_1",
            "route_scope": sorted(scenarios),
            "rgb_prior_enabled": bool(state.get("rgb_prior_enabled", False)),
            "separate_confidence_head": True,
            "level_calibration_enabled": bool(
                state.get("level_calibration_enabled", False)
            ),
            "calibration_scale_limit": float(
                state.get("calibration_scale_limit", 0.05)
            ),
            "calibration_bias_limit_m": float(
                state.get("calibration_bias_limit_m", 0.02)
            ),
            "robust_depth_anchor_enabled": False,
            "recommended_interface_confidence_threshold": 0.50,
            "initial_checkpoint": str(args.initialize_from.resolve()),
            "layer_representation": (
                "unordered recurrent Laplace components with bidirectional maximum-intensity loss"
            ),
        }
        torch.save(checkpoint, args.output_dir / "last.pth")
        if selection_score > best_score:
            best_score = selection_score
            torch.save(checkpoint, args.output_dir / "best.pth")

    summary = {
        "best_score": best_score,
        "best_checkpoint": str((args.output_dir / "best.pth").resolve()),
        "training_samples": len(train_set),
        "validation_samples": len(val_set),
        "scenarios": sorted(scenarios),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
