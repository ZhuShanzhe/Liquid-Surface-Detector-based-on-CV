#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

RANGE_BINS = (
    ("0.1-0.3m", 0.1, 0.3),
    ("0.3-1m", 0.3, 1.0),
    ("1-3m", 1.0, 3.0),
    ("3-10m", 3.0, 10.0001),
)


def move_target(target: dict, device):
    return {name: value.to(device, non_blocking=True) for name, value in target.items()}


def accumulate_level_metrics(
    store: dict[str, float | int],
    selected,
    level_error,
    target_level,
    level_tolerance,
) -> None:
    selected_count = int(selected.sum())
    if selected_count == 0:
        return
    store["abs"] += float(level_error[selected].sum())
    store["rel"] += float((level_error[selected] / target_level[selected].clamp_min(1e-6)).sum())
    store["tol"] += int((level_error[selected] <= level_tolerance[selected]).sum())
    store["n"] += selected_count


def parse_priority_multipliers(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        tag, raw_value = item.split("=", maxsplit=1)
        result[tag.strip().lower()] = float(raw_value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the universal 0.1-10 m RGB-D liquid model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", default="320,180")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--relative-weight", type=float, default=1.0)
    parser.add_argument("--tolerance-weight", type=float, default=0.5)
    parser.add_argument("--uncertainty-weight", type=float, default=0.2)
    parser.add_argument("--surface-level-weight", type=float, default=0.5)
    parser.add_argument("--confidence-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--confidence-absolute-floor-m", type=float, default=0.005)
    parser.add_argument("--confidence-only-epochs", type=int, default=0)
    parser.add_argument("--surface-absolute-weight", type=float, default=0.0)
    parser.add_argument("--surface-tolerance-weight", type=float, default=0.0)
    parser.add_argument("--surface-quantile-weight", type=float, default=0.0)
    parser.add_argument("--surface-quantile", type=float, default=0.90)
    parser.add_argument("--ordinary-loss-boost", type=float, default=0.0)
    parser.add_argument("--calibration-regularization-weight", type=float, default=0.0)
    parser.add_argument("--level-calibration-head", action="store_true")
    parser.add_argument("--calibration-scale-limit", type=float, default=0.05)
    parser.add_argument("--calibration-bias-limit-m", type=float, default=0.02)
    parser.add_argument("--range-balance-strength", type=float, default=0.0)
    parser.add_argument("--range-balance-max-factor", type=float, default=4.0)
    parser.add_argument("--selection-ordinary-weight", type=float, default=0.0)
    parser.add_argument("--rgb-prior", action="store_true")
    parser.add_argument(
        "--difficulty-boosts", default="depth_failure=3,compound=2,multilayer=1.5,low_light=1.5,glare=1.25"
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("scenario_severity", "difficulty_tags"),
        default="scenario_severity",
    )
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")
    if not 0.0 <= args.range_balance_strength <= 1.0:
        parser.error("--range-balance-strength must be in [0, 1]")
    if not 0.0 <= args.surface_quantile < 1.0:
        parser.error("--surface-quantile must be in [0, 1)")
    if not 0.0 <= args.selection_ordinary_weight <= 1.0:
        parser.error("--selection-ordinary-weight must be in [0, 1]")

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from liquid_depth.models.universal import UniversalLiquidSurfaceNet, UniversalMultiTaskLoss
    from liquid_depth.sampling import (
        balanced_sample_weights,
        range_balanced_sample_weights,
        scenario_severity_sample_weights,
    )
    from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    image_size = tuple(map(int, args.image_size.split(",")))
    if len(image_size) != 2:
        parser.error("--image-size must be width,height")
    train_set = UniversalMultiTaskDataset(
        args.manifest,
        "train",
        image_size,
        augment=True,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    val_set = UniversalMultiTaskDataset(
        args.manifest,
        "val",
        image_size,
        augment=False,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )
    priority_multipliers = parse_priority_multipliers(args.difficulty_boosts)
    weight_function = (
        scenario_severity_sample_weights
        if args.sampling_mode == "scenario_severity"
        else balanced_sample_weights
    )
    sample_weights = weight_function(train_set.rows, priority_multipliers=priority_multipliers)
    if args.range_balance_strength > 0.0:
        sample_weights = range_balanced_sample_weights(
            train_set.rows,
            sample_weights,
            manifest_root=Path(args.manifest).resolve().parent,
            strength=args.range_balance_strength,
            maximum_factor=args.range_balance_max_factor,
        )
    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = UniversalLiquidSurfaceNet(
        args.base_channels,
        args.min_depth_m,
        args.max_depth_m,
        rgb_prior_enabled=args.rgb_prior,
        separate_confidence_head=True,
        level_calibration_enabled=args.level_calibration_head,
        calibration_scale_limit=args.calibration_scale_limit,
        calibration_bias_limit_m=args.calibration_bias_limit_m,
    ).to(device)
    criterion = UniversalMultiTaskLoss(
        relative_weight=args.relative_weight,
        tolerance_weight=args.tolerance_weight,
        uncertainty_weight=args.uncertainty_weight,
        surface_level_weight=args.surface_level_weight,
        surface_absolute_weight=args.surface_absolute_weight,
        surface_tolerance_weight=args.surface_tolerance_weight,
        surface_quantile_weight=args.surface_quantile_weight,
        surface_quantile=args.surface_quantile,
        ordinary_loss_boost=args.ordinary_loss_boost,
        calibration_regularization_weight=(args.calibration_regularization_weight),
        confidence_relative_tolerance=args.confidence_relative_tolerance,
        confidence_absolute_floor_m=args.confidence_absolute_floor_m,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    start_epoch, best_score, initial_checkpoint = 1, float("-inf"), None
    source_path = args.resume or args.initialize_from
    if source_path:
        state = torch.load(source_path, map_location="cpu", weights_only=False)
        if args.rgb_prior or args.level_calibration_head:
            incompatible = model.load_state_dict(state["model"], strict=False)
            invalid_missing = [
                name
                for name in incompatible.missing_keys
                if not name.startswith(("rgb_prior", "confidence_head", "level_calibration_head"))
            ]
            if invalid_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Incompatible initialization: missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
                )
        else:
            model.load_state_dict(state["model"], strict=True)
        initial_checkpoint = source_path.resolve().as_posix()
        if args.resume:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            best_score = float(state.get("best_score", float("-inf")))

    for epoch in range(start_epoch, args.epochs + 1):
        confidence_only = epoch <= args.confidence_only_epochs
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not confidence_only or name.startswith("confidence_head"))
        model.train()
        running = 0.0
        for step, (inputs, target) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            target = move_target(target, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs)
                losses = criterion(prediction, target)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(losses["total"].detach())
            if args.log_every and step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(train_loader),
                            "mean_train_loss": running / step,
                        }
                    ),
                    flush=True,
                )
        scheduler.step()
        model.eval()
        totals = {
            "sq": 0.0,
            "abs": 0.0,
            "rel": 0.0,
            "tol": 0,
            "brier": 0.0,
            "n": 0,
        }
        bins = {name: {"abs": 0.0, "rel": 0.0, "tol": 0, "n": 0} for name, _, _ in RANGE_BINS}
        level_totals = {"abs": 0.0, "rel": 0.0, "tol": 0, "n": 0}
        ordinary_level_totals = {
            "abs": 0.0,
            "rel": 0.0,
            "tol": 0,
            "n": 0,
        }
        intersection = union = 0
        with torch.inference_mode():
            for inputs, target in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                target = move_target(target, device)
                prediction = model(inputs)
                valid = target["valid"] > 0
                truth = target["depth_m"]
                error = prediction["depth_m"] - truth
                abs_error = error.abs()
                tolerance = torch.maximum(
                    truth * args.confidence_relative_tolerance,
                    torch.full_like(truth, args.confidence_absolute_floor_m),
                )
                totals["sq"] += float((error[valid] ** 2).sum())
                totals["abs"] += float(abs_error[valid].sum())
                totals["rel"] += float((abs_error[valid] / truth[valid].clamp_min(1e-6)).sum())
                reliable = (abs_error <= tolerance).float()
                totals["tol"] += int((reliable.bool() & valid).sum())
                totals["brier"] += float(((prediction["confidence"] - reliable) ** 2)[valid].sum())
                totals["n"] += int(valid.sum())
                valid_float = target["valid"].float()
                per_sample_count = valid_float.flatten(1).sum(dim=1)
                supported = per_sample_count > 0
                denominator = per_sample_count.clamp_min(1.0)
                predicted_level = (prediction["depth_m"].flatten(1) * valid_float.flatten(1)).sum(
                    dim=1
                ) / denominator
                target_level = (truth.flatten(1) * valid_float.flatten(1)).sum(dim=1) / denominator
                level_error = (predicted_level - target_level).abs()
                level_tolerance = torch.maximum(
                    target_level * args.confidence_relative_tolerance,
                    torch.full_like(target_level, args.confidence_absolute_floor_m),
                )

                accumulate_level_metrics(
                    level_totals,
                    supported,
                    level_error,
                    target_level,
                    level_tolerance,
                )
                ordinary = target["ordinary"].reshape(-1) > 0.5
                accumulate_level_metrics(
                    ordinary_level_totals,
                    supported & ordinary,
                    level_error,
                    target_level,
                    level_tolerance,
                )
                for name, low, high in RANGE_BINS:
                    selected = valid & (truth >= low) & (truth < high)
                    bins[name]["abs"] += float(abs_error[selected].sum())
                    bins[name]["rel"] += float((abs_error[selected] / truth[selected].clamp_min(1e-6)).sum())
                    bins[name]["tol"] += int(((abs_error <= tolerance) & selected).sum())
                    bins[name]["n"] += int(selected.sum())
                predicted_mask = prediction["mask_logits"].sigmoid() >= 0.5
                target_mask = target["mask"] > 0
                intersection += int((predicted_mask & target_mask).sum())
                union += int((predicted_mask | target_mask).sum())
        count = max(totals["n"], 1)
        level_count = max(level_totals["n"], 1)
        ordinary_level_count = max(ordinary_level_totals["n"], 1)
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "val_depth_rmse_m": (totals["sq"] / count) ** 0.5,
            "val_depth_mae_m": totals["abs"] / count,
            "val_abs_rel": totals["rel"] / count,
            "val_within_tolerance_rate": totals["tol"] / count,
            "val_confidence_brier_score": totals["brier"] / count,
            "val_mask_iou": intersection / max(union, 1),
            "val_surface_level_mae_m": (level_totals["abs"] / level_count),
            "val_surface_level_abs_rel": (level_totals["rel"] / level_count),
            "val_surface_level_within_tolerance_rate": (level_totals["tol"] / level_count),
            "val_ordinary_level_samples": ordinary_level_totals["n"],
            "val_ordinary_surface_level_mae_m": (ordinary_level_totals["abs"] / ordinary_level_count),
            "val_ordinary_surface_level_abs_rel": (ordinary_level_totals["rel"] / ordinary_level_count),
            "val_ordinary_surface_level_within_tolerance_rate": (
                ordinary_level_totals["tol"] / ordinary_level_count
            ),
            "range_metrics": {
                name: {
                    "pixels": values["n"],
                    "mae_m": values["abs"] / max(values["n"], 1),
                    "abs_rel": values["rel"] / max(values["n"], 1),
                    "within_tolerance_rate": values["tol"] / max(values["n"], 1),
                }
                for name, values in bins.items()
            },
            "learning_rate": scheduler.get_last_lr()[0],
        }
        ordinary_weight = args.selection_ordinary_weight if ordinary_level_totals["n"] > 0 else 0.0
        score = (
            (1.0 - ordinary_weight) * metrics["val_surface_level_within_tolerance_rate"]
            + ordinary_weight * metrics["val_ordinary_surface_level_within_tolerance_rate"]
            - 0.1
            * (
                (1.0 - ordinary_weight) * metrics["val_surface_level_abs_rel"]
                + ordinary_weight * metrics["val_ordinary_surface_level_abs_rel"]
            )
            - 0.05 * metrics["val_confidence_brier_score"]
        )
        metrics["selection_score"] = score
        print(json.dumps(metrics), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        improved = score > best_score
        best_score = max(best_score, score)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "total_epochs": args.epochs,
            "best_score": best_score,
            "metrics": metrics,
            "image_size": image_size,
            "base_channels": args.base_channels,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "depth_encoding": "log",
            "model_family": (
                "universal_liquid_surface_v7_ordinary_precision"
                if args.level_calibration_head
                else "universal_liquid_surface_v6_confidence_calibrated"
            ),
            "rgb_prior_enabled": args.rgb_prior,
            "uncertainty_weight": args.uncertainty_weight,
            "surface_level_weight": args.surface_level_weight,
            "surface_absolute_weight": args.surface_absolute_weight,
            "surface_tolerance_weight": args.surface_tolerance_weight,
            "surface_quantile_weight": args.surface_quantile_weight,
            "surface_quantile": args.surface_quantile,
            "ordinary_loss_boost": args.ordinary_loss_boost,
            "calibration_regularization_weight": (args.calibration_regularization_weight),
            "level_calibration_enabled": args.level_calibration_head,
            "calibration_scale_limit": args.calibration_scale_limit,
            "calibration_bias_limit_m": args.calibration_bias_limit_m,
            "range_balance_strength": args.range_balance_strength,
            "range_balance_max_factor": args.range_balance_max_factor,
            "selection_ordinary_weight": args.selection_ordinary_weight,
            "separate_confidence_head": True,
            "confidence_relative_tolerance": args.confidence_relative_tolerance,
            "confidence_absolute_floor_m": args.confidence_absolute_floor_m,
            "confidence_only_epochs": args.confidence_only_epochs,
            "difficulty_boosts": priority_multipliers,
            "sampling_mode": args.sampling_mode,
            "initial_checkpoint": initial_checkpoint,
            "input_contract": "RGB ImageNet-normalized + log-depth[0.1,10m] + validity",
        }
        torch.save(checkpoint, args.output_dir / "latest.pth")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pth")


if __name__ == "__main__":
    main()
