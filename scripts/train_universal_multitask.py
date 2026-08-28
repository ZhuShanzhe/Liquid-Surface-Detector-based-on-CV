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
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from liquid_depth.models.universal import UniversalLiquidSurfaceNet, UniversalMultiTaskLoss
    from liquid_depth.sampling import balanced_sample_weights
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
    weights = torch.as_tensor(balanced_sample_weights(train_set.rows), dtype=torch.double)
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
        args.base_channels, args.min_depth_m, args.max_depth_m
    ).to(device)
    criterion = UniversalMultiTaskLoss(
        relative_weight=args.relative_weight,
        tolerance_weight=args.tolerance_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    start_epoch, best_score, initial_checkpoint = 1, float("-inf"), None
    source_path = args.resume or args.initialize_from
    if source_path:
        state = torch.load(source_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        initial_checkpoint = source_path.resolve().as_posix()
        if args.resume:
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            best_score = float(state.get("best_score", float("-inf")))

    for epoch in range(start_epoch, args.epochs + 1):
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
        totals = {"sq": 0.0, "abs": 0.0, "rel": 0.0, "tol": 0, "n": 0}
        bins = {name: {"abs": 0.0, "rel": 0.0, "tol": 0, "n": 0} for name, _, _ in RANGE_BINS}
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
                tolerance = torch.maximum(truth * 0.01, torch.full_like(truth, 0.003))
                totals["sq"] += float((error[valid] ** 2).sum())
                totals["abs"] += float(abs_error[valid].sum())
                totals["rel"] += float((abs_error[valid] / truth[valid].clamp_min(1e-6)).sum())
                totals["tol"] += int(((abs_error <= tolerance) & valid).sum())
                totals["n"] += int(valid.sum())
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
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "val_depth_rmse_m": (totals["sq"] / count) ** 0.5,
            "val_depth_mae_m": totals["abs"] / count,
            "val_abs_rel": totals["rel"] / count,
            "val_within_tolerance_rate": totals["tol"] / count,
            "val_mask_iou": intersection / max(union, 1),
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
        score = metrics["val_within_tolerance_rate"] - 0.1 * metrics["val_abs_rel"]
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
            "model_family": "universal_liquid_surface_v3",
            "initial_checkpoint": initial_checkpoint,
            "input_contract": "RGB ImageNet-normalized + log-depth[0.1,10m] + validity",
        }
        torch.save(checkpoint, args.output_dir / "latest.pth")
        if improved:
            torch.save(checkpoint, args.output_dir / "best.pth")


if __name__ == "__main__":
    main()
