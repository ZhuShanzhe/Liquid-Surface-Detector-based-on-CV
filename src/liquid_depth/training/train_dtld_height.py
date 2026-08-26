from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _move(target: dict, device):
    return {name: value.to(device, non_blocking=True) for name, value in target.items()}


def _balanced_weights(rows: list[dict[str, str]]) -> list[float]:
    keys = [
        (
            row["object_id"],
            int(float(row["liquid_height_mm"]) // 10.0),
        )
        for row in rows
    ]
    counts = Counter(keys)
    raw = [1.0 / counts[key] for key in keys]
    mean = sum(raw) / max(len(raw), 1)
    return [value / mean for value in raw]


def _evaluate(model, loader, device) -> dict[str, float]:
    import torch

    model.eval()
    errors = []
    squared_error = 0.0
    contact_intersection = 0
    contact_union = 0
    confidence = []
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = _move(target, device)
            prediction = model(inputs, target["object_index"], target["pose"])
            error = (prediction["height_mm"] - target["height_mm"]).abs()
            errors.append(error.cpu())
            squared_error += float(error.square().sum())
            confidence.append(prediction["height_confidence"].cpu())
            predicted_contact = prediction["contact_logits"].sigmoid() >= 0.5
            target_contact = target["contact"] >= 0.5
            contact_intersection += int((predicted_contact & target_contact).sum())
            contact_union += int((predicted_contact | target_contact).sum())
    absolute = torch.cat(errors).numpy()
    confidence_values = torch.cat(confidence).numpy()
    correlation = (
        float(np.corrcoef(confidence_values, absolute)[0, 1])
        if np.std(confidence_values) > 1e-8 and np.std(absolute) > 1e-8
        else 0.0
    )
    return {
        "val_height_mae_mm": float(absolute.mean()),
        "val_height_rmse_mm": float(np.sqrt(squared_error / max(len(absolute), 1))),
        "val_height_p95_mm": float(np.percentile(absolute, 95)),
        "val_within_10mm_ratio": float(np.mean(absolute <= 10.0)),
        "val_contact_iou": contact_intersection / max(contact_union, 1),
        "val_confidence_error_correlation": correlation,
        "val_samples": len(absolute),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DTLD contact-line and metric liquid-height baseline")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", default="320,180", help="width,height")
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--max-height-mm", type=float, default=120.0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from liquid_depth.training.dtld_height import (
        DTLDContactHeightDataset,
        DTLDContactHeightLoss,
        DTLDContactHeightNet,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DTLD training")
    device = torch.device("cuda")
    image_size = tuple(map(int, args.image_size.split(",")))
    if len(image_size) != 2:
        raise ValueError("--image-size must be width,height")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = DTLDContactHeightDataset(
        args.manifest,
        "train",
        image_size,
        args.max_depth_m,
        augment=True,
    )
    val_set = DTLDContactHeightDataset(
        args.manifest,
        "val",
        image_size,
        args.max_depth_m,
        augment=False,
    )
    if args.train_stride > 1:
        train_set.rows = train_set.rows[:: args.train_stride]
    if args.val_stride > 1:
        val_set.rows = val_set.rows[:: args.val_stride]
    weights = torch.as_tensor(_balanced_weights(train_set.rows), dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )
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
    model = DTLDContactHeightNet(
        args.base_channels,
        args.max_height_mm,
    ).to(device)
    criterion = DTLDContactHeightLoss(args.max_height_mm)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    start_epoch = 1
    best_mae = float("inf")
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_mae = float(state.get("best_mae_mm", best_mae))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (inputs, target) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            target = _move(target, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs, target["object_index"], target["pose"])
                losses = criterion(prediction, target)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(losses["total"].detach())
            if args.log_every > 0 and step % args.log_every == 0:
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
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "learning_rate": scheduler.get_last_lr()[0],
            **_evaluate(model, val_loader, device),
        }
        print(json.dumps(metrics), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        improved = metrics["val_height_mae_mm"] < best_mae
        if improved:
            best_mae = metrics["val_height_mae_mm"]
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_mae_mm": best_mae,
            "metrics": metrics,
            "image_size": image_size,
            "base_channels": args.base_channels,
            "max_depth_m": args.max_depth_m,
            "max_height_mm": args.max_height_mm,
            "input_contract": "instance RGB-D crop + object id + 6D pose",
            "acceptance_target": {
                "working_distance_m": 1.0,
                "absolute_error_mm": 10.0,
            },
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if improved:
            torch.save(checkpoint, output_dir / "best.pth")


if __name__ == "__main__":
    main()
