from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np


def _move(target: dict, device):
    return {name: value.to(device, non_blocking=True) for name, value in target.items()}


def _parse_object_boosts(specifications: list[str]) -> dict[str, float]:
    boosts = {}
    for specification in specifications:
        object_id, separator, multiplier = specification.partition(":")
        if not separator or object_id not in {"15", "16", "17", "19"}:
            raise ValueError(f"Invalid --object-boost: {specification!r}")
        value = float(multiplier)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("Object boost multipliers must be finite and positive")
        boosts[object_id] = value
    return boosts


def _balanced_weights(
    rows: list[dict[str, str]], object_boosts: dict[str, float] | None = None
) -> list[float]:
    keys = [
        (
            row["object_id"],
            int(float(row["liquid_height_mm"]) // 10.0),
            row["sequence_id"],
        )
        for row in rows
    ]
    counts = Counter(keys)
    boosts = object_boosts or {}
    raw = [
        boosts.get(row["object_id"], 1.0) / np.sqrt(counts[key]) for row, key in zip(rows, keys, strict=True)
    ]
    mean = sum(raw) / max(len(raw), 1)
    return [value / mean for value in raw]


def _evaluate(model, loader, device, image_size: tuple[int, int]) -> dict[str, object]:
    import torch

    model.eval()
    curve_errors = []
    confidence = []
    object_indices = []
    intersection = 0
    union = 0
    residual_squared = 0.0
    residual_values = 0
    scale = torch.tensor(image_size, device=device, dtype=torch.float32)
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            target = _move(target, device)
            prediction = model(inputs, target["object_index"], target["pose"])
            error = torch.linalg.vector_norm(
                (prediction["contact_curve"] - _sample_target(target)) * scale,
                dim=2,
            ).mean(dim=1)
            curve_errors.append(error.cpu())
            confidence.append(prediction["curve_confidence"].cpu())
            object_indices.append(target["object_index"].cpu())
            predicted_contact = prediction["contact_logits"].sigmoid() >= 0.5
            target_contact = target["contact"] >= 0.5
            intersection += int((predicted_contact & target_contact).sum())
            union += int((predicted_contact | target_contact).sum())
            difference = prediction["color_residual"] - target["color_residual"]
            support = target["contact"] > 0.05
            residual_squared += float((difference.square() * support).sum())
            residual_values += int(support.sum()) * 3
    values = torch.cat(curve_errors).numpy()
    confidence_values = torch.cat(confidence).numpy()
    object_values = torch.cat(object_indices).numpy()
    object_ids = ("15", "16", "17", "19")
    by_object = {
        object_id: {
            "samples": int((object_values == index).sum()),
            "mean_px": float(values[object_values == index].mean()),
            "p95_px": float(np.percentile(values[object_values == index], 95)),
        }
        for index, object_id in enumerate(object_ids)
        if np.any(object_values == index)
    }
    correlation = (
        float(np.corrcoef(confidence_values, values)[0, 1])
        if np.std(confidence_values) > 1e-8 and np.std(values) > 1e-8
        else 0.0
    )
    return {
        "val_curve_mae_px": float(values.mean()),
        "val_curve_p95_px": float(np.percentile(values, 95)),
        "val_contact_iou": intersection / max(union, 1),
        "val_ali_residual_rmse": float(np.sqrt(residual_squared / max(residual_values, 1))),
        "val_confidence_error_correlation": correlation,
        "val_curve_by_object": by_object,
        "val_worst_object_mae_px": max(group["mean_px"] for group in by_object.values()),
        "val_samples": len(values),
    }


def _sample_target(target: dict):
    from liquid_depth.training.dtld_contact import sample_cubic_bezier

    return sample_cubic_bezier(target["bezier_control_points"])


def _load_compatible(model, checkpoint: Path) -> int:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = state.get("model", state)
    current = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in current and current[name].shape == value.shape
    }
    expert_heads = {
        "contact_head.weight",
        "contact_head.bias",
        "control_heatmap_head.weight",
        "control_heatmap_head.bias",
    }
    for name in expert_heads:
        if name not in source or name not in current or name in compatible:
            continue
        value = source[name]
        target = current[name]
        if target.shape[1:] == value.shape[1:] and target.shape[0] % value.shape[0] == 0:
            repeats = (target.shape[0] // value.shape[0],) + (1,) * (value.ndim - 1)
            compatible[name] = value.repeat(repeats)
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CRM + Bezier contact perception for explicit container geometry"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", default="320,180", help="width,height")
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--backbone", choices=("unet", "resnet34"), default="unet")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--object-boost",
        action="append",
        default=[],
        metavar="OBJECT_ID:MULTIPLIER",
    )
    parser.add_argument("--geometry-conditioning", action="store_true")
    parser.add_argument("--object-experts", action="store_true")
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--decoupled-uncertainty", action="store_true")
    parser.add_argument("--uncertainty-weight", type=float, default=0.1)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--tail-weight", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from liquid_depth.training.dtld_contact import (
        DTLDContactGeometryLoss,
        build_dtld_contact_model,
    )
    from liquid_depth.training.dtld_height import DTLDContactHeightDataset

    if args.resume and args.warm_start:
        raise ValueError("Use either --resume or --warm-start, not both")
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
    object_boosts = _parse_object_boosts(args.object_boost)
    weights = torch.as_tensor(_balanced_weights(train_set.rows, object_boosts), dtype=torch.double)
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

    model = build_dtld_contact_model(
        args.backbone,
        args.base_channels,
        pretrained_backbone=args.pretrained_backbone and not args.resume,
        geometry_conditioning=args.geometry_conditioning,
        object_experts=args.object_experts,
    ).to(device)
    criterion = DTLDContactGeometryLoss(
        consistency_weight=args.consistency_weight,
        decoupled_uncertainty=args.decoupled_uncertainty,
        uncertainty_weight=args.uncertainty_weight,
        tail_fraction=args.tail_fraction,
        tail_weight=args.tail_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 1
    best_curve_px = float("inf")
    best_p95_px = float("inf")
    best_worst_object_px = float("inf")
    if args.warm_start:
        loaded = _load_compatible(model, args.warm_start)
        print(json.dumps({"warm_start": str(args.warm_start), "compatible_tensors": loaded}))
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_curve_px = float(state.get("best_curve_px", best_curve_px))
        best_p95_px = float(state.get("best_p95_px", best_p95_px))
        best_worst_object_px = float(state.get("best_worst_object_px", best_worst_object_px))

    metrics_path = output_dir / "metrics.jsonl"
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
            **_evaluate(model, val_loader, device, image_size),
        }
        print(json.dumps(metrics), flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        improved = metrics["val_curve_mae_px"] < best_curve_px
        improved_p95 = metrics["val_curve_p95_px"] < best_p95_px
        improved_worst = metrics["val_worst_object_mae_px"] < best_worst_object_px
        if improved:
            best_curve_px = metrics["val_curve_mae_px"]
        if improved_p95:
            best_p95_px = metrics["val_curve_p95_px"]
        if improved_worst:
            best_worst_object_px = metrics["val_worst_object_mae_px"]
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_curve_px": best_curve_px,
            "best_p95_px": best_p95_px,
            "best_worst_object_px": best_worst_object_px,
            "metrics": metrics,
            "image_size": image_size,
            "base_channels": args.base_channels,
            "backbone": args.backbone,
            "pretrained_backbone": args.pretrained_backbone,
            "object_boosts": object_boosts,
            "geometry_conditioning": args.geometry_conditioning,
            "object_experts": args.object_experts,
            "consistency_weight": args.consistency_weight,
            "decoupled_uncertainty": args.decoupled_uncertainty,
            "uncertainty_weight": args.uncertainty_weight,
            "tail_fraction": args.tail_fraction,
            "tail_weight": args.tail_weight,
            "max_depth_m": args.max_depth_m,
            "architecture": (
                "crm_resnet34_bezier_explicit_geometry_v5"
                if args.backbone == "resnet34"
                else "crm_bezier_object_experts_explicit_geometry_v4"
                if args.object_experts
                else "crm_bezier_pose_film_explicit_geometry_v3"
                if args.geometry_conditioning
                else "crm_bezier_spatial_explicit_geometry_v2"
            ),
            "input_contract": "instance RGB-D crop + object id + 6D pose",
            "output_contract": "contact heatmap + cubic Bezier + uncalibrated geometric confidence",
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if improved:
            torch.save(checkpoint, output_dir / "best.pth")
        if improved_p95:
            torch.save(checkpoint, output_dir / "best_p95.pth")
        if improved_worst:
            torch.save(checkpoint, output_dir / "best_worst_object.pth")


if __name__ == "__main__":
    main()
