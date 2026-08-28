from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def _move(target: dict, device):
    return {name: value.to(device, non_blocking=True) for name, value in target.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the project RGB-D multi-task network")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", default="640,360", help="width,height")
    parser.add_argument("--max-depth-m", type=float, default=3.0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if args.resume and args.initialize_from:
        parser.error("--resume and --initialize-from are mutually exclusive")

    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from liquid_depth.models import LiquidSurfaceMultiTaskNet, MultiTaskLoss
    from liquid_depth.sampling import balanced_sample_weights
    from liquid_depth.training.dataset import MultiTaskDataset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the configured training workflow")
    device = torch.device("cuda")
    image_size = tuple(map(int, args.image_size.split(",")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = MultiTaskDataset(args.manifest, "train", image_size, args.max_depth_m, augment=True)
    val_set = MultiTaskDataset(args.manifest, "val", image_size, args.max_depth_m, augment=False)
    weights = torch.as_tensor(balanced_sample_weights(train_set.rows), dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        train_set,
        args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = LiquidSurfaceMultiTaskNet(args.base_channels, args.max_depth_m).to(device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_rmse = float("inf")
    start_epoch = 1
    initial_checkpoint: str | None = None
    metrics_path = output_dir / "metrics.jsonl"
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        saved_total_epochs = int(state.get("total_epochs", args.epochs))
        if saved_total_epochs != args.epochs:
            raise ValueError(
                f"Resume checkpoint was scheduled for {saved_total_epochs} epochs, not {args.epochs}"
            )
        best_rmse = float(state.get("best_rmse", float("inf")))
        start_epoch = int(state["epoch"]) + 1
        initial_checkpoint = state.get("initial_checkpoint")
    elif args.initialize_from:
        state = torch.load(
            args.initialize_from,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state["model"], strict=True)
        initial_checkpoint = args.initialize_from.resolve().as_posix()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (inputs, target) in enumerate(train_loader, start=1):
            inputs, target = inputs.to(device, non_blocking=True), _move(target, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(inputs)
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

        model.eval()
        squared_error = absolute_error = valid_count = intersection = union = 0.0
        with torch.inference_mode():
            for inputs, target in val_loader:
                inputs, target = inputs.to(device, non_blocking=True), _move(target, device)
                prediction = model(inputs)
                valid = target["valid"] > 0
                error = prediction["depth_m"] - target["depth_m"]
                squared_error += float((error[valid] ** 2).sum())
                absolute_error += float(error[valid].abs().sum())
                valid_count += int(valid.sum())
                predicted_mask = prediction["mask_logits"].sigmoid() >= 0.5
                target_mask = target["mask"] > 0
                intersection += int((predicted_mask & target_mask).sum())
                union += int((predicted_mask | target_mask).sum())
        metrics = {
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "val_depth_rmse_m": (squared_error / max(valid_count, 1)) ** 0.5,
            "val_depth_mae_m": absolute_error / max(valid_count, 1),
            "val_mask_iou": intersection / max(union, 1),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        print(json.dumps(metrics))
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        improved = metrics["val_depth_rmse_m"] < best_rmse
        if improved:
            best_rmse = metrics["val_depth_rmse_m"]
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "total_epochs": args.epochs,
            "best_rmse": best_rmse,
            "metrics": metrics,
            "image_size": image_size,
            "base_channels": args.base_channels,
            "max_depth_m": args.max_depth_m,
            "initial_checkpoint": initial_checkpoint,
            "input_contract": "RGB ImageNet-normalized + depth/max_depth_m + validity",
        }
        torch.save(checkpoint, output_dir / "latest.pth")
        if improved:
            torch.save(checkpoint, output_dir / "best.pth")


if __name__ == "__main__":
    main()
