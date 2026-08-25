from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iou(prediction, target, torch) -> float:
    prediction = prediction == 1
    target = target == 1
    union = (prediction | target).sum().item()
    return float((prediction & target).sum().item() / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DeepLabV3 liquid segmentation model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--image-size", default="512,512", help="width,height")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

    from .dataset import SegmentationDataset

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the configured training workflow")
    image_size = tuple(map(int, args.image_size.split(",")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_set = SegmentationDataset(args.manifest, "train", image_size, augment=True)
    val_set = SegmentationDataset(args.manifest, "val", image_size, augment=False)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0
    )
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.workers, pin_memory=True)
    weights = None if args.no_pretrained else DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights, num_classes=2).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best_iou = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)["out"]
                loss = torch.nn.functional.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss)

        model.eval()
        val_ious = []
        with torch.inference_mode():
            for images, targets in val_loader:
                prediction = model(images.cuda(non_blocking=True))["out"].argmax(1).cpu()
                val_ious.extend(_iou(p, t, torch) for p, t in zip(prediction, targets))
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_iou": sum(val_ious) / len(val_ious),
        }
        print(json.dumps(metrics))
        if metrics["val_iou"] > best_iou:
            best_iou = metrics["val_iou"]
            torch.save({"model": model.state_dict(), "metrics": metrics, "image_size": image_size}, output_dir / "best.pth")


if __name__ == "__main__":
    main()

