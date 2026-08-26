from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DTLD contact perception with grouped selective-risk metrics"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from liquid_depth.training.contact_metrics import summarize_curve_group
    from liquid_depth.training.dtld_contact import (
        build_dtld_contact_model,
        sample_cubic_bezier,
    )
    from liquid_depth.training.dtld_height import DTLDContactHeightDataset

    if args.stride < 1:
        raise ValueError("--stride must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = tuple(int(value) for value in state.get("image_size", (320, 180)))
    base_channels = int(state.get("base_channels", 24))
    max_depth_m = float(state.get("max_depth_m", 3.0))
    dataset = DTLDContactHeightDataset(
        args.manifest,
        args.split,
        image_size,
        max_depth_m,
        augment=False,
    )
    if args.stride > 1:
        dataset.rows = dataset.rows[:: args.stride]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = build_dtld_contact_model(
        state.get("backbone", "unet"),
        base_channels,
        pretrained_backbone=False,
        geometry_conditioning=bool(state.get("geometry_conditioning", False)),
        object_experts=bool(state.get("object_experts", False)),
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    scale = torch.tensor(image_size, device=device, dtype=torch.float32)
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True)
            object_index = target["object_index"].to(device, non_blocking=True)
            pose = target["pose"].to(device, non_blocking=True)
            prediction = model(inputs, object_index, pose)
            target_control = target["bezier_control_points"].to(device, non_blocking=True)
            target_curve = sample_cubic_bezier(target_control)
            curve_error = torch.linalg.vector_norm(
                (prediction["contact_curve"] - target_curve) * scale,
                dim=2,
            ).mean(dim=1)
            predicted_contact = prediction["contact_logits"].sigmoid() >= 0.5
            target_contact = target["contact"].to(device, non_blocking=True) >= 0.5
            intersection = (predicted_contact & target_contact).flatten(1).sum(dim=1)
            union = (predicted_contact | target_contact).flatten(1).sum(dim=1)
            color_target = target["color_residual"].to(device, non_blocking=True)
            support = target_contact.expand_as(color_target)
            residual_squared = (
                ((prediction["color_residual"] - color_target).square() * support).flatten(1).sum(dim=1)
            )
            residual_count = support.flatten(1).sum(dim=1)
            batch_values = {
                "curve_error": curve_error,
                "confidence": prediction["curve_confidence"],
                "intersection": intersection,
                "union": union,
                "residual_squared": residual_squared,
                "residual_count": residual_count,
                "row_index": target["row_index"],
            }
            for name, value in batch_values.items():
                values[name].append(value.detach().cpu().numpy())

    arrays = {name: np.concatenate(chunks) for name, chunks in values.items()}

    def summarize(indices: list[int] | np.ndarray) -> dict[str, object]:
        index = np.asarray(indices, dtype=np.int64)
        summary = summarize_curve_group(arrays["curve_error"][index], arrays["confidence"][index])
        summary["contact_iou_micro"] = float(
            arrays["intersection"][index].sum() / max(float(arrays["union"][index].sum()), 1.0)
        )
        summary["ali_residual_rmse"] = float(
            np.sqrt(
                arrays["residual_squared"][index].sum()
                / max(float(arrays["residual_count"][index].sum()), 1.0)
            )
        )
        summary["confidence_mean"] = float(arrays["confidence"][index].mean())
        return summary

    groups: dict[str, dict[str, list[int]]] = {
        "object_id": defaultdict(list),
        "difficulty_tag": defaultdict(list),
        "scenario": defaultdict(list),
        "sequence_id": defaultdict(list),
        "object_sequence": defaultdict(list),
    }
    for position, row_index in enumerate(arrays["row_index"].astype(int)):
        row = dataset.rows[row_index]
        groups["object_id"][row["object_id"]].append(position)
        groups["scenario"][row.get("scenario", "unknown")].append(position)
        groups["sequence_id"][row["sequence_id"]].append(position)
        groups["object_sequence"][f"{row['object_id']}:{row['sequence_id']}"].append(position)
        tags = [tag for tag in row.get("difficulty_tags", "").split(";") if tag]
        for tag in tags or ["untagged"]:
            groups["difficulty_tag"][tag].append(position)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "checkpoint_validation_metrics": state.get("metrics", {}),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "stride": args.stride,
        "image_size": image_size,
        "device": str(device),
        "overall": summarize(np.arange(len(dataset))),
        "groups": {
            group_name: {name: summarize(indices) for name, indices in sorted(mapping.items())}
            for group_name, mapping in groups.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
