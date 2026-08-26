#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _summary(records: list[dict], confidence_threshold: float) -> dict:
    if not records:
        return {"samples": 0, "accepted": 0, "coverage": 0.0}
    accepted = [record for record in records if record["confidence"] >= confidence_threshold]
    result = {
        "samples": len(records),
        "accepted": len(accepted),
        "coverage": len(accepted) / len(records),
        "confidence_threshold": confidence_threshold,
    }
    if not accepted:
        return result
    error = np.asarray([record["error_mm"] for record in accepted])
    truth = np.asarray([record["truth_mm"] for record in accepted])
    relative = error / np.maximum(np.abs(truth), 1.0)
    result.update(
        {
            "mae_mm": float(error.mean()),
            "rmse_mm": float(np.sqrt(np.mean(error**2))),
            "p95_absolute_error_mm": float(np.percentile(error, 95)),
            "mape_percent": float(relative.mean() * 100.0),
            "p95_relative_error_percent": float(np.percentile(relative, 95) * 100.0),
            "within_1percent_ratio": float(np.mean(relative <= 0.01)),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a DTLD contact-height checkpoint on a held-out split"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from liquid_depth.training.dtld_height import (
        DTLDContactHeightDataset,
        DTLDContactHeightNet,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for DTLD evaluation")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = DTLDContactHeightDataset(
        args.manifest,
        args.split,
        tuple(state["image_size"]),
        float(state["max_depth_m"]),
        augment=False,
    )
    if args.stride > 1:
        dataset.rows = dataset.rows[:: args.stride]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = DTLDContactHeightNet(
        int(state["base_channels"]),
        float(state["max_height_mm"]),
    ).cuda()
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    records = []
    with torch.inference_mode():
        for inputs, target in loader:
            prediction = model(
                inputs.cuda(non_blocking=True),
                target["object_index"].cuda(non_blocking=True),
                target["pose"].cuda(non_blocking=True),
            )
            predicted = prediction["height_mm"].cpu().numpy()
            confidence = prediction["height_confidence"].cpu().numpy()
            truth = target["height_mm"].numpy()
            for offset, row_index in enumerate(target["row_index"].tolist()):
                row = dataset.rows[row_index]
                records.append(
                    {
                        "frame_id": row["frame_id"],
                        "sequence_id": row["sequence_id"],
                        "object_id": row["object_id"],
                        "truth_mm": float(truth[offset]),
                        "prediction_mm": float(predicted[offset]),
                        "error_mm": float(abs(predicted[offset] - truth[offset])),
                        "confidence": float(confidence[offset]),
                    }
                )

    by_object = defaultdict(list)
    by_sequence = defaultdict(list)
    for record in records:
        by_object[record["object_id"]].append(record)
        by_sequence[record["sequence_id"]].append(record)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "acceptance_target": {
            "relative_error_percent": 1.0,
            "example_distance_m": 1.0,
            "example_tolerance_mm": 10.0,
        },
        "overall": _summary(records, args.confidence_threshold),
        "by_object": {
            key: _summary(value, args.confidence_threshold) for key, value in sorted(by_object.items())
        },
        "by_sequence": {
            key: _summary(value, args.confidence_threshold) for key, value in sorted(by_sequence.items())
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
