#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _threshold_grid(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) == 3:
        start, stop, count = float(parts[0]), float(parts[1]), int(parts[2])
        values = np.linspace(start, stop, count)
    else:
        values = np.asarray([float(item) for item in parts])
    if values.size == 0 or np.any((values <= 0) | (values >= 1)):
        raise ValueError("Confidence thresholds must be inside (0, 1)")
    return sorted({round(float(item), 6) for item in values})


def _model(checkpoint: dict, device):

    from liquid_depth.models.universal import UniversalLiquidSurfaceNet

    model = UniversalLiquidSurfaceNet(
        int(checkpoint["base_channels"]),
        float(checkpoint.get("min_depth_m", 0.1)),
        float(checkpoint.get("max_depth_m", 10.0)),
        rgb_prior_enabled=bool(checkpoint.get("rgb_prior_enabled", False)),
        separate_confidence_head=bool(checkpoint.get("separate_confidence_head", False)),
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval().to(device)


def _empty() -> dict[str, float]:
    return {
        "frames": 0.0,
        "accepted": 0.0,
        "evaluable": 0.0,
        "absolute": 0.0,
        "relative": 0.0,
        "within": 0.0,
    }


def _summarize(store: dict[str, float]) -> dict[str, float | int | None]:
    frames = max(store["frames"], 1.0)
    accepted = store["accepted"]
    evaluable = store["evaluable"]
    return {
        "frames": int(store["frames"]),
        "accepted_frames": int(accepted),
        "evaluable_frames": int(evaluable),
        "coverage": accepted / frames,
        "evaluable_acceptance_rate": evaluable / max(accepted, 1.0),
        "mae_m": store["absolute"] / evaluable if evaluable else None,
        "abs_rel": store["relative"] / evaluable if evaluable else None,
        "within_tolerance_rate": store["within"] / evaluable if evaluable else 0.0,
    }


def _evaluate_split(
    checkpoint: dict,
    manifest: Path,
    split: str,
    thresholds: list[float],
    *,
    batch_size: int,
    workers: int,
    relative_tolerance: float,
    absolute_floor_m: float,
) -> tuple[dict[str, dict[str, dict]], Counter]:
    import torch
    from torch.utils.data import DataLoader

    from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset

    image_size = tuple(map(int, checkpoint["image_size"]))
    minimum = float(checkpoint.get("min_depth_m", 0.1))
    maximum = float(checkpoint.get("max_depth_m", 10.0))
    dataset = UniversalMultiTaskDataset(
        manifest,
        split,
        image_size,
        augment=False,
        min_depth_m=minimum,
        max_depth_m=maximum,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _model(checkpoint, device)
    stores: defaultdict[str, dict[float, dict[str, float]]] = defaultdict(
        lambda: {threshold: _empty() for threshold in thresholds}
    )
    scenario_counts = Counter(row.get("scenario", "unknown") for row in dataset.rows)
    offset = 0
    with torch.inference_mode():
        for inputs, target in loader:
            prediction = model(inputs.to(device, non_blocking=True))
            confidence = prediction["confidence"].cpu().numpy()
            predicted_mask = prediction["mask_logits"].sigmoid().cpu().numpy() >= 0.5
            depth = prediction["depth_m"].cpu().numpy()
            truth = target["depth_m"].numpy()
            truth_valid = target["valid"].numpy() > 0
            for batch_index in range(inputs.shape[0]):
                scenario = dataset.rows[offset + batch_index].get("scenario", "unknown")
                valid_count = int(truth_valid[batch_index].sum())
                minimum_points = max(64, int(0.01 * max(valid_count, 1)))
                for threshold in thresholds:
                    store = stores[scenario][threshold]
                    store["frames"] += 1.0
                    selected = predicted_mask[batch_index] & (confidence[batch_index] >= threshold)
                    if int(selected.sum()) < minimum_points:
                        continue
                    store["accepted"] += 1.0
                    overlap = selected & truth_valid[batch_index]
                    if (
                        int(overlap.sum()) < minimum_points
                        or float(overlap.sum()) / max(float(selected.sum()), 1.0) < 0.50
                    ):
                        continue
                    estimate = float(np.median(depth[batch_index][overlap]))
                    reference = float(np.median(truth[batch_index][overlap]))
                    absolute = abs(estimate - reference)
                    tolerance = max(absolute_floor_m, relative_tolerance * reference)
                    store["evaluable"] += 1.0
                    store["absolute"] += absolute
                    store["relative"] += absolute / max(reference, 1e-6)
                    store["within"] += float(absolute <= tolerance)
            offset += inputs.shape[0]
    result = {
        scenario: {f"{threshold:.6f}": _summarize(store) for threshold, store in values.items()}
        for scenario, values in sorted(stores.items())
    }
    return result, scenario_counts


def _choose(
    candidates: dict[str, dict],
    *,
    minimum_coverage: float,
    minimum_evaluable_rate: float,
    maximum_abs_rel: float,
    minimum_within_rate: float,
) -> dict:
    viable = []
    for raw_threshold, metrics in candidates.items():
        if metrics["evaluable_frames"] == 0:
            continue
        qualified = (
            metrics["coverage"] >= minimum_coverage
            and metrics["evaluable_acceptance_rate"] >= minimum_evaluable_rate
            and metrics["abs_rel"] <= maximum_abs_rel
            and metrics["within_tolerance_rate"] >= minimum_within_rate
        )
        if qualified:
            viable.append((metrics["coverage"], -metrics["mae_m"], raw_threshold, metrics))
    if viable:
        _, _, threshold, metrics = max(viable)
        return {"threshold": float(threshold), "qualified": True, "metrics": metrics}

    fallback = []
    for raw_threshold, metrics in candidates.items():
        if metrics["evaluable_frames"] == 0:
            continue
        coverage_penalty = max(0.0, minimum_coverage - metrics["coverage"])
        risk = metrics["abs_rel"] + 2.0 * coverage_penalty
        fallback.append((-risk, metrics["coverage"], raw_threshold, metrics))
    if not fallback:
        return {"threshold": 1.0, "qualified": False, "metrics": None}
    _, _, threshold, metrics = max(fallback)
    return {
        "threshold": float(threshold),
        "qualified": False,
        "metrics": metrics,
        "reason": "no_threshold_met_all_quality_constraints",
    }


def _apply_policy(
    evaluation: dict[str, dict[str, dict]],
    policy: dict[str, dict],
) -> dict:
    selected = {}
    totals = _empty()
    for scenario, choice in policy.items():
        threshold = f"{choice['threshold']:.6f}"
        metrics = evaluation.get(scenario, {}).get(threshold)
        selected[scenario] = {
            "threshold": choice["threshold"],
            "calibration_qualified": choice["qualified"],
            "metrics": metrics,
        }
        if metrics is None:
            continue
        totals["frames"] += metrics["frames"]
        totals["accepted"] += metrics["accepted_frames"]
        totals["evaluable"] += metrics["evaluable_frames"]
        if metrics["evaluable_frames"]:
            totals["absolute"] += metrics["mae_m"] * metrics["evaluable_frames"]
            totals["relative"] += metrics["abs_rel"] * metrics["evaluable_frames"]
            totals["within"] += metrics["within_tolerance_rate"] * metrics["evaluable_frames"]
    return {"overall": _summarize(totals), "by_scenario": selected}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit scenario-specific confidence thresholds on validation data"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--thresholds", default="0.30,0.995,48")
    parser.add_argument("--relative-tolerance", type=float, default=0.02)
    parser.add_argument("--absolute-floor-m", type=float, default=0.005)
    parser.add_argument("--minimum-coverage", type=float, default=0.30)
    parser.add_argument("--minimum-evaluable-rate", type=float, default=0.90)
    parser.add_argument("--maximum-abs-rel", type=float, default=0.03)
    parser.add_argument("--minimum-within-rate", type=float, default=0.50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch

    thresholds = _threshold_grid(args.thresholds)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fit, fit_counts = _evaluate_split(
        checkpoint,
        args.manifest,
        args.fit_split,
        thresholds,
        batch_size=args.batch_size,
        workers=args.workers,
        relative_tolerance=args.relative_tolerance,
        absolute_floor_m=args.absolute_floor_m,
    )
    policy = {
        scenario: _choose(
            candidates,
            minimum_coverage=args.minimum_coverage,
            minimum_evaluable_rate=args.minimum_evaluable_rate,
            maximum_abs_rel=args.maximum_abs_rel,
            minimum_within_rate=args.minimum_within_rate,
        )
        for scenario, candidates in fit.items()
    }
    evaluation, eval_counts = _evaluate_split(
        checkpoint,
        args.manifest,
        args.eval_split,
        thresholds,
        batch_size=args.batch_size,
        workers=args.workers,
        relative_tolerance=args.relative_tolerance,
        absolute_floor_m=args.absolute_floor_m,
    )
    report = {
        "schema_version": 1,
        "checkpoint": args.checkpoint.resolve().as_posix(),
        "manifest": args.manifest.resolve().as_posix(),
        "quality_profile": {
            "relative_tolerance": args.relative_tolerance,
            "absolute_floor_m": args.absolute_floor_m,
            "minimum_coverage": args.minimum_coverage,
            "minimum_evaluable_acceptance_rate": args.minimum_evaluable_rate,
            "maximum_abs_rel": args.maximum_abs_rel,
            "minimum_within_tolerance_rate": args.minimum_within_rate,
        },
        "fit_split": args.fit_split,
        "fit_scenario_counts": dict(fit_counts),
        "policy": policy,
        "fit_applied": _apply_policy(fit, policy),
        "eval_split": args.eval_split,
        "eval_scenario_counts": dict(eval_counts),
        "evaluation": _apply_policy(evaluation, policy),
        "synthetic_only": True,
        "hardware_qualified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output.resolve().as_posix(),
                "policy": policy,
                "evaluation": report["evaluation"]["overall"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
