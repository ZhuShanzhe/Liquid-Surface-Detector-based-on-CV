#!/usr/bin/env python3
"""Deterministic point-level qualification for temporal anchor recovery."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from liquid_depth.anchor_memory import TemporalAnchorMemory
from liquid_depth.container_geometry import ContainerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def project(points_m, matrix, translation):
    camera = points_m + translation
    homogeneous = camera @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, 31), np.linspace(-0.3, 0.3, 17))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    model = ContainerModel(points, np.array([0.0, 1.0, 0.0]), np.zeros(3))
    row = points[np.isclose(points[:, 1], 0.0)][::2]
    matrix = np.array([[240.0, 0.0, 320.0], [0.0, 240.0, 180.0], [0.0, 0.0, 1.0]])
    rotation = np.eye(3)
    memory = TemporalAnchorMemory(
        max_age_frames=45,
        max_history_frames=12,
        min_confidence=0.55,
        min_current_points=2,
        min_total_points=6,
        min_occupied_bins=3,
        max_memory_fraction=0.80,
        max_model_match_px=3.0,
        spatial_match_px=9.0,
    )
    scenarios = (
        [("warmup", 1.0, False)] * 5
        + [("large_failure", 0.25, False)] * 30
        + [("severe_failure", 0.12, False)] * 20
        + [("appearance_change", 0.25, True)] * 10
    )
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    latencies = []
    for index, (scenario, fresh_ratio, changed_rgb) in enumerate(scenarios):
        translation = np.array([0.04 * np.sin(index / 8.0), 0.015 * np.cos(index / 11.0), 2.0])
        truth = project(row, matrix, translation)
        current = truth + rng.normal(0.0, 0.7, size=truth.shape)
        confidence = np.full(len(current), 0.12)
        fresh_count = max(0, round(len(current) * fresh_ratio))
        fresh = rng.choice(len(current), size=fresh_count, replace=False)
        confidence[fresh] = 0.88
        image = np.full(
            (360, 640, 3),
            (240, 25, 25) if changed_rgb else (80, 120, 160),
            dtype=np.uint8,
        )
        started = perf_counter()
        fusion = memory.fuse(
            image,
            current,
            confidence,
            matrix,
            rotation,
            translation,
            roi_xyxy=(190, 100, 450, 260),
        )
        latencies.append((perf_counter() - started) * 1000.0)
        reliable = confidence >= memory.min_confidence
        baseline_bins = 0
        if np.any(reliable):
            normalized = np.clip((current[reliable, 0] - 190.0) / 260.0, 0, 1)
            baseline_bins = len(
                np.unique(
                    np.minimum(
                        (normalized * memory.horizontal_bins).astype(int),
                        memory.horizontal_bins - 1,
                    )
                )
            )
        baseline_accepted = int(
            reliable.sum() >= memory.min_total_points and baseline_bins >= memory.min_occupied_bins
        )
        bucket = totals[scenario]
        bucket["frames"] += 1
        bucket["baseline_accepted"] += baseline_accepted
        bucket["memory_accepted"] += int(fusion.accepted)
        bucket["recovered_points"] += fusion.recovered_points
        if fusion.accepted and not changed_rgb and fusion.total_reliable_points >= 2:
            fused_points = fusion.points_px[fusion.confidences >= memory.min_confidence]
            nearest = np.min(
                np.linalg.norm(fused_points[:, None, :] - truth[None, :, :], axis=2),
                axis=1,
            )
            bucket["accepted_point_error_px"] += float(nearest.mean())
            bucket["error_samples"] += 1
        if not changed_rgb and fusion.accepted:
            memory.commit(
                image,
                current,
                confidence,
                model,
                matrix,
                rotation,
                translation,
            )
    report = {"schema_version": 1, "seed": args.seed, "scenarios": {}}
    for name, value in totals.items():
        frames = value["frames"]
        report["scenarios"][name] = {
            "frames": int(frames),
            "baseline_acceptance": value["baseline_accepted"] / frames,
            "memory_acceptance": value["memory_accepted"] / frames,
            "mean_recovered_points": value["recovered_points"] / frames,
            "mean_accepted_point_error_px": (
                value["accepted_point_error_px"] / value["error_samples"] if value["error_samples"] else None
            ),
        }
    report["latency_ms"] = {
        "mean": float(np.mean(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "max": float(np.max(latencies)),
        "scope": "anchor_memory_fusion_only_cpu",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
