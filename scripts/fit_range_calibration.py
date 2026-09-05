#!/usr/bin/env python3
"""Fit simulator-only noise and support confidence using development seed 10607."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from liquid_depth.range_calibration import RANGE_EDGES, SCORE_EDGES
from liquid_depth.surface_video_runtime import SequencePredictor


def noise_fit(samples):
    z = np.asarray([s[0] for s in samples])
    y = np.asarray([s[1] for s in samples])
    design = np.column_stack((np.ones(len(z)), z * z))
    # Relative weighting prevents metre-range noise from setting the near-range
    # intercept. No held-out measurements are used to choose these coefficients.
    weight = 1 / np.maximum(y, 0.0003)
    wd, wy = design * weight[:, None], y * weight
    candidates = [
        np.maximum(np.linalg.lstsq(wd, wy, rcond=None)[0], 0),
        np.array([np.sum(weight * wy) / np.sum(weight**2), 0.0]),
        np.array([0.0, np.sum(wd[:, 1] * wy) / np.sum(wd[:, 1] ** 2)]),
    ]
    coef = min(candidates, key=lambda c: np.mean((wd @ c - wy) ** 2))
    # Conservative robust envelope of clean per-frame noise, not error bias.
    ratio = y / np.maximum(design @ coef, 0.0001)
    return coef * max(1.0, float(np.quantile(ratio, 0.90)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    rows = [
        r
        for r in json.loads((args.data / "sequences.json").read_text())
        if r["seed"] == 10607 and r["motion"] == "static"
    ]
    if not rows:
        raise ValueError("Missing explicit development split")
    predictor = SequencePredictor(args.checkpoint)
    samples, observations = defaultdict(list), defaultdict(list)
    for row in rows:
        path = Path(row["path"])
        a = np.load(path / "frame.npz")
        raw, truth = a["depth"], a["truth_depth"]
        mask = a["truth_mask"].astype(bool)
        pred = predictor.predict(cv2.imread(str(path / "rgb.png")), raw)
        interior = cv2.erode(pred["mask"].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        support = interior & (raw > 0) & np.isfinite(raw)
        if not support.any():
            continue
        distance = float(np.median(raw[support]))
        real = mask & (raw > 0) & np.isfinite(raw)
        err = raw[real] - truth[real]
        sigma = 1.4826 * np.median(abs(err - np.median(err)))
        samples[row["sensor"]].append([distance, float(max(sigma, 0.0001))])
        y, x = np.nonzero(support)
        take = np.linspace(0, len(x) - 1, min(4096, len(x))).astype(int)
        y, x = y[take], x[take]
        observations[row["sensor"]].append(
            (distance, pred["confidence"][y, x], abs(raw[y, x] - truth[y, x]), mask[y, x])
        )
    sensors = {}
    for sensor, sample in samples.items():
        coef = noise_fit(sample)
        shape = (len(RANGE_EDGES) - 1, len(SCORE_EDGES) - 1)
        count, good = np.zeros(shape), np.zeros(shape)
        for d, conf, error, valid in observations[sensor]:
            ri = np.clip(np.searchsorted(RANGE_EDGES, d, side="right") - 1, 0, shape[0] - 1)
            ci = np.clip(np.searchsorted(SCORE_EDGES, conf, side="right") - 1, 0, shape[1] - 1)
            label = valid & (error <= 3 * max(0.0001, coef[0] + coef[1] * d * d) + 0.001)
            np.add.at(count, (np.full(len(ci), ri), ci), 1)
            np.add.at(good, (np.full(len(ci), ri), ci), label.astype(int))
        probability = np.divide(good, count, out=np.zeros_like(good), where=count > 0)
        n = np.maximum(count, 1)
        z = 1.96
        lower = (
            probability
            + z * z / (2 * n)
            - z * np.sqrt(probability * (1 - probability) / n + z * z / (4 * n * n))
        ) / (1 + z * z / n)
        lower[count == 0] = 0
        lower = np.clip(lower, 0, 1)
        sensors[sensor] = {
            "sigma_coefficients": coef.tolist(),
            "probabilities": probability.tolist(),
            "wilson_lower": lower.tolist(),
            "counts": count.astype(int).tolist(),
            "development_noise_samples": sample,
        }
    payload = {
        "schema_version": 1,
        "development_seed": 10607,
        "fit_frames": len(rows),
        "range_edges": RANGE_EDGES,
        "score_edges": SCORE_EDGES,
        "sensors": sensors,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "label": "true liquid pixel AND raw-depth error <= 3 fitted sigma + 1mm",
        "scope": "simulator empirical pixel support reliability, NOT liquid-depth accuracy probability",
        "selection": "Wilson lower >= 0.90 and at least 128 sampled pixels; correlated pixels are not independent confidence trials",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False))
    print(json.dumps({s: v["sigma_coefficients"] for s, v in sensors.items()}))


if __name__ == "__main__":
    main()
