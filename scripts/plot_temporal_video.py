#!/usr/bin/env python3
"""Plot held-out video predictions, preserving missing/rejected outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.report.read_text())["frames"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    cases = [("static", "drop90"), ("moving", "drop90"), ("step", "normal"), ("static", "slow_echo")]
    for axis, (motion, variant) in zip(axes.flat, cases):
        subset = [r for r in rows if r["seed"] == 9323 and r["motion"] == motion and r["variant"] == variant]
        t = np.array([r["index"] / 10 for r in subset])
        axis.plot(t, [r["truth_m"] * 1000 for r in subset], "k--", label="Truth", lw=2)
        for key, label, color in [
            ("network", "Learned depth only", "#c87b2a"),
            ("surface_memory", "Guarded adaptive memory", "#167a76"),
        ]:
            values = [r[key]["level_m"] * 1000 if r[key]["accepted"] else np.nan for r in subset]
            axis.plot(t, values, label=label, color=color, lw=1.7)
        for r in subset:
            if not r["surface_memory"]["accepted"]:
                axis.axvspan(
                    r["index"] / 10 - 0.05, r["index"] / 10 + 0.05, color="#c74646", alpha=0.13, lw=0
                )
        axis.set(
            title=f"{motion} / {variant}", xlabel="Time (s)", ylabel="Liquid depth (mm)", ylim=(180, 480)
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Held-out simulation: reliable outputs and unresolved drift\nRed shading = rejected adaptive output",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
