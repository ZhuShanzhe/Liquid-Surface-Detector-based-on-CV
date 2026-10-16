"""Plot paired V51 absolute and relative liquid-height error distributions."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ABS_EDGES_M = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, np.inf])
REL_EDGES = np.array([0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, np.inf])
ABS_LABELS = ["0–5", "5–10", "10–20", "20–50", "50–100", "100–200", ">200"]
REL_LABELS = ["0–0.5", "0.5–1", "1–2", "2–3", "3–5", "5–10", ">10"]


def series(rows, loss):
    picked = [
        r for r in rows
        if r["paired_eligible"] and r["target_loss_pct"] == loss
        and r["broad_container_median_m"] is not None
    ]
    absolute = np.asarray([
        abs(r["broad_container_median_m"] - r["true_height_m"]) for r in picked
    ])
    relative = np.asarray([
        abs(r["broad_container_median_m"] - r["true_height_m"]) / r["true_height_m"]
        for r in picked if r["true_height_m"] > 0
    ])
    return picked, absolute, relative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.open(encoding="utf-8")]
    sample = {level: series(rows, level) for level in (0, 50)}
    assert all(len(sample[level][0]) == 353 for level in (0, 50))

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = {0: "#2374AB", 50: "#D97732"}
    positions = np.arange(7)
    width = 0.38
    for level in (0, 50):
        _, abs_error, rel_error = sample[level]
        shift = (-0.5 if level == 0 else 0.5) * width
        axes[0].bar(
            positions + shift,
            np.histogram(abs_error, bins=ABS_EDGES_M)[0] / len(abs_error) * 100,
            width=width, color=colors[level], label=f"{level}% missing",
        )
        axes[1].bar(
            positions + shift,
            np.histogram(rel_error, bins=REL_EDGES)[0] / len(rel_error) * 100,
            width=width, color=colors[level], label=f"{level}% missing",
        )
        x = np.sort(abs_error * 1000)
        y = np.arange(1, len(x) + 1) / len(x) * 100
        axes[2].step(x, y, where="post", linewidth=2.0, color=colors[level],
                     label=f"{level}% missing (P95 {np.percentile(x, 95):.1f} mm)")

    axes[0].set_title("Absolute-error intervals")
    axes[0].set_ylabel("Share of numeric frames (%)")
    axes[0].set_xlabel("Absolute error (mm)")
    axes[0].set_xticks(positions, ABS_LABELS, rotation=35, ha="right")
    axes[0].set_ylim(0, 45)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].set_title("Relative-error intervals")
    axes[1].set_ylabel("Share of nonempty frames (%)")
    axes[1].set_xlabel("Relative error (%)")
    axes[1].set_xticks(positions, REL_LABELS, rotation=35, ha="right")
    axes[1].set_ylim(0, 75)

    axes[2].set_title("Cumulative absolute error")
    axes[2].set_ylabel("Frames at or below error (%)")
    axes[2].set_xlabel("Absolute error (mm)")
    axes[2].set_xlim(0, 200)
    axes[2].set_ylim(0, 100)
    axes[2].legend(frameon=False, fontsize=9, loc="lower right")
    for ax in axes:
        ax.grid(axis="y", color="#D7DEE5", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Ordinary-scene candidate-free estimates · paired V50 test · n=353 per level",
                 fontsize=14)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    levels = (0, 10, 20, 30, 40, 50)
    all_samples = {level: series(rows, level) for level in levels}
    assert all(len(all_samples[level][0]) == 353 for level in levels)
    abs_grid = np.stack([
        np.histogram(all_samples[level][1], ABS_EDGES_M)[0] / 353 * 100
        for level in levels
    ])
    rel_grid = np.stack([
        np.histogram(all_samples[level][2], REL_EDGES)[0] /
        len(all_samples[level][2]) * 100 for level in levels
    ])
    fig2, heat_axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, grid, labels, title, denominator in zip(
        heat_axes,
        (abs_grid, rel_grid),
        (ABS_LABELS, REL_LABELS),
        ("Absolute-error distribution", "Relative-error distribution"),
        ("all numeric frames", "nonempty frames"),
    ):
        ax.imshow(grid, cmap="Blues", vmin=0, vmax=70, aspect="auto")
        ax.set_title(f"{title} (% of {denominator})")
        ax.set_xticks(np.arange(7), labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(6), [f"{level}% missing" for level in levels])
        ax.set_xlabel("Absolute error (mm)" if ax is heat_axes[0] else "Relative error (%)")
        ax.set_ylabel("Target missing-surface rate")
        for row in range(6):
            for col in range(7):
                value = grid[row, col]
                ax.text(col, row, f"{value:.1f}", ha="center", va="center",
                        color="white" if value > 38 else "#132A3A", fontsize=9)
    fig2.suptitle("Candidate-free estimate error shares · paired V50 test · n=353 per level",
                  fontsize=14)
    all_levels_prefix = args.output_prefix.with_name(args.output_prefix.name + "_all_levels")
    fig2.savefig(all_levels_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig2.savefig(all_levels_prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig2)
    for level in (0, 50):
        picked, abs_error, rel_error = sample[level]
        print(json.dumps({
            "loss_pct": level,
            "n": len(picked),
            "mae_mm": round(abs_error.mean() * 1000, 2),
            "p95_mm": round(np.percentile(abs_error, 95) * 1000, 2),
            "absolute_bins_pct": (np.histogram(abs_error, ABS_EDGES_M)[0] / len(abs_error) * 100).round(2).tolist(),
            "relative_bins_pct": (np.histogram(rel_error, REL_EDGES)[0] / len(rel_error) * 100).round(2).tolist(),
        }))


if __name__ == "__main__":
    main()
