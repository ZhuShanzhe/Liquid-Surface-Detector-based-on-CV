#!/usr/bin/env python3
"""Condense v5 artifacts without choosing deployment thresholds on test data."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from evaluate_long_surface_candidates import summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--long", type=Path, required=True)
    p.add_argument("--sr", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    long, sr = json.loads(a.long.read_text()), json.loads(a.sr.read_text())
    rows = long["frames"]
    methods = ("legacy", "free", "gravity", "early", "wave")
    buckets = defaultdict(list)
    for r in rows:
        base = f"{r['motion']}/{r['variant']}"
        buckets[f"{base}/d{r['standoff_m']:g}/h{r['truth_m']:g}"].append(r)
        if r["free"]["available"] and not r["legacy"]["available"]:
            buckets[f"{base}/newly_available"].append(r)
        if r["free"]["available"] and r["free"]["tilt_deg"] > 12:
            buckets[f"{base}/tilt_over_12"].append(r)
        if r["early"]["history_used"]:
            buckets[f"{base}/history_used"].append(r)
    results = {}
    for name, q in buckets.items():
        results[name] = {m: summary(q, m) for m in methods}
        if name.endswith("tilt_over_12"):
            err = np.array([abs(r["free"]["local_level_m"] - r["truth"]["mean_depth_m"]) for r in q])
            tol = np.array([max(0.005, 0.02 * abs(r["truth"]["mean_depth_m"])) for r in q])
            results[name]["free_local_center"] = {
                "frames": len(q),
                "mae_mm": float(err.mean() * 1000),
                "within_tolerance": float(np.mean(err <= tol)),
                "over_100mm": float(np.mean(err > 0.1)),
            }
    # Compare SR on identical available images, not dissimilar accepted subsets.
    sr_buckets = defaultdict(dict)
    for r in sr["frames"]:
        sr_buckets[(r["sequence"], r["index"], r["degradation"])][r["method"]] = r
    common = defaultdict(list)
    for frame in sr_buckets.values():
        if all(frame[m]["available"] for m in ("low", "bicubic", "swinir_guarded", "native")):
            common[f"d{frame['low']['distance']:g}/{frame['low']['degradation']}"].append(frame)
    paired = {}
    for name, frames in common.items():
        paired[name] = {
            m: {
                "frames": len(frames),
                "mae_mm": float(np.mean([abs(f[m]["error_m"]) for f in frames]) * 1000),
                "resolution_eligible": sum(f[m]["resolution_eligible"] for f in frames),
            }
            for m in ("low", "bicubic", "swinir_guarded", "native")
        }
    result = {
        "schema_version": 1,
        "scored_frames": long["scored_frames"],
        "protocol": long["protocol"],
        "checkpoint_sha256": long["checkpoint_sha256"],
        "source_artifacts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.long, a.sr)
        },
        "long_summaries": long["summaries"],
        "diagnostics": results,
        "sr_summaries": sr["summaries"],
        "sr_common_frame_comparison": paired,
        "sr_checkpoint_sha256": sr["checkpoint_sha256"],
        "warning": "Candidate coverage is not trusted output coverage; conditional errors are not directly comparable across different availability masks.",
    }
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
