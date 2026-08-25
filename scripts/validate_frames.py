#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from liquid_depth.io import load_frame


def inspect(frame_dir: Path) -> dict:
    frame = load_frame(frame_dir)
    if frame.depth.shape != frame.rgb_bgr.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {frame.rgb_bgr.shape[:2]} vs {frame.depth.shape}")
    if frame.camera_matrix.shape != (3, 3) or not np.isfinite(frame.camera_matrix).all():
        raise ValueError("Camera matrix must be a finite 3x3 matrix")
    depth = frame.depth.astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    return {
        "frame_id": frame.frame_id,
        "shape": list(frame.depth.shape),
        "valid_depth_ratio": float(valid.mean()),
        "median_positive_depth_raw": float(np.median(depth[valid])) if np.any(valid) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the stable RGB-D frame-directory contract")
    parser.add_argument("root", type=Path, help="One frame directory or a root containing frame directories")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args()
    candidates = [args.root] if (args.root / "rgb.png").exists() else sorted(p for p in args.root.iterdir() if p.is_dir())
    valid, failures = [], []
    for candidate in candidates:
        try:
            valid.append(inspect(candidate))
        except Exception as exc:  # report all bad captures in one pass
            failures.append({"path": str(candidate), "error": str(exc)})
    report = {"root": str(args.root.resolve()), "valid_frames": valid, "failures": failures}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if failures or not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
