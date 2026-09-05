#!/usr/bin/env python3
"""Same-image legacy vs continuous RGB witness; availability and error reported separately."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from liquid_depth.rgb_continuous import RGBContinuousWitness
from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.super_resolution import scaled_intrinsics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cv2.setNumThreads(2)
    groups = defaultdict(list)
    for row in json.loads((args.data / "sequences.json").read_text()):
        groups[row["sequence"]].append(row)
    rows = []
    for name, frames in groups.items():
        witnesses = {}
        for row in frames:
            a = np.load(Path(row["path"]) / "frame.npz")
            high = cv2.imread(str(Path(row["path"]) / "rgb.png"))
            pose = a["camera_to_world"].copy()
            pose[:3, :3] = pose[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            for size in ((320, 180), (1280, 720)):
                rgb = cv2.resize(high, size, interpolation=cv2.INTER_AREA)
                k = scaled_intrinsics(a["intrinsics"], size[0] / 1280)
                mask = cv2.resize(a["truth_mask"], size, interpolation=cv2.INTER_NEAREST)
                for method, cls in (("grid", RGBContourWitness), ("continuous", RGBContinuousWitness)):
                    key = (size, method)
                    if row["index"] == 0:
                        witness = cls()
                        try:
                            witness.calibrate(
                                rgb,
                                mask,
                                row["known_level_m"],
                                k,
                                pose,
                                row["bottom_world_m"],
                                row["radius_x_m"],
                                row["radius_y_m"],
                            )
                        except ValueError:
                            pass
                        witnesses[key] = witness
                        continue
                    for degradation in ("clean", "blur_jpeg", "shift3", "dim"):
                        image = rgb.copy()
                        if degradation == "blur_jpeg":
                            image = cv2.GaussianBlur(image, (5, 5), 0.8)
                            _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        elif degradation == "shift3":
                            image = cv2.warpAffine(image, np.array([[1.0, 0, 3], [0, 1, 0]]), size)
                        elif degradation == "dim":
                            image = (image * 0.35).astype(np.uint8)
                        start = perf_counter()
                        cue = witnesses[key].estimate(image, k, pose, resolution_checks=True)
                        ms = (perf_counter() - start) * 1000
                        available = bool(cue.get("available"))
                        bound = cue.get("error_bound_proxy_m")
                        error = cue["level_m"] - row["truth_m"] if available else None
                        budget = max(0.005, 0.02 * max(0, cue["level_m"] - bound)) if available else 0
                        eligible = bool(available and bound < budget)
                        bad_pass = bool(
                            eligible and abs(row["truth_m"] + 0.02 - cue["level_m"]) <= budget - bound
                        )
                        rows.append(
                            {
                                "sequence": name,
                                "index": row["index"],
                                "distance": row["standoff_m"],
                                "radius": row["radius_x_m"],
                                "width": size[0],
                                "method": method,
                                "degradation": degradation,
                                "available": available,
                                "error_m": error,
                                "bound_m": bound,
                                "eligible": eligible,
                                "bad_plus20mm_pass": bad_pass,
                                "latency_ms": ms,
                            }
                        )
        print(json.dumps({"sequence": name, "done": True}), flush=True)
    buckets = defaultdict(list)
    paired = defaultdict(dict)
    for r in rows:
        group = f"d{r['distance']:g}/{r['width']}/{r['degradation']}"
        buckets[group + "/" + r["method"]].append(r)
        paired[(group, r["sequence"], r["index"])][r["method"]] = r
    summary = {}
    for group, q in buckets.items():
        v = [r for r in q if r["available"]]
        summary[group] = {
            "frames": len(q),
            "available": len(v),
            "eligible": sum(r["eligible"] for r in q),
            "mae_mm": float(np.mean([abs(r["error_m"]) for r in v]) * 1000) if v else None,
            "p95_mm": float(np.percentile([abs(r["error_m"]) for r in v], 95) * 1000) if v else None,
            "bound_violations": sum(abs(r["error_m"]) > r["bound_m"] for r in v),
            "bad_plus20mm_pass": sum(r["bad_plus20mm_pass"] for r in q),
            "latency_p95_ms": float(np.percentile([r["latency_ms"] for r in q], 95)),
        }
    common = defaultdict(list)
    for (group, _, _), pair in paired.items():
        if all(r["available"] for r in pair.values()):
            common[group].append(pair)
    common_out = {
        key: {
            "frames": len(q),
            **{m: float(np.mean([abs(r[m]["error_m"]) for r in q]) * 1000) for m in ("grid", "continuous")},
        }
        for key, q in common.items()
    }
    args.output.write_text(
        json.dumps(
            {
                "scope": "RGB-only paired test; +20mm is synthetic verifier admission probe, not full runtime false-accept rate",
                "data": str(args.data),
                "summaries": summary,
                "common_frame_mae_mm": common_out,
                "frames": rows,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
