#!/usr/bin/env python3
"""Repair v6 family/result key collision without changing a single estimated value."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from evaluate_long_surface_candidates import summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Preserve the original audit artifact")
    source = json.loads(args.input.read_text())
    family_buckets = defaultdict(list)
    valid_groups = set()
    for row in source["frames"]:
        family = next(
            (s for s in ("active_stereo", "structured_light", "tof") if row["sequence"].endswith("_" + s)),
            None,
        )
        if family is None:
            raise ValueError("Cannot recover family from sequence identity")
        row["sensor_family"] = family
        if isinstance(row.get("sensor"), str):
            del row["sensor"]  # wave rows had metadata here, not a method result
        base = f"{row['motion']}/{row['variant']}"
        valid_groups.update((base, f"{base}/d{row['standoff_m']:g}/h{row['truth_m']:g}", f"{base}/{family}"))
        if row["motion"] == "static":
            family_buckets[f"{base}/{family}"].append(row)
    # Existing global/range summaries are retained bit-for-bit.
    source["summaries"] = {k: v for k, v in source["summaries"].items() if k in valid_groups}
    for key, rows in family_buckets.items():
        source["summaries"][key] = {m: summary(rows, m) for m in ("gravity", "early", "balanced", "sensor")}
    source["metadata_audit"] = {
        "original_path": str(args.input),
        "original_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "change": "Restore sensor_family from sequence identity; rebuild only static family summaries; estimates/global/range summaries unchanged.",
    }
    args.output.write_text(json.dumps(source, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
