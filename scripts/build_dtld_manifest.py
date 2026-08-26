#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_depth.dtld import write_dtld_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an instance-level DTLD RGB-D/contact-line/liquid-height "
            "manifest without mixing official capture splits"
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Keep annotation rows while recording absent image paths as empty",
    )
    parser.add_argument(
        "--split-map",
        type=Path,
        help="JSON mapping scene IDs such as 000013 to train, val, or test",
    )
    args = parser.parse_args()
    split_map = (
        json.loads(args.split_map.read_text(encoding="utf-8"))
        if args.split_map
        else None
    )
    counts = write_dtld_manifest(
        args.root,
        args.output,
        allow_missing=args.allow_missing,
        split_map=split_map,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "rows_by_split": counts,
                "rows": sum(counts.values()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
