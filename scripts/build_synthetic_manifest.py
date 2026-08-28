#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_depth.simulation import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a synthetic liquid dataset manifest")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "manifest.csv"
    rows = build_manifest(args.root, output)
    print(json.dumps({"output": output.resolve().as_posix(), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
