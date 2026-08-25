#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_depth.depth_evaluation import evaluate_depth_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate depth restoration baselines with one metric contract"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_depth_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
