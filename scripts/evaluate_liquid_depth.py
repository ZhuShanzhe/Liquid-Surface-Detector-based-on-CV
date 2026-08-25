#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_depth.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end liquid-depth predictions")
    parser.add_argument("--ground-truth", required=True, help="CSV: frame_id,depth[,scenario]")
    parser.add_argument("--predictions", required=True, help="Root containing depth_result.json files")
    parser.add_argument("--tolerance", type=float, default=1.0, help="Allowed absolute error in CSV depth units")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = evaluate(args.ground_truth, args.predictions, args.tolerance)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
