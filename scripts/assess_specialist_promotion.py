#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_depth.promotion import (
    PromotionRequirements,
    assess_depth_specialist_promotion,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply industrial promotion gates to a depth specialist"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--max-mae-m", type=float, required=True)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument(
        "--min-within-tolerance-coverage",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--min-mae-improvement-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument("--max-failed-frame-ratio", type=float, default=0.01)
    parser.add_argument("--max-median-latency-ms", type=float, default=500.0)
    parser.add_argument(
        "--guard-scenario",
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    requirements = PromotionRequirements(
        max_mae_m=args.max_mae_m,
        min_prediction_coverage=args.min_coverage,
        min_within_tolerance_coverage=(
            args.min_within_tolerance_coverage
        ),
        min_mae_improvement_fraction=(
            args.min_mae_improvement_fraction
        ),
        max_failed_frame_ratio=args.max_failed_frame_ratio,
        max_median_latency_ms=args.max_median_latency_ms,
    )
    result = assess_depth_specialist_promotion(
        baseline,
        candidate,
        target_scenario=args.scenario,
        requirements=requirements,
        guard_scenarios=tuple(args.guard_scenario),
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["accepted"] else 2)


if __name__ == "__main__":
    main()
