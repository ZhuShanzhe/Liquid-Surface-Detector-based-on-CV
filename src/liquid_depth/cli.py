from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import fit_bottom, infer_frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="liquid-depth", description="RGB-D liquid depth pipeline")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Pipeline YAML configuration")
    commands = parser.add_subparsers(dest="command", required=True)

    bottom = commands.add_parser("fit-bottom", help="Fit and save the empty-container bottom plane")
    bottom.add_argument("--frame", required=True, help="Empty-container RGB-D frame directory")
    bottom.add_argument("--output", required=True, help="Bottom plane JSON output path")

    infer = commands.add_parser("infer", help="Estimate liquid depth for one RGB-D frame")
    infer.add_argument("--frame", required=True, help="RGB-D frame directory")
    infer.add_argument("--bottom-plane", required=True, help="Bottom plane JSON from fit-bottom")
    infer.add_argument("--output-dir", required=True, help="Directory for inference artifacts")

    batch = commands.add_parser("batch", help="Estimate every child frame directory")
    batch.add_argument("--input-dir", required=True)
    batch.add_argument("--bottom-plane", required=True)
    batch.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    if args.command == "fit-bottom":
        result = fit_bottom(args.frame, args.output, config)
        print(json.dumps(result, indent=2))
        return
    if args.command == "infer":
        result = infer_frame(args.frame, args.bottom_plane, args.output_dir, config)
        print(json.dumps(result, indent=2))
        return

    input_dir = Path(args.input_dir)
    results = []
    for frame in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        required = (frame / "rgb.png", frame / "depth.npy", frame / "depth_info.json")
        if not all(path.exists() for path in required):
            continue
        result = infer_frame(frame, args.bottom_plane, Path(args.output_dir) / frame.name, config)
        results.append(result)
        print(f"{frame.name}: {result['liquid_depth']:.3f} {result['liquid_depth_unit']}")
    print(f"Processed {len(results)} frame(s)")


if __name__ == "__main__":
    main()

