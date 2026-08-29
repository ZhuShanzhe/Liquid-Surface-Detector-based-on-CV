#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from liquid_depth.virtual_camera import replay_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay synthetic RGB-D samples as ROS-compatible capture directories"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--scenarios", nargs="*")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    captures = replay_manifest(
        args.manifest,
        args.output,
        split=args.split,
        scenarios=args.scenarios,
        limit=args.frames,
    )
    print(
        json.dumps(
            {
                "mode": "virtual_rgbd_no_usb",
                "frames": len(captures),
                "output": args.output.resolve().as_posix(),
                "latest": captures[-1].as_posix(),
                "hardware_validated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
