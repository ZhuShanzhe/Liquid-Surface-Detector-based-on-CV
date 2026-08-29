#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

from liquid_depth.camera_qualification import DEFAULT_QUALIFICATION_DISTANCES_M


def _capture_once(incoming: Path, service: str) -> Path:
    before = {path.resolve() for path in incoming.iterdir() if path.is_dir()}
    subprocess.run(
        ["ros2", "service", "call", service, "std_srvs/srv/Trigger"],
        check=True,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        created = [path for path in incoming.iterdir() if path.is_dir() and path.resolve() not in before]
        if len(created) == 1:
            return created[0]
        time.sleep(0.1)
    raise RuntimeError("The capture service did not create exactly one frame directory")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively capture a five-distance diffuse-plane qualification set"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, required=True)
    parser.add_argument(
        "--distances-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUALIFICATION_DISTANCES_M),
    )
    parser.add_argument("--frames-per-distance", type=int, default=60)
    parser.add_argument("--interval-seconds", type=float, default=0.2)
    parser.add_argument("--service", default="/rgbd_frame_saver/save")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    incoming = args.incoming.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    incoming.mkdir(parents=True, exist_ok=True)
    if args.frames_per_distance < 4:
        parser.error("--frames-per-distance must be at least 4")

    for distance_m in args.distances_m:
        target = output / f"{distance_m:g}m"
        target.mkdir(parents=True, exist_ok=True)
        if not args.yes:
            input(
                f"Place a matte plane {distance_m:g} m from the camera optical center, "
                "make it fill the center ROI, then press Enter: "
            )
        for index in range(args.frames_per_distance):
            captured = _capture_once(incoming, args.service)
            destination = target / captured.name
            shutil.move(str(captured), destination)
            print(f"[{distance_m:g} m] {index + 1}/{args.frames_per_distance}: {destination}")
            time.sleep(max(args.interval_seconds, 0.0))


if __name__ == "__main__":
    main()
