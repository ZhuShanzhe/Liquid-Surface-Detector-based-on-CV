#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def matches(root: Path, patterns: list[str]) -> list[str]:
    found: set[str] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            control_file = Path(str(path) + ".aria2")
            if path.exists() and path.suffix != ".aria2" and not control_file.exists():
                found.add(str(path))
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official transparent-depth baseline readiness")
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument(
        "--research-root", type=Path, default=Path("/root/autodl-tmp/liquid-depth-data/research")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report = {"baselines": {}, "selection_metrics": config["selection_metrics"]}
    for name, item in config["baselines"].items():
        source = args.research_root / item["source"]
        current_revision = revision(source)
        data = matches(args.research_root, item.get("data_globs", []))
        checkpoints = matches(args.research_root, item.get("checkpoint_globs", []))
        expected = str(item["expected_revision"])
        report["baselines"][name] = {
            "source_ready": current_revision is not None,
            "revision": current_revision,
            "revision_matches": bool(current_revision and current_revision.startswith(expected)),
            "data_ready": bool(data),
            "checkpoint_ready": bool(checkpoints),
            "data_matches": data[:20],
            "checkpoint_matches": checkpoints[:20],
            "adaptation": item["adaptation"],
        }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
