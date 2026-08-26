#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

TEXT_SOURCE_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".xml",
    ".cfg",
    ".ini",
    ".sh",
    ".cpp",
    ".cc",
    ".c",
    ".hpp",
    ".h",
}


def _git_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(path for path in result.stdout.splitlines() if path)


def _matches_scope(path: str, deposit: dict[str, Any]) -> bool:
    if not any(path.startswith(prefix) for prefix in deposit["include_prefixes"]):
        return False
    if any(path.startswith(prefix) for prefix in deposit["exclude_prefixes"]):
        return False
    if any(path.endswith(suffix) for suffix in deposit["exclude_suffixes"]):
        return False
    return Path(path).suffix.lower() in TEXT_SOURCE_SUFFIXES


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return isinstance(value, str) and "TO_BE_CONFIRMED" in value


def _file_record(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    payload = path.read_bytes()
    text = payload.decode("utf-8")
    return {
        "path": relative_path,
        "bytes": len(payload),
        "lines": len(text.splitlines()),
        "nonblank_lines": sum(bool(line.strip()) for line in text.splitlines()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and inventory the independently developed V1.0 software source candidate"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/software_product_v1.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("version") != 1:
        raise SystemExit("software product config must be version 1")

    tracked_and_candidate = _git_files(root)
    records = [
        _file_record(root, path)
        for path in tracked_and_candidate
        if _matches_scope(path, config["source_deposit"])
    ]
    total_lines = sum(record["lines"] for record in records)
    total_nonblank = sum(record["nonblank_lines"] for record in records)
    candidate_document = root / config["source_deposit"]["documentation_candidate"]
    root_license_files = [name for name in ("LICENSE", "LICENSE.md", "COPYING") if (root / name).exists()]
    ros_package = root / "ros2_ws/src/liquid_depth_camera/package.xml"
    ros_license = None
    if ros_package.exists():
        package_text = ros_package.read_text(encoding="utf-8")
        start = package_text.find("<license>")
        end = package_text.find("</license>")
        if start >= 0 and end > start:
            ros_license = package_text[start + len("<license>") : end].strip()

    checks = [
        {
            "name": "product_metadata_complete",
            "ok": not _contains_placeholder(config),
            "detail": "confirm applicant, dates, publication status, and license policy",
        },
        {
            "name": "candidate_document_exists",
            "ok": candidate_document.exists(),
            "detail": str(candidate_document.relative_to(root)),
        },
        {
            "name": "candidate_source_nonempty",
            "ok": bool(records) and total_nonblank >= 1000,
            "detail": f"{len(records)} files, {total_lines} lines, {total_nonblank} nonblank lines",
        },
        {
            "name": "third_party_excluded",
            "ok": all(not record["path"].startswith("third_party/") for record in records),
            "detail": "third-party source is outside the candidate inventory",
        },
        {
            "name": "root_license_policy_consistent",
            "ok": bool(root_license_files)
            and config["release"]["root_license_policy"] != "TO_BE_CONFIRMED",
            "detail": {
                "root_license_files": root_license_files,
                "configured_policy": config["release"]["root_license_policy"],
                "ros_package_declares": ros_license,
            },
        },
    ]
    report = {
        "ok": all(check["ok"] for check in checks),
        "config": str(config_path.relative_to(root)),
        "proposed_product": config["product"],
        "checks": checks,
        "source_inventory": {
            "files": len(records),
            "lines": total_lines,
            "nonblank_lines": total_nonblank,
            "records": records,
        },
        "excluded_boundaries": {
            "prefixes": config["source_deposit"]["exclude_prefixes"],
            "suffixes": config["source_deposit"]["exclude_suffixes"],
        },
        "filing_note": (
            "This inventory supports engineering review only. Applicant identity, ownership proof, "
            "official forms, page selection, and portal submission require human confirmation."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.strict and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
