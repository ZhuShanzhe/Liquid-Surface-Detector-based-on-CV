#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import yaml


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(name: str) -> None:
    normalized = PurePosixPath(name.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"Unsafe archive member: {name}")


def verify_file(path: Path, spec: dict) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    expected_bytes = spec.get("expected_bytes")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"Size mismatch for {path}: {actual_bytes} != {expected_bytes}")
    checksum_spec = spec.get("checksum")
    actual_checksum = None
    if checksum_spec:
        algorithm, expected_checksum = checksum_spec.split(":", maxsplit=1)
        actual_checksum = checksum(path, algorithm)
        if actual_checksum.lower() != expected_checksum.lower():
            raise ValueError(f"Checksum mismatch for {path}: {actual_checksum} != {expected_checksum}")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "checksum": actual_checksum,
    }


def test_archive(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"Corrupt ZIP member in {path}: {bad}")
            for member in archive.infolist():
                safe_member_path(member.filename)
        return "zip"
    if lower.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                safe_member_path(member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Archive links are not extracted: {member.name}")
        return "tar"
    if lower.endswith(".7z"):
        if shutil.which("7z") is None:
            raise RuntimeError("7z is required for .7z archives; install p7zip-full")
        subprocess.run(
            ["7z", "t", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        listing = subprocess.run(
            ["7z", "l", "-slt", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for line in listing.splitlines():
            if line.startswith("Path = "):
                member = line.removeprefix("Path = ")
                if member != str(path):
                    safe_member_path(member)
        return "7z"
    return "plain"


def extract_archive(path: Path, archive_type: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive_type == "zip":
        with zipfile.ZipFile(path) as archive:
            archive.extractall(destination)
    elif archive_type == "tar":
        with tarfile.open(path) as archive:
            archive.extractall(destination)
    elif archive_type == "7z":
        subprocess.run(
            ["7z", "x", str(path), f"-o{destination}", "-y"],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and safely extract registered research archives")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/research_datasets.yaml"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/liquid-depth-data/research"),
    )
    parser.add_argument("--dataset", nargs="+", required=True)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract only after size, checksum, and archive tests pass",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override extraction root; defaults to <dataset destination>/extracted",
    )
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["datasets"]
    report: dict[str, object] = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
    }
    for name in args.dataset:
        item = registry.get(name)
        if item is None:
            raise SystemExit(f"Unknown dataset: {name}")
        if item["method"] != "http":
            raise SystemExit(f"{name} does not register HTTP archive files")
        archive_root = args.root / item["destination"]
        extraction_root = args.output_root or archive_root.parent / "extracted"
        dataset_report = []
        for spec in item["files"]:
            path = archive_root / spec["filename"]
            record = verify_file(path, spec)
            archive_type = test_archive(path)
            record["archive_type"] = archive_type
            if args.extract and archive_type != "plain":
                destination = extraction_root / path.name
                for suffix in (".tar.gz", ".tgz", ".tar", ".zip", ".7z"):
                    if destination.name.lower().endswith(suffix):
                        destination = destination.with_name(destination.name[: -len(suffix)])
                        break
                extract_archive(path, archive_type, destination)
                record["extracted_to"] = str(destination)
            dataset_report.append(record)
            print(f"Verified: {path} ({archive_type})")
        report["datasets"][name] = dataset_report
        if args.extract:
            marker = extraction_root / f".{name}.complete"
            marker.write_text(report["verified_at"] + "\n", encoding="utf-8")

    report_dir = args.root / "manifests"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_name = "archive_verification_" + "-".join(args.dataset) + ".json"
    report_path = report_dir / report_name
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    print(f"Free disk: {shutil.disk_usage(args.root).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
