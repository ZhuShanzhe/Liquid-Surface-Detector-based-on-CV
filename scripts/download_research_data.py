#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def download(name: str, item: dict, root: Path, accept_license: bool, extract: bool) -> Path:
    if item.get("requires_acceptance") and not accept_license:
        raise RuntimeError(f"{name} requires --accept-license ({item['license']})")
    destination = root / item["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    method = item["method"]
    if method == "manual":
        raise RuntimeError(f"manual download/registration required: {item['url']}")
    if method == "huggingface":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=item["repo_id"],
            repo_type=item.get("repo_type", "dataset"),
            local_dir=destination,
        )
    elif method == "gdrive":
        import gdown

        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / item["filename"]
        if not archive.exists():
            result = gdown.download(id=item["file_id"], output=str(archive), quiet=False)
            if not result:
                raise RuntimeError(f"Google Drive download failed for {name}")
        if extract:
            safe_extract(archive, destination)
    elif method == "git":
        if (destination / ".git").exists():
            subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)
        else:
            subprocess.run(["git", "clone", "--depth", "1", item["url"], str(destination)], check=True)
    else:
        raise ValueError(f"Unknown download method: {method}")

    marker = destination / f".{name}.download.json"
    marker.write_text(
        json.dumps(
            {
                "dataset": name,
                "source": item.get("url", item.get("repo_id", item.get("file_id"))),
                "license": item["license"],
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="License-aware downloader for research baselines and datasets")
    parser.add_argument("--registry", type=Path, default=Path("configs/research_datasets.yaml"))
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/liquid-depth-data/research"))
    parser.add_argument("--dataset", nargs="+", help="Registry names to download")
    parser.add_argument("--list", action="store_true", help="List registry entries without downloading")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--accept-license", action="store_true", help="Acknowledge listed non-commercial licenses")
    parser.add_argument("--extract", action="store_true", help="Safely extract downloaded ZIP archives")
    args = parser.parse_args()
    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8"))["datasets"]

    if args.list or not args.dataset:
        for name, item in registry.items():
            print(
                f"{name:26} priority={item['priority']:6} method={item['method']:11} "
                f"size~{item['expected_size_gb']}GB license={item['license']}"
            )
        if not args.dataset:
            return

    unknown = sorted(set(args.dataset) - set(registry))
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}")
    requested_size = sum(float(registry[name]["expected_size_gb"]) for name in args.dataset)
    free_gb = shutil.disk_usage(args.root.parent if args.root.parent.exists() else Path.cwd()).free / 1e9
    print(f"Requested estimated size: {requested_size:.1f} GB; free space: {free_gb:.1f} GB")
    if requested_size * 1.15 > free_gb:
        raise SystemExit("Insufficient free space including a 15% safety margin")
    if args.dry_run:
        return
    for name in args.dataset:
        try:
            target = download(name, registry[name], args.root, args.accept_license, args.extract)
            print(f"Ready: {name} -> {target}")
        except RuntimeError as exc:
            print(f"Skipped: {name}: {exc}")


if __name__ == "__main__":
    main()
