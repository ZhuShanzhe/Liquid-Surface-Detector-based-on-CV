#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from liquid_depth.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the reusable liquid-depth runtime before experiments")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    checks = []

    def check(name: str, action) -> None:
        try:
            detail = action()
            checks.append({"name": name, "ok": True, "detail": str(detail)})
        except Exception as exc:
            checks.append({"name": name, "ok": False, "detail": str(exc)})

    check("python", lambda: sys.version.split()[0])
    for module in ("numpy", "cv2", "yaml", "liquid_depth"):
        check(f"import:{module}", lambda module=module: importlib.import_module(module).__file__)
    check("config", lambda: sorted(load_config(args.config)))
    check("disk", lambda: shutil.disk_usage(args.data_root or Path.cwd()).free)
    if args.data_root:
        check("data_root", lambda: args.data_root.resolve(strict=True))

    def cuda_status():
        import torch

        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but unavailable")
        return {
            "torch": torch.__version__,
            "available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    check("cuda", cuda_status)
    report = {"ok": all(item["ok"] for item in checks), "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
