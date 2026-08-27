#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _cpu_benchmark(action, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        action()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        action()
        values.append((time.perf_counter() - started) * 1000.0)
    return _percentiles(values)


def _cuda_benchmark(action, torch, warmup: int, iterations: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            action()
        torch.cuda.synchronize()
        values = []
        for _ in range(iterations):
            started = time.perf_counter()
            action()
            torch.cuda.synchronize()
            values.append((time.perf_counter() - started) * 1000.0)
    return _percentiles(values)


def _benchmark_seegroup(args, torch) -> dict[str, object]:
    source = args.seegroup_source.expanduser().resolve()
    checkpoint = args.seegroup_checkpoint.expanduser().resolve()
    if not source.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("SeeGroup source or checkpoint is missing")
    sys.path.insert(0, source.as_posix())
    from model import init_model
    from util.config import get_config_from_path

    config = get_config_from_path((source / "config" / "val.py").as_posix())
    config["local_rank"] = 0
    config["rank"] = 0
    config["cli"] = "eval"
    config["resumed_from"] = checkpoint.as_posix()
    model = init_model(config).eval()
    image = torch.rand(1, 3, args.height, args.width, device="cuda")
    results = {}
    for input_size in args.seegroup_input_sizes:
        torch.cuda.reset_peak_memory_stats()
        timing = _cuda_benchmark(
            lambda input_size=input_size: model.forward_unscale(
                {"image": image}, input_size=input_size
            ),
            torch,
            args.warmup,
            args.iterations,
        )
        timing["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
        results[str(input_size)] = timing
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "input_sizes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark complex-scene additions at deployment resolution"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--feature-channels", type=int, default=32)
    parser.add_argument("--feature-downsample", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seegroup-source", type=Path)
    parser.add_argument("--seegroup-checkpoint", type=Path)
    parser.add_argument(
        "--seegroup-input-sizes",
        type=int,
        nargs="+",
        default=(322, 392, 518),
    )
    args = parser.parse_args()

    import torch

    from liquid_depth.illumination import (
        adaptive_exposure_correction,
        measure_illumination,
    )
    from liquid_depth.models.layered import RayLayerHead

    rng = np.random.default_rng(2026)
    image = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
    dark_image = (image.astype(np.float32) * 0.2).astype(np.uint8)
    report: dict[str, object] = {
        "resolution": [args.width, args.height],
        "cpu": {
            "illumination_measurement": _cpu_benchmark(
                lambda: measure_illumination(image), args.warmup, args.iterations
            ),
            "conditional_gamma_correction": _cpu_benchmark(
                lambda: adaptive_exposure_correction(
                    dark_image,
                    {"enabled": True, "clahe_clip_limit": 0.0},
                ),
                args.warmup,
                args.iterations,
            ),
        },
    }
    if torch.cuda.is_available():
        feature_height = max(1, args.height // args.feature_downsample)
        feature_width = max(1, args.width // args.feature_downsample)
        head = RayLayerHead(args.feature_channels).cuda().eval()
        features = torch.randn(
            1,
            args.feature_channels,
            feature_height,
            feature_width,
            device="cuda",
        )
        report["cuda"] = {
            "device": torch.cuda.get_device_name(0),
            "ray_layer_head": {
                **_cuda_benchmark(
                    lambda: head(features), torch, args.warmup, args.iterations
                ),
                "parameters": sum(parameter.numel() for parameter in head.parameters()),
            },
        }
        if args.seegroup_source or args.seegroup_checkpoint:
            if not args.seegroup_source or not args.seegroup_checkpoint:
                raise ValueError("Both SeeGroup paths must be provided")
            report["cuda"]["seegroup"] = _benchmark_seegroup(args, torch)
    elif args.seegroup_source or args.seegroup_checkpoint:
        raise RuntimeError("CUDA is required to benchmark SeeGroup")

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
