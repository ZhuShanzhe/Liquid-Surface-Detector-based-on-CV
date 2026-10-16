"""Resumable-status first pilot. Leaves old code and archived results untouched."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path("/root/autodl-tmp")
REPO = BASE / "Liquid-Surface-Detector-based-on-CV"
ROOT = BASE / "liquid-depth-data/synthetic/visual-evidence-v1-20261015"
OUT = BASE / "liquid-depth-artifacts/visual-route-v1-20261015"
PY = BASE / "envs/liquid-depth/bin/python"
BLENDER = BASE / "tools/blender/blender-4.2.0-linux-x64/blender"
SEED = 202610153007


def status(stage, **kw):
    (OUT / "status.json").write_text(
        json.dumps({"stage": stage, "updated_unix": time.time(), **kw}, indent=2)
    )
    print(stage, kw, flush=True)


def main():
    archive = BASE / "liquid-depth-archives/pre-visual-route-20261015/manifest.json"
    if not archive.exists():
        raise RuntimeError("verified_baseline_archive_required")
    OUT.mkdir(parents=True, exist_ok=False)
    os.chdir(REPO)
    plan = {
        "seed": SEED,
        "installations": 594,
        "repeats": 2,
        "train": 198,
        "validation": 198,
        "test": 198,
        "scenes": ["ordinary", "low_light", "glare", "floating_objects", "uneven_surface", "depth_failure"],
        "sensors": ["active_stereo", "structured_light", "tof"],
        "prior_calibration": "one real simulated dry floor + noisy operator outline and independent metric anchor",
        "liquid_tolerance": "max(20mm,2% actual bottom-relative liquid depth)",
        "highest_tolerance": "max(50mm,5% highest bottom-relative height)",
        "eligibility_frozen_before_test": {
            "min_returns": 24,
            "min_tiles": 3,
            "relative_candidate_margin": 0.15,
            "max_uninformative_rgb_fraction": 0.95,
        },
        "eligibility_note": "Input-only selector shared across all methods. Full-frame scores and excluded counts also retained. Candidate confidence is not probability of metrological correctness.",
        "transparency": "Archived five-prior refraction route retained; this pilot does not claim a new transparent-liquid model.",
        "baseline_archive": str(archive),
        "epochs": 60,
    }
    (OUT / "experiment-plan.json").write_text(json.dumps(plan, indent=2))
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
    processes = []
    logs = []
    status("generating")
    for worker in range(4):
        log = (OUT / f"generate-worker-{worker}.log").open("w")
        logs.append(log)
        cmd = [
            str(BLENDER),
            "-b",
            "-t",
            "4",
            "--python",
            "scripts/generate_visual_evidence_v1.py",
            "--",
            "--root",
            str(ROOT),
            "--seed",
            str(SEED),
            "--count",
            "594",
            "--worker",
            str(worker),
            "--workers",
            "4",
        ]
        processes.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        status(
            "generating",
            complete_installations=len(list(ROOT.glob("*/frames.json"))) if ROOT.exists() else 0,
            workers_alive=sum(p.poll() is None for p in processes),
        )
        time.sleep(20)
    for log in logs:
        log.close()
    if any(p.returncode for p in processes) or len(list(ROOT.glob("*/frames.json"))) != 594:
        raise RuntimeError("generation_incomplete_check_worker_logs")
    status("training")
    with (OUT / "training.log").open("w") as log:
        subprocess.run(
            [
                str(PY),
                "scripts/train_visual_regions_v1.py",
                "--root",
                str(ROOT),
                "--output",
                str(OUT / "model"),
                "--seed",
                str(SEED),
                "--epochs",
                "60",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    status("evaluating")
    with (OUT / "evaluation.log").open("w") as log:
        subprocess.run(
            [
                str(PY),
                "scripts/evaluate_visual_route_v1.py",
                "--root",
                str(ROOT),
                "--model",
                str(OUT / "model/best.pt"),
                "--output",
                str(OUT / "evaluation"),
                "--seed",
                str(SEED),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    sources = list((REPO / "src/liquid_depth").glob("*visual*py")) + [
        REPO / "scripts/generate_visual_evidence_v1.py",
        REPO / "scripts/train_visual_regions_v1.py",
    ]
    (OUT / "source-sha256.json").write_text(
        json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)
    )
    status("complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            status("failed", error=repr(exc))
        raise
