"""Continue the completed capture after a corrected seed-initialization failure."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path("/root/autodl-tmp")
REPO = BASE / "Liquid-Surface-Detector-based-on-CV"
OUT = BASE / "liquid-depth-artifacts/visual-route-v1-20261015"
ROOT = BASE / "liquid-depth-data/synthetic/visual-evidence-v1-20261015"
PY = BASE / "envs/liquid-depth/bin/python"
SEED = 202610153007


def status(stage, **kw):
    (OUT / "status.json").write_text(
        json.dumps({"stage": stage, "updated_unix": time.time(), **kw}, indent=2)
    )


def main():
    if len(list(ROOT.glob("*/frames.json"))) != 594:
        raise RuntimeError("complete_capture_required")
    os.chdir(REPO)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}
    if not (OUT / "model-v1/best.pt").exists():
        status("training", note="seed initialization corrected; capture unchanged")
        with (OUT / "training-v1.log").open("w") as log:
            subprocess.run(
                [
                    str(PY),
                    "scripts/train_visual_regions_v1.py",
                    "--root",
                    str(ROOT),
                    "--output",
                    str(OUT / "model-v1"),
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
    with (OUT / "evaluation-v1.log").open("w") as log:
        subprocess.run(
            [
                str(PY),
                "scripts/evaluate_visual_route_v1.py",
                "--root",
                str(ROOT),
                "--model",
                str(OUT / "model-v1/best.pt"),
                "--output",
                str(OUT / "evaluation-v1"),
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
        status("failed", error=repr(exc))
        raise
