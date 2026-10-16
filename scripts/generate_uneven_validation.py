"""Blender entry point for new uneven wet frames and matched dry geometry."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_fullframe_v16 import context_proxy
from generate_other_scenes_v34 import generate, make_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wet-root", type=Path, required=True)
    p.add_argument("--dry-root", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cases", type=int, default=99)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    a.wet_root.mkdir(parents=True, exist_ok=True)
    a.dry_root.mkdir(parents=True, exist_ok=True)
    for case in range(a.worker, a.cases, a.workers):
        slot = case * 6 + 4
        generate(a.wet_root, slot, a.seed, a.repeats)
        wet_meta = json.loads((a.wet_root / f"{slot:05d}" / "metadata.json").read_text())
        reference = a.wet_root / f"{case * 6:05d}"
        reference.mkdir(exist_ok=False)
        metadata = {**wet_meta, "scenario": "ordinary", "slot": case * 6, "calibration_metadata_only": True}
        (reference / "metadata.json").write_text(json.dumps(metadata, indent=2))
        scene, _ = make_scene(case * 6, a.seed)
        gen.clear_scene()
        rng = random.Random(a.seed + case * 104729 + 47)
        gen.create_environment(scene, rng)
        gen.create_container(scene)
        gen.create_camera(scene)
        gen.create_lighting(scene, rng)
        zeros = np.zeros((scene.height, scene.width), np.float32)
        dry, _, _, _, _ = context_proxy(scene, zeros, zeros, zeros.astype(np.uint8))
        folder = a.dry_root / f"{case:04d}"
        folder.mkdir(exist_ok=False)
        np.save(folder / "dry_depth_m.npy", dry, allow_pickle=False)
        (folder / "metadata.json").write_text(
            json.dumps(
                {
                    "case": case,
                    "seed": a.seed,
                    "contains_liquid": False,
                    "matched_camera_and_geometry": True,
                    "one_dry_observation": True,
                },
                indent=2,
            )
        )
        print(
            json.dumps(
                {"case": case, "seed": a.seed, "repeats": a.repeats, "dry_valid": int((dry > 0).sum())}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
