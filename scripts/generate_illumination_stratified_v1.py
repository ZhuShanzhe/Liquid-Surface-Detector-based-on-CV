"""Stratified top-view glare and low-light RGB-D validation scenes.

The case geometry and dry RGB-D reference match the V34 generator, while actual
illumination/exposure and sensor corruption vary together across three levels.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_other_scenes_v34 import generate, make_scene

SCENES = ("low_light", "glare")
LEVELS = ("mild", "moderate", "severe")
LIGHT = (.18, .08, .03)
EXPOSURE = (-1.4, -2.6, -3.8)
SEVERITY = (.18, .48, .78)
GLARE_LIGHT = (.2, .45, 1.0)
GLARE_EXPOSURE = (-1.3, -.5, .2)
ROUGHNESS = (.12, .055, .018)


def stratified_scene(case: int, scenario: str, level: int, seed: int):
    offset = 1 if scenario == "low_light" else 2
    base_slot = case * 6 + offset
    slot = (case * 3 + level) * 6 + offset
    scene, extra = make_scene(base_slot, seed)
    scene = replace(scene, index=slot, corruption_severity=SEVERITY[level])
    if scenario == "low_light":
        scene = replace(scene, light_level=LIGHT[level],
                        exposure=EXPOSURE[level])
    else:
        scene = replace(scene, liquid_roughness=ROUGHNESS[level],
                        light_level=GLARE_LIGHT[level],
                        exposure=GLARE_EXPOSURE[level])
    extra = {**extra, "illumination_level": LEVELS[level],
             "illumination_level_index": level,
             "illumination_control": {
                 "corruption_severity": scene.corruption_severity,
                 "light_level": scene.light_level,
                 "exposure": scene.exposure,
                 "liquid_roughness": scene.liquid_roughness,
             }}
    return slot, (scene, extra)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cases", type=int, default=99)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.cases > 99 or args.cases < 1:
        raise ValueError("stratified_protocol_requires_1_to_99_cases")
    args.root.mkdir(parents=True, exist_ok=True)
    for case in range(args.worker, args.cases, args.workers):
        for level in range(len(LEVELS)):
            for scenario in SCENES:
                slot, override = stratified_scene(case, scenario, level, args.seed)
                count = generate(args.root, slot, args.seed, args.repeats,
                                 scene_override=override)
                print(json.dumps({"case": case, "scenario": scenario,
                                  "level": override[1]["illumination_level"],
                                  "frames": count}), flush=True)


if __name__ == "__main__":
    main()
