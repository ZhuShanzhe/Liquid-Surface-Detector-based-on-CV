"""Independent floating-only installations with consistent stereo/SL/ToF proxies."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_float_tail_v3 as source

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--installations', type=int, required=True)
    p.add_argument('--worker', type=int, default=0); p.add_argument('--workers', type=int, default=1)
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:]); a.root.mkdir(parents=True, exist_ok=True)
    source.capture.RANDOM_HEADROOM = True
    source.capture.previous.make_scene = source.capture.expanded_scene
    source.capture.gen.create_floating_objects = source.small_floats
    for slot in range(a.worker, a.installations, a.workers):
        case = 3+6*slot
        source.capture.previous.render_one(a.root, case, a.seed, 2)
        source.capture.augment_observations(a.root, case, a.seed, True)
