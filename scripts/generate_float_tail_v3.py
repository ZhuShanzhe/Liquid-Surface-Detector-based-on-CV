"""Small-target enriched captures; reuse locked sensor/geometry formation."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_adaptive_evidence_v2 as capture


class SmallTargetRandom:
    def __init__(self, rng): self.rng = rng
    def __getattr__(self, key): return getattr(self.rng, key)
    def choice(self, sequence):
        if sequence == [3, 5, 8, 12, 20, 32, 48]:
            return self.rng.choice([3, 4, 5, 6, 8, 8, 12, 16, 24, 32, 48])
        return self.rng.choice(sequence)


def small_floats(scene, rng):
    capture.diverse_floats(scene, SmallTargetRandom(rng))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.root.mkdir(parents=True, exist_ok=True)
    capture.RANDOM_HEADROOM = True
    capture.previous.make_scene = capture.expanded_scene
    capture.gen.create_floating_objects = small_floats
    for index in range(args.start + args.worker, args.count, args.workers):
        capture.previous.render_one(args.root, index, args.seed, 2)
        capture.augment_observations(args.root, index, args.seed, True)
