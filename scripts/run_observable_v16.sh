#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/envs/liquid-depth/bin/python
OUT=/root/autodl-tmp/liquid-depth-artifacts/training/observable-v16-20260916
DEV=/root/autodl-tmp/liquid-depth-data/synthetic/observable-v16-development/manifest.csv
test -f "$DEV"
"$PY" scripts/train_observable_v16.py --stage locator > "$OUT/locator.log" 2>&1
for route in full gated; do
 "$PY" scripts/train_observable_v16.py --stage depth --route "$route" > "$OUT/depth-$route.log" 2>&1
done
echo TRAINING_FROZEN
/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender -b -t 4 --python scripts/generate_fullframe_v16.py -- --fresh --output /root/autodl-tmp/liquid-depth-data/synthetic/observable-v16-fresh --seed 20260916031 --count 128 > "$OUT/generation-test.log" 2>&1
"$PY" scripts/train_observable_v16.py --stage test --manifest /root/autodl-tmp/liquid-depth-data/synthetic/observable-v16-fresh/manifest.csv > "$OUT/test.log" 2>&1
echo ALL_FINISHED
