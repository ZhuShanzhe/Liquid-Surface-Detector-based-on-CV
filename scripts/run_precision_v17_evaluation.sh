#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/envs/liquid-depth/bin/python
OUT=/root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916
test -f "$OUT/r4/frozen.json"
test -f "$OUT/r0/frozen.json"
"$PY" scripts/evaluate_precision_v17.py --stage select > "$OUT/selection.log" 2>&1
/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender -b -t 4 --python scripts/generate_fullframe_v16.py -- --fresh --output /root/autodl-tmp/liquid-depth-data/synthetic/precision-v17-fresh --seed 20260917071 --count 256 > "$OUT/generation-test.log" 2>&1
"$PY" scripts/evaluate_precision_v17.py --stage test --manifest /root/autodl-tmp/liquid-depth-data/synthetic/precision-v17-fresh/manifest.csv > "$OUT/test.log" 2>&1
echo ALL_FINISHED
