#!/usr/bin/env bash
set -eu
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/envs/liquid-depth/bin/python
OUT=/root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916
"$PY" -m pytest tests/test_precision_v17.py -q > "$OUT/contract-tests.log" 2>&1
for radius in 4 0; do
 "$PY" scripts/train_precision_v17.py --stage train --radius "$radius" > "$OUT/train-r$radius.log" 2>&1
done
echo TRAINING_FINISHED
