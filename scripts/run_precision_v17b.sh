#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
P=/root/autodl-tmp/envs/liquid-depth/bin/python
$P -m pytest tests/test_precision_v17.py -q
$P -u scripts/evaluate_precision_v17b.py --stage select
/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender -b -t 4 --python scripts/generate_fullframe_v16.py -- --fresh --output /root/autodl-tmp/liquid-depth-data/synthetic/precision-v17b-fresh --seed 20260917081 --count 256 > /root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916/phase-b/generation.log 2>&1
$P -u scripts/evaluate_precision_v17b.py --stage test
echo PHASE_B_FINISHED
