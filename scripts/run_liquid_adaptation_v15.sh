set -eu
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/envs/liquid-depth/bin/python
OUT=/root/autodl-tmp/liquid-depth-artifacts/training/liquid-adaptation-v15-20260915
NEW=/root/autodl-tmp/liquid-depth-data/synthetic/liquid-adaptation-v15-fresh-20260915
for arm in clean robust pretrained plane; do
 "$PY" scripts/train_liquid_adaptation_v15.py --arm "$arm" > "$OUT/$arm.log" 2>&1
 echo "ARM_FROZEN $arm"
done
/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender --background --python scripts/generate_candidate_holdout_v13.py -- --output-root "$NEW" --seed 20260915201 --repeats 4 > "$OUT/generation.log" 2>&1
for arm in clean robust pretrained plane; do
 "$PY" scripts/train_liquid_adaptation_v15.py --arm "$arm" --stage test --manifest "$NEW/manifest.csv" > "$OUT/$arm-test.log" 2>&1
 echo "ARM_TESTED $arm"
done
