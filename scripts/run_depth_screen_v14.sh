set -eu
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
PY=/root/autodl-tmp/envs/liquid-depth/bin/python
OUT=/root/autodl-tmp/liquid-depth-artifacts/training/multimethod-v14-20260915
OLD=/root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v3-calibration-top-12000/manifest_multilayer_v4_refined.csv
NEW=/root/autodl-tmp/liquid-depth-data/synthetic/multimethod-v14-fresh-20260915
for method in raw direct v12 dfnet_native dfnet_wide swin_native swin_wide lingbot plane_direct plane_v12; do
 [ -f "$OUT/dev-$method.json" ] && continue
 "$PY" scripts/screen_depth_methods_v14.py --method "$method" --manifest "$OLD" --split val --fold 0 --out "$OUT/dev-$method.json" > "$OUT/dev-$method.log" 2>&1
 echo "DEV_DONE $method"
done
/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender --background --python scripts/generate_candidate_holdout_v13.py -- --output-root "$NEW" --seed 20260915107 --repeats 4 > "$OUT/generation.log" 2>&1
for method in raw direct v12 dfnet_native dfnet_wide swin_native swin_wide lingbot plane_direct plane_v12; do
 "$PY" scripts/screen_depth_methods_v14.py --method "$method" --manifest "$NEW/manifest.csv" --out "$OUT/fresh-$method.json" > "$OUT/fresh-$method.log" 2>&1
 echo "FRESH_DONE $method"
done
for method in direct v12 lingbot plane_direct dfnet_wide plane_v12; do
 "$PY" scripts/screen_depth_methods_v14.py --method "$method" --manifest "$NEW/manifest.csv" --stress --out "$OUT/stress-$method.json" > "$OUT/stress-$method.log" 2>&1
 echo "STRESS_DONE $method"
done
