set -eu
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=4
PY=/root/autodl-tmp/envs/liquid-depth/bin/python
OUT=/root/autodl-tmp/liquid-depth-artifacts/training/liquid-adaptation-v15-20260915
NEW=/root/autodl-tmp/liquid-depth-data/synthetic/liquid-adaptation-v15-fresh-20260915/manifest.csv
ready=0
for i in $(seq 1 180); do
 if "$PY" -c "import json;from pathlib import Path;p=Path('$OUT/strong/robust/frozen.json');assert p.exists() and json.loads(p.read_text()).get('variant')=='robust_strong'" >/dev/null 2>&1; then ready=1;break;fi
 sleep 5
done
[ "$ready" = 1 ]
"$PY" scripts/train_quality_context_v15.py --kind strong > "$OUT/context-strong.log" 2>&1
echo CONTEXT_STRONG_FROZEN
ready=0
for i in $(seq 1 180); do
 if [ -f "$NEW" ] && [ -f "$OUT/context-plane/plane/frozen.json" ]; then ready=1;break;fi
 sleep 5
done
[ "$ready" = 1 ]
"$PY" scripts/train_liquid_adaptation_strong_v15.py --arm robust --stage test --manifest "$NEW" > "$OUT/strong-test.log" 2>&1
echo STRONG_TESTED
"$PY" scripts/train_quality_context_v15.py --kind strong --stage test --manifest "$NEW" > "$OUT/context-strong-test.log" 2>&1
echo CONTEXT_STRONG_TESTED
"$PY" scripts/train_quality_context_v15.py --kind plane --stage test --manifest "$NEW" > "$OUT/context-plane-test.log" 2>&1
echo CONTEXT_PLANE_TESTED
"$PY" scripts/evaluate_v12_reference_v15.py --manifest "$NEW" > "$OUT/v12-reference.log" 2>&1
echo V12_REFERENCE_TESTED
