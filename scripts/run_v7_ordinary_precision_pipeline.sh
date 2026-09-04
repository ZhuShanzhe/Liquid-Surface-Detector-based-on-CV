#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/liquid-depth/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/liquid-depth-data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/liquid-depth-artifacts}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/synthetic/liquid-sim-v3-calibration-top-12000/manifest.csv}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${ARTIFACT_ROOT}/training/universal-liquid-v6-v3-balanced-full/best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/training/universal-liquid-v7-ordinary-precision}"
EVALUATION_DIR="${EVALUATION_DIR:-${ARTIFACT_ROOT}/evaluation}"
MODEL_DIR="${MODEL_DIR:-${ARTIFACT_ROOT}/models}"
EPOCHS="${EPOCHS:-8}"

mkdir -p "${OUTPUT_DIR}" "${EVALUATION_DIR}" "${MODEL_DIR}"

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/train_universal_multitask.py \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size 24 \
  --workers 12 \
  --base-channels 24 \
  --image-size 320,180 \
  --min-depth-m 0.1 \
  --max-depth-m 10.0 \
  --learning-rate 0.00002 \
  --rgb-prior \
  --level-calibration-head \
  --calibration-scale-limit 0.03 \
  --calibration-bias-limit-m 0.01 \
  --initialize-from "${INITIAL_CHECKPOINT}" \
  --relative-weight 1.0 \
  --tolerance-weight 0.75 \
  --surface-level-weight 0.5 \
  --surface-absolute-weight 0.5 \
  --surface-tolerance-weight 1.0 \
  --surface-quantile-weight 0.25 \
  --surface-quantile 0.90 \
  --ordinary-loss-boost 2.0 \
  --calibration-regularization-weight 0.01 \
  --confidence-relative-tolerance 0.02 \
  --confidence-absolute-floor-m 0.005 \
  --sampling-mode scenario_severity \
  --difficulty-boosts ordinary=3,depth_failure=1,compound=0.75,multilayer=0.75,low_light=1,glare=1,transparent=1,translucent=1 \
  --range-balance-strength 1.0 \
  --range-balance-max-factor 4.0 \
  --selection-ordinary-weight 0.70 \
  --log-every 100

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/evaluate_universal_checkpoint.py \
  --checkpoint "${OUTPUT_DIR}/best.pth" \
  --manifest "${MANIFEST}" \
  --split test \
  --batch-size 16 \
  --workers 8 \
  --output "${EVALUATION_DIR}/universal-v7-ordinary-precision-test.json"

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/calibrate_scenario_confidence.py \
  --checkpoint "${OUTPUT_DIR}/best.pth" \
  --manifest "${MANIFEST}" \
  --fit-split val \
  --eval-split test \
  --batch-size 16 \
  --workers 8 \
  --relative-tolerance 0.02 \
  --absolute-floor-m 0.005 \
  --minimum-coverage 0.30 \
  --minimum-evaluable-rate 0.90 \
  --maximum-abs-rel 0.03 \
  --minimum-within-rate 0.50 \
  --output "${EVALUATION_DIR}/scenario-confidence-v7-ordinary-candidate.json"

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/calibrate_scenario_confidence.py \
  --checkpoint "${OUTPUT_DIR}/best.pth" \
  --manifest "${MANIFEST}" \
  --fit-split val \
  --eval-split test \
  --batch-size 16 \
  --workers 8 \
  --relative-tolerance 0.02 \
  --absolute-floor-m 0.005 \
  --minimum-coverage 0.95 \
  --minimum-evaluable-rate 0.99 \
  --maximum-abs-rel 0.015 \
  --minimum-within-rate 0.90 \
  --output "${EVALUATION_DIR}/scenario-confidence-v7-ordinary-engineering.json"

"${PYTHON_BIN}" scripts/export_multitask.py \
  --checkpoint "${OUTPUT_DIR}/best.pth" \
  --output "${MODEL_DIR}/universal-liquid-v7-ordinary-precision.ts" \
  --device cpu

echo "V7 ordinary precision pipeline completed."
echo "Checkpoint: ${OUTPUT_DIR}/best.pth"
echo "Engineering gate: ${EVALUATION_DIR}/scenario-confidence-v7-ordinary-engineering.json"
