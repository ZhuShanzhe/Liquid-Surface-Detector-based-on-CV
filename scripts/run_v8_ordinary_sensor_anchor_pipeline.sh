#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/liquid-depth/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/liquid-depth-data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/liquid-depth-artifacts}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/synthetic/liquid-sim-v3-calibration-top-12000/manifest.csv}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${ARTIFACT_ROOT}/training/universal-liquid-v6-v3-balanced-full/best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACT_ROOT}/training/universal-liquid-v8-ordinary-sensor-anchor}"
EVALUATION_DIR="${EVALUATION_DIR:-${ARTIFACT_ROOT}/evaluation}"
MODEL_DIR="${MODEL_DIR:-${ARTIFACT_ROOT}/models}"
CHECKPOINT="${OUTPUT_DIR}/best.pth"

mkdir -p "${OUTPUT_DIR}" "${EVALUATION_DIR}" "${MODEL_DIR}"

"${PYTHON_BIN}" scripts/enable_robust_anchor_checkpoint.py \
  --source "${SOURCE_CHECKPOINT}" \
  --output "${CHECKPOINT}" \
  --mask-threshold 0.5 \
  --bias-limit-m 0.25

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/evaluate_universal_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --split test \
  --batch-size 16 \
  --workers 8 \
  --output "${EVALUATION_DIR}/universal-v8-ordinary-sensor-anchor-test.json"

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/calibrate_scenario_confidence.py \
  --checkpoint "${CHECKPOINT}" \
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
  --output "${EVALUATION_DIR}/scenario-confidence-v8-ordinary-engineering.json"

"${PYTHON_BIN}" scripts/export_multitask.py \
  --checkpoint "${CHECKPOINT}" \
  --output "${MODEL_DIR}/universal-liquid-v8-ordinary-sensor-anchor.ts" \
  --device cpu

echo "V8 ordinary sensor-anchor pipeline completed."
