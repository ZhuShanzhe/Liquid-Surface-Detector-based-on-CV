#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/envs/liquid-depth/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/liquid-depth-data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/liquid-depth-artifacts}"
MANIFEST="${MANIFEST:-${DATA_ROOT}/synthetic/liquid-sim-v3-calibration-top-12000/manifest.csv}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-${ARTIFACT_ROOT}/training/universal-liquid-v5-calibration-aware/best.pth}"
CONFIDENCE_DIR="${CONFIDENCE_DIR:-${ARTIFACT_ROOT}/training/universal-liquid-v6-v3-confidence-full}"
JOINT_DIR="${JOINT_DIR:-${ARTIFACT_ROOT}/training/universal-liquid-v6-v3-balanced-full}"
EVALUATION_DIR="${EVALUATION_DIR:-${ARTIFACT_ROOT}/evaluation}"
MODEL_DIR="${MODEL_DIR:-${ARTIFACT_ROOT}/models}"

mkdir -p "${CONFIDENCE_DIR}" "${JOINT_DIR}" "${EVALUATION_DIR}" "${MODEL_DIR}"

"${PYTHON_BIN}" scripts/validate_synthetic_dataset.py \
  --manifest "${MANIFEST}" \
  --min-samples 12000 \
  --max-errors 0

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/evaluate_universal_checkpoint.py \
  --checkpoint "${INITIAL_CHECKPOINT}" \
  --manifest "${MANIFEST}" \
  --split test \
  --batch-size 16 \
  --workers 8 \
  --output "${EVALUATION_DIR}/universal-v5-v3-calibration-test.json"

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/train_universal_multitask.py \
  --manifest "${MANIFEST}" \
  --output-dir "${CONFIDENCE_DIR}" \
  --epochs 2 \
  --confidence-only-epochs 2 \
  --batch-size 24 \
  --workers 12 \
  --base-channels 24 \
  --image-size 320,180 \
  --min-depth-m 0.1 \
  --max-depth-m 10.0 \
  --learning-rate 0.0003 \
  --rgb-prior \
  --initialize-from "${INITIAL_CHECKPOINT}" \
  --confidence-relative-tolerance 0.02 \
  --confidence-absolute-floor-m 0.005 \
  --sampling-mode scenario_severity \
  --difficulty-boosts depth_failure=1.5,compound=1.5,multilayer=1.2,low_light=1.2,glare=1.2,transparent=1.2,translucent=1.2,large_depth_failure=1.5 \
  --log-every 100

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/train_universal_multitask.py \
  --manifest "${MANIFEST}" \
  --output-dir "${JOINT_DIR}" \
  --epochs 6 \
  --batch-size 24 \
  --workers 12 \
  --base-channels 24 \
  --image-size 320,180 \
  --min-depth-m 0.1 \
  --max-depth-m 10.0 \
  --learning-rate 0.00003 \
  --rgb-prior \
  --initialize-from "${CONFIDENCE_DIR}/best.pth" \
  --confidence-relative-tolerance 0.02 \
  --confidence-absolute-floor-m 0.005 \
  --surface-absolute-weight 0.5 \
  --sampling-mode scenario_severity \
  --difficulty-boosts depth_failure=1.5,compound=1.5,multilayer=1.2,low_light=1.2,glare=1.2,transparent=1.2,translucent=1.2,large_depth_failure=1.5 \
  --log-every 100

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/evaluate_universal_checkpoint.py \
  --checkpoint "${JOINT_DIR}/best.pth" \
  --manifest "${MANIFEST}" \
  --split test \
  --batch-size 16 \
  --workers 8 \
  --output "${EVALUATION_DIR}/universal-v6-v3-calibration-test.json"

OPENCV_LOG_LEVEL=ERROR "${PYTHON_BIN}" scripts/calibrate_scenario_confidence.py \
  --checkpoint "${JOINT_DIR}/best.pth" \
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
  --output "${EVALUATION_DIR}/scenario-confidence-v6-v3-calibration.json"

"${PYTHON_BIN}" scripts/export_multitask.py \
  --checkpoint "${JOINT_DIR}/best.pth" \
  --output "${MODEL_DIR}/universal-liquid-v6-v3-calibration.ts" \
  --device cpu

echo "V6 V3 calibration pipeline completed."
echo "Checkpoint: ${JOINT_DIR}/best.pth"
echo "Confidence report: ${EVALUATION_DIR}/scenario-confidence-v6-v3-calibration.json"
echo "TorchScript: ${MODEL_DIR}/universal-liquid-v6-v3-calibration.ts"
