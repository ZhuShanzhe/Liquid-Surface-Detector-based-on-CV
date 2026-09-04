#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/envs/liquid-depth/bin/python}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v3-calibration-top-12000/manifest.csv}"
V4_MANIFEST="${V4_MANIFEST:-/root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v3-calibration-top-12000/manifest_multilayer_v4_refined.csv}"
V4_LABEL_ROOT="${V4_LABEL_ROOT:-/root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v4-multilayer-labels}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v8-ordinary-sensor-anchor/best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v9-1-transparent-multilayer}"

cd "${PROJECT_ROOT}"
if [[ ! -f "${V4_MANIFEST}" ]]; then
  "${PYTHON}" scripts/upgrade_multilayer_labels.py \
    --manifest "${SOURCE_MANIFEST}" \
    --output-manifest "${V4_MANIFEST}" \
    --output-root "${V4_LABEL_ROOT}"
fi

"${PYTHON}" scripts/train_transparent_multilayer.py \
  --manifest "${V4_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --initialize-from "${INITIAL_CHECKPOINT}" \
  --epochs "${EPOCHS:-10}" \
  --head-only-epochs "${HEAD_ONLY_EPOCHS:-4}" \
  --batch-size "${BATCH_SIZE:-12}" \
  --workers "${WORKERS:-10}" \
  --image-size "${IMAGE_SIZE:-320,180}" \
  --backbone-learning-rate-scale "${BACKBONE_LEARNING_RATE_SCALE:-0.05}"

"${PYTHON}" scripts/evaluate_transparent_multilayer.py \
  --manifest "${V4_MANIFEST}" \
  --checkpoint "${OUTPUT_DIR}/best.pth" \
  --output "${OUTPUT_DIR}/confidence_calibration.json"
