#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${LIQUID_DEPTH_ENV:-/root/autodl-tmp/envs/liquid-depth}"
PROJECT_DIR="${LIQUID_DEPTH_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CACHE_DIR="${LIQUID_DEPTH_CACHE:-/root/autodl-tmp/cache}"

mkdir -p "$(dirname "${ENV_PREFIX}")" "${CACHE_DIR}/pip" "${CACHE_DIR}/torch"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export TORCH_HOME="${CACHE_DIR}/torch"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_PREFIX}" python=3.11 pip
fi

PYTHON="${ENV_PREFIX}/bin/python"
"${PYTHON}" -m pip install --upgrade pip setuptools wheel

# CUDA 12.8+ is required for the RTX 5090 (sm_120).
"${PYTHON}" -m pip install +  torch==2.11.0 torchvision==0.26.0 +  --index-url https://download.pytorch.org/whl/cu128
"${PYTHON}" -m pip install -e "${PROJECT_DIR}[train,dev]"

"${PYTHON}" - <<'PY'
import cv2
import torch
import liquid_depth

print("liquid_depth", liquid_depth.__version__)
print("opencv", cv2.__version__)
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch")
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
tensor = torch.randn(1024, 1024, device="cuda")
print("cuda_smoke_sum", float(tensor.square().sum()))
PY

