#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${LIQUID_PYTHON:-python}" -m liquid_depth.scene_console_web \
  --checkpoint "${PROJECT_ROOT}/assets/models/scene_router/scene_head.pt" \
  --work "${PROJECT_ROOT}/runs/scene-console" "$@"
