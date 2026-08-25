#!/usr/bin/env bash
set -euo pipefail

install_system_deps=false
if [[ ${1:-} == "--install-system-deps" ]]; then
  install_system_deps=true
  shift
fi
research_root=${1:-/root/autodl-tmp/liquid-depth-data/research}
source_root="$research_root/sources/ClearGrasp"
gaps_root="$source_root/api/depth2depth/gaps"
makefile="$gaps_root/apps/depth2depth/Makefile"

if [[ ! -f "$makefile" ]]; then
  echo "ClearGrasp source not found at $source_root" >&2
  exit 2
fi
if $install_system_deps; then
  if [[ $(id -u) -ne 0 ]]; then
    echo "--install-system-deps requires root" >&2
    exit 2
  fi
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1-mesa-dev libglu1-mesa-dev libglfw3-dev libhdf5-dev \
    libsuitesparse-dev libjpeg-dev libpng-dev
fi

# The upstream Ubuntu example passes the HDF5 directory as a bare compiler token.
# Make the include path explicit; this edit is idempotent.
sed -i 's|USER_CFLAGS=-DRN_USE_CSPARSE "/usr/include/hdf5/serial/"|USER_CFLAGS=-DRN_USE_CSPARSE -I/usr/include/hdf5/serial/|' "$makefile"
make -C "$gaps_root" -j"$(nproc)"
executable="$gaps_root/bin/x86_64/depth2depth"
if [[ ! -x "$executable" ]]; then
  echo "ClearGrasp depth2depth build did not produce $executable" >&2
  exit 1
fi

(
  cd "$gaps_root"
  bash depth2depth.sh
)
python - "$gaps_root" <<'PY'
from pathlib import Path
import sys
import cv2
import numpy as np
root = Path(sys.argv[1]) / "sample_files"
actual = cv2.imread(str(root / "output-depth.png"), cv2.IMREAD_UNCHANGED)
expected = cv2.imread(str(root / "expected-output-depth.png"), cv2.IMREAD_UNCHANGED)
if actual is None or expected is None or actual.shape != expected.shape:
    raise SystemExit("ClearGrasp smoke-test output is missing or has an invalid shape")
delta = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
print(f"ClearGrasp depth2depth ready: max_abs={delta.max()}, mean_abs={delta.mean():.6f}")
if int(delta.max()) > 8:
    raise SystemExit("ClearGrasp smoke test deviates materially from the released reference")
PY
