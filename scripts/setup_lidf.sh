#!/usr/bin/env bash
set -euo pipefail

research_root=${1:-/root/autodl-tmp/liquid-depth-data/research}
env_root=${2:-/root/autodl-tmp/envs/liquid-depth}
source_root="$research_root/sources/LIDF"
cuda_root=${CUDA_HOME:-/usr/local/cuda-12.8}
python_bin="$env_root/bin/python"

if [[ ! -f "$source_root/src/extensions/ray_aabb/ray_aabb_cuda_kernel.cu" ]]; then
  echo "LIDF source not found at $source_root" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python environment not found at $env_root" >&2
  exit 2
fi
if [[ ! -x "$cuda_root/bin/nvcc" ]]; then
  echo "CUDA compiler not found at $cuda_root/bin/nvcc" >&2
  exit 2
fi

export CUDA_HOME="$cuda_root"
export PATH="$env_root/bin:$cuda_root/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export MAX_JOBS=${MAX_JOBS:-4}
export PYTORCH_EXTENSIONS_DIR=${PYTORCH_EXTENSIONS_DIR:-$research_root/lidf/torch_extensions}
mkdir -p "$PYTORCH_EXTENSIONS_DIR"

"$python_bin" -m pip install \
  ninja easydict plyfile matplotlib scikit-learn scikit-image Shapely six imageio
"$python_bin" -m pip install --no-deps 'imgaug==0.4.0'
if ! "$python_bin" -c 'import torch_scatter' >/dev/null 2>&1; then
  "$python_bin" -m pip install --no-build-isolation 'torch-scatter==2.1.2'
fi

# imgaug 0.4.0 reads np.sctypes, which NumPy 2 removed. Patch the vendored
# research checkout instead of downgrading NumPy or the project OpenCV build.
"$python_bin" - "$source_root/src/utils/data_augmentation.py" <<'PYPATCH'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
needle = "import numpy as np\n"
compat = '''import numpy as np

# imgaug 0.4.0 still reads np.sctypes, removed by NumPy 2.0.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "float": [np.float16, np.float32, np.float64],
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
    }
'''
if compat not in text:
    if needle not in text:
        raise SystemExit(f"Cannot locate NumPy import in {path}")
    path.write_text(text.replace(needle, compat, 1))
PYPATCH

(
  cd "$source_root/src"
  "$python_bin" - <<'PYSMOKE'
import torch
from extensions.pcl_aabb.jit import pcl_aabb
from extensions.ray_aabb.jit import ray_aabb
import models.pipeline
import datasets.cleargrasp_dataset

device = "cuda"
points = torch.tensor([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]], device=device)
bounds = torch.tensor([[-1.0, -1.0, 0.0, 1.0, 1.0, 2.0]], device=device)
point_batch = torch.zeros(2, dtype=torch.int32, device=device)
voxel_batch = torch.zeros(1, dtype=torch.int32, device=device)
inside = pcl_aabb.forward(points, bounds, point_batch, voxel_batch)
rays = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], device=device)
ray_mask, _ = ray_aabb.forward(rays, bounds, point_batch, voxel_batch)
if inside.cpu().tolist() != [[1, 0]] or ray_mask.cpu().tolist() != [[1, 1]]:
    raise SystemExit("LIDF CUDA extension smoke test returned unexpected values")
print("LIDF dependencies, imports, and CUDA extensions are ready")
PYSMOKE
)
