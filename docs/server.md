# Server layout

```text
/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/  # Git checkout
/root/autodl-tmp/envs/liquid-depth/                    # Conda environment
/root/autodl-tmp/liquid-depth-data/                    # raw RGB-D data
/root/autodl-tmp/liquid-depth-artifacts/               # generated results
```

Activate the environment with:

```bash
conda activate /root/autodl-tmp/envs/liquid-depth
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
```

The RTX 5090 is a Blackwell GPU. Use an official PyTorch wheel built with CUDA 12.8 or newer; older CUDA wheels do
not contain `sm_120` support.

