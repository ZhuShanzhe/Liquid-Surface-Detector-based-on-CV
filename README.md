# Liquid Surface Detector based on RGB-D Computer Vision

An end-to-end system for synchronized Orbbec RGB-D capture, liquid-surface segmentation, robust 3-D plane fitting,
and liquid-depth output. The repository separates camera acquisition, algorithm code, large datasets, generated
artifacts, and future deep-learning training so work can move between an edge camera computer and a GPU server.

## What is ready

- ROS 2 Humble acquisition node for synchronized RGB, depth, and camera intrinsics.
- Pinned upstream Orbbec Gemini 2 ROS driver as a Git submodule.
- Stable frame-directory contract independent of ROS.
- Classical segmentation baseline compatible with the May 2026 prototype.
- RANSAC + robust-refinement 3-D liquid and bottom plane fitting.
- One-command depth inference with masks, plane metrics, confidence, and a quality gate.
- Swappable TorchScript segmentation backend and DeepLabV3 training baseline.
- Server bootstrap for an RTX 5090 using a CUDA 12.8 PyTorch build.

## Repository layout

```text
configs/                   pipeline and backend configuration
src/liquid_depth/          reusable inference and geometry package
src/liquid_depth/training/ deep-learning dataset and training baseline
ros2_ws/src/               camera acquisition ROS 2 package
third_party/               pinned OrbbecSDK_ROS2 submodule
scripts/                   server, camera, data, and training entry points
tests/                     deterministic unit tests
docs/                      architecture, deployment, camera, and training notes
data/                      documentation only; large captures live outside Git
artifacts/                 documentation only; generated outputs live outside Git
```

## Server quick start

```bash
git clone --recurse-submodules git@github.com:ZhuShanzhe/Liquid-Surface-Detector-based-on-CV.git
cd Liquid-Surface-Detector-based-on-CV
bash scripts/bootstrap_server.sh
conda activate /root/autodl-tmp/envs/liquid-depth
pytest -q
```

The configured deployment uses:

- project: `/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV`
- environment: `/root/autodl-tmp/envs/liquid-depth`
- raw data: `/root/autodl-tmp/liquid-depth-data`
- outputs/models: `/root/autodl-tmp/liquid-depth-artifacts`

## Offline end-to-end inference

First fit the empty-container bottom plane:

```bash
liquid-depth --config configs/pipeline.yaml fit-bottom \
  --frame /root/autodl-tmp/liquid-depth-data/legacy_rgbd/20260517_144437 \
  --output /root/autodl-tmp/liquid-depth-artifacts/calibration/bottom_plane.json
```

Then estimate one liquid depth:

```bash
liquid-depth --config configs/pipeline.yaml infer \
  --frame /root/autodl-tmp/liquid-depth-data/legacy_rgbd/20260517_143323 \
  --bottom-plane /root/autodl-tmp/liquid-depth-artifacts/calibration/bottom_plane.json \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/inference/20260517_143323
```

The primary machine-readable output is `depth_result.json`. A result with `accepted: false` should be treated as
uncertain and not sent to downstream control logic.

The default scale is geometric centimeters per meter. For calibrated physical depth, update
`output.calibration_scale_per_meter` from a reference measurement or multi-point calibration.

## Camera acquisition

See [docs/camera.md](docs/camera.md). The camera-facing machine and GPU server are deliberately decoupled: a rented
cloud server cannot access a USB camera attached elsewhere. They exchange the stable RGB-D frame format.

## Deep-learning work

See [docs/training.md](docs/training.md). The next milestone is a reviewed, leakage-free labeled dataset spanning
complex conditions. Do not optimize only on the 2026 prototype sequence: that sequence is useful for regression tests
but is too small and homogeneous for robustness claims.

## Security and data policy

Credentials, raw data, generated point clouds, environments, model weights, and build outputs are ignored by Git.
Never commit server passwords, private keys, or access tokens.

