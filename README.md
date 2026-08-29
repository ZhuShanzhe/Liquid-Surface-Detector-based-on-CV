# Liquid Surface Detector based on RGB-D Computer Vision

An end-to-end system for synchronized Orbbec RGB-D capture, liquid-surface perception, transparent/specular depth
restoration, calibrated 3-D geometry, and quality-gated liquid-depth output. Camera acquisition, GPU research, raw data,
and generated artifacts have stable boundaries so models can change without rewriting the complete system.

## System status

- ROS 2 Humble acquisition node for synchronized RGB, depth, and camera intrinsics.
- Pinned upstream Orbbec Gemini 2 ROS driver as a Git submodule.
- Stable frame-directory contract independent of ROS.
- Classical and TorchScript segmentation backends.
- Identity and TorchScript RGB-D depth-refinement backends with per-pixel confidence.
- RANSAC plus robust-refinement plane fitting and bottom/liquid physical consistency checks.
- One-command single-frame or batch inference with explicit accepted/rejected output.
- DeepLabV3 training baseline and TorchScript export.
- Five-channel RGB-D multi-task network for mask, metric depth, normal, uncertainty, and confidence.
- Difficulty-balanced training, meniscus-aware geometry, and confidence-aware robust Kalman filtering.
- Scenario-aware end-to-end evaluation, frame validation, system audit, and research-data registry.
- Server bootstrap for an RTX 5090 using CUDA 12.8 PyTorch.

The current classical path is a validated regression baseline. Transparent liquid and strong reflection require learned
RGB-D restoration and uncertainty rather than RANSAC tuning alone. See
[the research plan](docs/robustness_research.md) and [readiness checklist](docs/system_readiness.md).

## Repository layout

```text
configs/                   runtime and research-dataset configuration
src/liquid_depth/          reusable inference, refinement, geometry, and evaluation package
src/liquid_depth/training/ deep-learning dataset and segmentation baseline
ros2_ws/src/               camera acquisition ROS 2 package
third_party/               pinned OrbbecSDK_ROS2 submodule
scripts/                   server, camera, data, audit, export, evaluation, and training entry points
tests/                     deterministic unit tests
docs/                      architecture, deployment, camera, research, and training notes
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
python scripts/audit_system.py --require-cuda --data-root /root/autodl-tmp/liquid-depth-data
```

Configured deployment locations:

- project: `/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV`
- environment: `/root/autodl-tmp/envs/liquid-depth`
- raw/research data: `/root/autodl-tmp/liquid-depth-data`
- outputs/models: `/root/autodl-tmp/liquid-depth-artifacts`

## Product control panel

The product layer supports two deliberately separate calibration modes:

- fixed or occasionally moved camera: operator-selected measurement rail, 3 points minimum and 5 recommended;
- continuously moving camera: metric container CAD/point cloud plus per-frame marker pose are required.

Start the local-only panel with:

```bash
liquid-depth-panel \
  --capture-dir /root/autodl-tmp/liquid-depth-data/live \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/live
```

Use SSH port forwarding to open the panel from an operator computer. The model/runtime modules are independent of the
HTML panel so the perception network can be replaced without rebuilding the operator workflow. See the
[Chinese product deployment and calibration guide](docs/product_deployment_zh.md).

## Offline end-to-end inference

Fit the empty-container bottom plane once for a fixed camera/container setup:

```bash
liquid-depth --config configs/pipeline.yaml fit-bottom \
  --frame /root/autodl-tmp/liquid-depth-data/legacy_rgbd/20260517_144437 \
  --output /root/autodl-tmp/liquid-depth-artifacts/calibration/bottom_plane.json
```

Estimate one liquid depth:

```bash
liquid-depth --config configs/pipeline.yaml infer \
  --frame /root/autodl-tmp/liquid-depth-data/legacy_rgbd/20260517_143323 \
  --bottom-plane /root/autodl-tmp/liquid-depth-artifacts/calibration/bottom_plane.json \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/inference/20260517_143323
```

`depth_result.json` contains the numeric result, model backends, confidence/validity, plane diagnostics, and quality-gate
decision. A result with `accepted: false` must not be sent to downstream control logic.

The default scale is geometric centimeters per meter. Determine `output.calibration_scale_per_meter` from a multi-point
physical calibration before reporting final accuracy.

## Research data

Inspect the registry before downloading. It records task match, approximate size, source, and license:

```bash
python scripts/download_research_data.py --list
```

Only code, configuration, provenance, loaders, and metrics belong in Git. Raw datasets, weights, environments, point
clouds, and generated artifacts stay on the server data disk.

## Documentation

- [System architecture](docs/architecture.md)
- [Executable algorithm roadmap](docs/algorithm_roadmap.md)
- [Baseline reproduction status](docs/baseline_results.md)
- [Transparent/specular robustness and SeeGroup plan](docs/robustness_research.md)
- [Complex-scene implementation and measured latency](docs/complex_scene_optimization.md)
- [Scene-adaptive routing and 500 ms deployment policy](docs/scene_adaptive_runtime.md)
- [Complex-scene evaluation protocol](docs/complex_scene_evaluation.md)
- [Top-view v4 simulation and benchmark results](docs/simulation_results_v2_top.md)
- [Site few-shot calibration and top-camera deployment](docs/site_few_shot_calibration.md)
- [Market-camera system-error calibration simulation](docs/site_calibration_simulation_results.md)
- [Five-distance physical-camera qualification](docs/camera_plane_qualification.md)
- [No-USB virtual RGB-D deployment test](docs/no_usb_deployment_test.md)
- [Industrial accuracy profile](configs/accuracy_profile_industrial_v1.yaml)
- [Software copyright and V1.0 productization plan](docs/software_copyright_plan.md)
- [System readiness](docs/system_readiness.md)
- [Camera acquisition](docs/camera.md)
- [Training](docs/training.md)
- [Server operations](docs/server.md)

Credentials and access tokens are never stored in this repository.
