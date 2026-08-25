# System readiness checklist

## Complete reusable layers

- Camera: pinned Orbbec ROS 2 driver, synchronized RGB/depth/intrinsics capture, stable frame contract.
- Runtime: server bootstrap, CUDA PyTorch environment, configuration loader, single-frame and batch CLI.
- Baseline: classical/neural segmentation interface, optional metric depth-refinement interface, robust plane fit,
  bottom calibration, physical plane-angle gate, confidence diagnostics, and machine-readable output.
- Training: reviewed-label manifest loader, DeepLabV3 baseline, best-checkpoint saving, TorchScript export.
- Data: byte-level inventory, frame-contract validation, license-aware research registry/downloader.
- Evaluation: scenario-level MAE/RMSE/bias/P95, prediction coverage, acceptance coverage, and accepted-only error.
- Quality: deterministic unit tests, environment/GPU/disk audit, explicit rejection instead of forced output.

## Items that necessarily remain project-data dependent

- Multi-point physical scale calibration for each camera/container geometry.
- Human-reviewed liquid masks and metric liquid heights for difficult scenes.
- A leakage-free split manifest with container/session/scenario metadata.
- Trained depth-restoration weights and confidence calibration.
- Runtime thresholds chosen from validation data, not from the held-out test set.

The reusable software foundation is complete when all tests, the system audit, frame validation, legacy regression, and
one export/load smoke test pass on the server. Algorithmic research can then change models behind the existing
segmentation and depth-refinement contracts without altering acquisition or downstream output.
