# System readiness checklist

## Product acceptance target adopted on 2026-09-04

- The current AbsRel <= 3%, max(5 mm, 2%) pass rate >= 50%, output coverage >= 30%, and evaluable-output rate >= 90% profile is a simulation-candidate gate only.
- Standard supported scenes require liquid-level AbsRel <= 1.5%, tolerance pass rate >= 90%, output coverage >= 95%, and evaluable-output rate >= 99%.
- Difficult supported scenes require liquid-level AbsRel <= 3%, tolerance pass rate >= 75%, output coverage >= 80%, and evaluable-output rate >= 98%.
- Unsupported or extreme scenes may reject, but the false-accept ratio must be <= 1% and every rejection must have a machine-readable reason.
- Product reports must also include P95 and maximum error, signed bias, temporal jitter, consecutive-bad-frame count, and recovery time after a scene change.
- V6 passes the simulation-candidate gate but does not pass the engineering-product gate overall. On the current independent synthetic test, only uneven-surface and partial-depth-failure routes meet every difficult-scene threshold.

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
