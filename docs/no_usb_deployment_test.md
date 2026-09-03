# No-USB virtual RGB-D deployment test

## Scope

A cloud GPU server cannot validate a USB camera transport directly. The virtual
camera replays synthetic samples through the same frame contract used by the ROS
capture node:

- `rgb.png`: BGR-readable color image;
- `depth.npy`: 2-D `uint16` depth in millimetres, including zero-valued holes;
- `depth_info.json`: image size, intrinsics, distortion fields, and depth scale;
- `virtual_camera.json`: source truth and an explicit
  `hardware_validated: false` provenance marker.

This validates capture-file compatibility, preprocessing, TorchScript inference,
confidence rejection, accuracy reporting, and latency without connecting a USB
camera. It does not validate USB transport, the ROS 2 camera driver, hardware
RGB/depth timestamp synchronization, thermal drift, or real optical error.

## Run

Replay capture files only:

```bash
python scripts/run_virtual_rgbd_camera.py \
  --manifest /data/synthetic/manifest.csv \
  --split test --frames 32 \
  --output /artifacts/virtual-camera
```

Run the complete software-path qualification:

```bash
python scripts/test_no_usb_deployment.py \
  --manifest /data/synthetic/manifest.csv \
  --split test --frames 84 \
  --capture-output /artifacts/no-usb/captures \
  --checkpoint /artifacts/training/best.pth \
  --model /artifacts/models/universal-liquid.ts \
  --confidence-report /artifacts/evaluation/scenario-confidence.json \
  --report /artifacts/no-usb/report.json
```

Output directories must be new so test evidence cannot be silently overwritten.

## V5 result on the independent synthetic test split

The server run used 84 top-camera frames and the V5 calibration-aware
TorchScript model. The canonical report is outside Git at
`/root/autodl-tmp/liquid-depth-artifacts/evaluation/no-usb-v5-final/report.json`.

| Gate or metric | Result |
|---|---:|
| Capture-contract failures | 0 / 84 |
| Software capture-to-inference path | Pass |
| Inference p95 | 1.40 ms |
| Frames within 500 ms | 100% |
| Accepted coverage at threshold 0.90 | 70.24% |
| Accepted-frame surface-depth MAE | 2.46 cm |
| Accepted-frame surface-depth AbsRel | 2.36% |
| Within max(3 mm, 1%) tolerance | 35.59% |
| Product accuracy gate | Fail |
| Hardware validated | No |
| Deployment ready | No |

The latency is pure GPU model execution after warm-up; it excludes USB capture,
ROS transport, display rendering, and geometric liquid-height calculation.

A threshold sweep confirmed a precision/coverage trade-off rather than a complete
fix: threshold 0.80 produced 88.10% coverage and 6.22 cm MAE, while threshold
0.95 produced 58.33% coverage and 1.55 cm MAE. The current model therefore needs
confidence calibration and additional large-depth-failure training. Threshold
selection alone cannot meet both the target accuracy and useful coverage.

The reported raw-sensor comparison uses the ground-truth liquid mask and only
frames with enough valid raw depth. It is an oracle diagnostic, not a deployable
baseline. Final liquid height still requires verified camera depth correction,
bottom/reference geometry, and site calibration.

## V6 scenario-calibrated result

The V6 replay used the same 84-frame no-USB protocol, but loaded the held-out
scenario calibration report instead of a global confidence threshold. The report is:

`/root/autodl-tmp/liquid-depth-artifacts/evaluation/no-usb-v6-v3-policy-final/report.json`

| Gate or metric | Result |
|---|---:|
| Capture-contract failures | 0 / 84 |
| Software capture-to-inference path | Pass |
| Inference p95 | 1.45 ms |
| Frames within 500 ms | 100% |
| Scenario-qualified accepted coverage | 54.76% |
| Accepted-frame surface-depth MAE | 3.77 cm |
| Accepted-frame surface-depth AbsRel | 2.14% |
| Within max(5 mm, 2%) tolerance | 56.52% |
| Configured synthetic quality profile | Pass |
| Hardware validated | No |
| Deployment ready | No |

The replay explicitly rejected multilayer, compound, severe, and extreme
depth-failure routes because those routes were not qualified by the held-out
calibration split. Partial and large depth failure use separate thresholds.
The quality profile is configurable and is no longer hard-coded to a 1% gate.
A profile pass remains synthetic evidence only; `deployment_ready` stays false
until the target camera and site pass physical validation.
## Required physical-camera acceptance

Before an industrial release:

1. Run the five-distance diffuse-plane qualification for each camera.
2. Validate RGB/depth alignment and timestamps through the real ROS 2 driver.
3. Repeat the no-USB software metrics on recorded site sequences.
4. Verify end-to-end liquid height, rather than only surface distance, against
   traceable physical measurements.
5. Keep `deployment_ready=false` until both the model accuracy gate and hardware
   qualification pass.
