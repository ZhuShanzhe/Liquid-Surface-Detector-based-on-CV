# Baseline reproduction status

Updated 2026-08-26 on the RTX 5090 server. “Runnable” means the official revision and released checkpoints load strictly under the modern compatibility adapter and complete a real RGB-D frame.

## Reproduction status

| Baseline | Official revision | Checkpoint | Runtime status | Legacy 10-frame MAE / RMSE | Provisional latency |
| --- | --- | --- | --- | --- | --- |
| Classical raw RGB-D | project regression | n/a | frozen control | 0.831 / 0.967 cm | not reprofiled |
| TransCG DFNet | 135f9e0 | corrected 2022-10-14, MD5 `6e6e00f7cc02c644a34b1ce6ee4f364fa` | runnable, official preprocessing | 0.535 / 0.669 cm | about 0.76 s including CPU nearest-neighbor fill |
| DREDS SwinDRNet | 1b0ac30 | official `model.pth`, strict 650-state load | runnable, official validation resize | 0.753 / 0.820 cm | about 0.48 s |
| ClearGrasp | 0688647 | official mask, outline, and normal checkpoints | runnable; three strict DRN loads plus released depth2depth optimizer | pending on project captures | 774.4 ms on ClearGrasp real-test |
| RGB-D LIDF (optional) | 4dc85bb | official Drive link currently unavailable | source imports and both CUDA extensions pass on PyTorch 2.11 / CUDA 12.8 / sm_120 | pending | pending |

ClearGrasp system dependencies and its upstream HDF5 include-path compatibility fix are automated by `scripts/setup_cleargrasp.sh`. The modern adapter loads all three official neural checkpoints directly and does not require the legacy Python-2 inference stack.

LIDF modern-environment setup is automated by `scripts/setup_lidf.sh`. It builds `torch-scatter` and the released point/AABB and ray/AABB CUDA extensions for the current GPU, applies the NumPy 2 compatibility shim to the research checkout, and runs numerical CUDA smoke tests. Full LIDF inference would require the released checkpoint archive, but it is no longer a baseline-selection blocker.

DFNet is the current leader on the 10 ordinary project frames. This sequence is only a regression gate and must not decide the research baseline by itself.

## Common DREDS STD real-data gate

The first common evaluation uses one frame from each of 12 real test sequences. Pixels whose source instance mask is `255` are excluded as background. Values below are evaluated over 334,483 object pixels.

| Method | Coverage | MAE | RMSE | Boundary RMSE | Median warm latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw RGB-D identity | 96.92% | 14.75 mm | 26.88 mm | 33.42 mm | 10.6 ms |
| TransCG DFNet | 100% | 14.84 mm | 22.95 mm | **28.22 mm** | 90.9 ms |
| DREDS SwinDRNet | 100% | **9.72 mm** | **19.47 mm** | 37.67 mm | **37.9 ms** |

SwinDRNet is the strongest overall DREDS-domain restoration baseline, while DFNet is better at this sample's object boundary.

## Full ClearGrasp real-test gate

The reproducible manifest contains all 113 D415/D435 frames from the released `real-test` split. Opaque captures are used as metric-depth targets.

| Backend | Coverage | MAE | RMSE | Boundary MAE | Median warm latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw RGB-D | 43.48% | 24.44 mm | 34.72 mm | 9.42 mm* | 4.2 ms |
| TransCG DFNet | 100% | 26.38 mm | 37.54 mm | 22.51 mm | 73.2 ms |
| DREDS SwinDRNet | 100% | **14.61 mm** | **21.52 mm** | 37.48 mm | **29.0 ms** |
| ClearGrasp | 100% | 24.82 mm | 51.60 mm | 65.15 mm | 774.4 ms |

`*` Raw boundary error is measured only where the sensor returned depth (43.48% coverage), so it is not directly comparable with completed-depth boundary errors.

This second gate confirms SwinDRNet as the strongest deployment baseline for the current RGB-D pipeline. ClearGrasp remains useful as a normal/boundary teacher and physics-based reference rather than the primary real-time backend. The next comparison gate will add an LIDF/RGB-D LIF-style method and explicit transparent/translucent/glare/saturated-highlight/container-edge partitions.

The current DREDS manifest labels each complete object frame with all three broad material tags, so its preliminary per-tag rows duplicate the overall metric. Material-specific masks must be built from DREDS metadata before any per-material claim is made.

Common outputs and evaluation artifacts stay under:

    /root/autodl-tmp/liquid-depth-artifacts/evaluation

## Project multi-task object pretraining

The 12-epoch object-domain pretraining run completed on 2026-08-26. Epoch 11 is
the selected checkpoint with validation depth RMSE 50.33 mm, MAE 28.85 mm, and
mask IoU 0.7258. Epoch 12 reached mask IoU 0.7258 but slightly worse depth RMSE.

| Public gate | Coverage | MAE | RMSE | Boundary RMSE | Median warm latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| DREDS STD 12-sequence sample | 100% | 64.99 mm | 103.44 mm | 120.84 mm | 23.5 ms |
| ClearGrasp real-test (113 frames) | 100% | 27.40 mm | 41.41 mm | 66.12 mm | 10.2 ms |

This checkpoint is useful for project mask/normal/feature initialization, but it
does not replace SwinDRNet and does not satisfy the 10 mm end-to-end liquid-depth
target. Confidence is derived from log variance using a bounded sigmoid and must
be calibrated on DTLD before it can drive selective rejection.

## Runtime backend selection

Runtime adapters are selected through `depth_refinement.backend`:

    identity
    transcg_dfnet
    dreds_swindrnet
    cleargrasp
    torchscript

Batch inference constructs each model once and reuses it across frames. DFNet and ClearGrasp expose a neutral confidence prior of 0.5 for valid output, while SwinDRNet exposes its raw-versus-restored blending confidence. These values are not calibrated selective-risk confidence until evaluated on difficult held-out sequences.
