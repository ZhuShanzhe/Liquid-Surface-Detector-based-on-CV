# Baseline reproduction status

Updated 2026-08-25 on the RTX 5090 server. “Runnable” means the official revision and released checkpoint load strictly under the modern compatibility adapter and complete a real project RGB-D frame. Promotion still requires the difficult-scene benchmark.

## Reproduction status

| Baseline | Official revision | Checkpoint | Runtime status | Legacy 10-frame MAE / RMSE | Provisional latency |
| --- | --- | --- | --- | --- | --- |
| Classical raw RGB-D | project regression | n/a | frozen control | 0.831 / 0.967 cm | not reprofiled |
| TransCG DFNet | 135f9e0 | corrected 2022-10-14, MD5 `6e6e00f7cc02c644a34b1ce6e4f364fa` | runnable, official preprocessing | 0.535 / 0.669 cm | about 0.76 s including CPU nearest-neighbor fill |
| DREDS SwinDRNet | 1b0ac30 | official `model.pth`, strict 650-state load | runnable, official validation resize | 0.753 / 0.820 cm | about 0.48 s |
| ClearGrasp | 0688647 | downloading | adapter/reproduction pending verified extraction | pending | pending |
| RGB-D LIDF | 4dc85bb | pending dataset/checkpoint | source ready; CUDA extension needs current-toolchain port | pending | pending |

DFNet is the current leader on the 10 ordinary project frames. This sequence is only a regression gate and must not decide the research baseline by itself.

## Common DREDS STD real-data gate

The first common evaluation uses one frame from each of 12 real test sequences. Pixels whose source instance mask is `255` are excluded as background. Values below are evaluated over 334,483 object pixels.

| Method | Coverage | MAE | RMSE | Boundary RMSE | Median warm latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw RGB-D identity | 96.92% | 14.75 mm | 26.88 mm | 33.42 mm | 10.6 ms |
| TransCG DFNet | 100% | 14.84 mm | 22.95 mm | **28.22 mm** | 90.9 ms |
| DREDS SwinDRNet | 100% | **9.72 mm** | **19.47 mm** | 37.67 mm | **37.9 ms** |

SwinDRNet is currently the strongest overall DREDS-domain restoration baseline, while DFNet is better at this sample's object boundary. The next promotion decision combines this gate, the project regression gate, and explicit transparent/translucent/glare/saturated-highlight/container-edge partitions after ClearGrasp and LIDF are reproduced.

The current manifest labels each complete object frame with all three broad DREDS material tags, so the preliminary per-tag rows duplicate the overall metric. Material-specific masks will be built from DREDS metadata before any per-material claim is made.

Common outputs and evaluation artifacts stay under:

    /root/autodl-tmp/liquid-depth-artifacts/evaluation

The runtime adapters are selected through `depth_refinement.backend`:

    identity
    transcg_dfnet
    dreds_swindrnet
    torchscript

Batch inference constructs the model once and reuses it across frames. Both released networks lack project-calibrated uncertainty: DFNet reports a neutral confidence of 0.5 for valid output, while SwinDRNet exposes its raw-versus-restored blending confidence. Neither value may be treated as calibrated selective-risk confidence until evaluated on difficult held-out sequences.
