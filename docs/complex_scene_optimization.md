# Complex-scene algorithm design

## Large-area depth failure: temporal anchor memory

The runtime separates two temporal mechanisms:

- the scalar robust Kalman filter stabilizes final liquid depth;
- `TemporalAnchorMemory` stores directly observed, high-confidence contact
  points and recovers spatial support when raw depth later fails.

The anchor path activates only in video temporal mode and when raw-depth
validity falls below 0.45 by default. Stored anchors are mapped to container
coordinates and reprojected with the current pose. Pyramidal Lucas-Kanade
optical flow supplies a second alignment estimate; contradictory pose and flow
tracks are rejected.

RGB guidance is a conservative gate rather than an unconstrained generator.
Lab-color similarity decays historical confidence, so an appearance change
blocks stale reuse. The observability gate still requires:

- fresh current-frame anchors;
- enough total anchors and occupied horizontal bins;
- a bounded memory-derived fraction;
- valid pose, flow consistency, age decay and RGB similarity.

Only fresh model predictions are committed back to memory, preventing recovered
points from recursively reinforcing themselves. Runtime output reports
activation, history size, recovered points, memory fraction, spatial coverage,
alignment error, RGB similarity and explicit rejection reasons.

### Synthetic qualification

A deterministic 65-frame point-level sequence was used to verify the recovery
and refusal logic independently of camera hardware:

- with 25% fresh anchors, accepted output increased from 0% without memory to
  100% with memory; accepted-point mean error was 0.28 px;
- with 12% fresh anchors, accepted output increased from 0% to 55%; accepted-
  point mean error was 0.24 px;
- after a deliberate RGB appearance change, memory acceptance remained 0% and
  no stale anchors were recovered;
- CPU fusion latency was 9.3 ms mean and 10.8 ms P95; one scheduling outlier
  reached 80.5 ms.

These figures qualify the sparse-anchor mechanism, not end-to-end metric liquid
depth. Simulation sequences and then real RGB-D video must still measure plane
error, liquid-depth error, coverage, false acceptance and total latency.

## Scope and conclusion

A ray passing through a transparent vessel may intersect the front wall, liquid
interface, rear wall, and background. A single-depth RGB-D camera can therefore
return a missing sample or a coherent but physically wrong layer. SeeGroup is
directly relevant to representing this ambiguity, but its monocular relative
layers are not a calibrated liquid-depth measurement.

The production design keeps metric RGB-D and calibrated geometry as the final
authority. Multi-layer prediction supplies candidates and uncertainty; a liquid
interface is selected only when it agrees with the liquid mask, metric raw-depth
evidence, bottom/CAD geometry, gravity, and temporal history. The system rejects
a frame when no candidate satisfies those constraints.

## Runtime path

```text
RGB + raw metric depth
  -> exposure diagnostics (conditional lightweight correction)
  -> liquid/contact segmentation and invalid-depth confidence
  -> metric depth/normal/uncertainty restoration
  -> optional multi-layer ray candidates
  -> metric-prior layer selection
  -> distributed sparse support + robust plane/contact-line fit
  -> bottom/CAD distance + temporal filter
  -> liquid depth, confidence, and explicit rejection reason
```

The current implementation adds:

- bounded illumination diagnostics and optional gamma/CLAHE preprocessing;
- spatial support checks that reject a compact accidental patch;
- a lightweight metric multi-layer head and permutation-invariant set loss;
- metric-prior layer selection that never promotes an unsupported layer;
- dark, shadow, glare-like, dropout, and floating-occluder augmentation;
- tail-aware contact-curve loss for improving P95 rather than only mean error.

Illumination correction and surface-support rejection are feature flags. They
remain conservative until the difficult-scenario validation split proves that
enabling them improves selective risk without lowering accepted coverage too far.

## Floating material and non-flat surfaces

Floating foam, solids, bubbles, and labels are first treated as occlusion, not as
liquid surface. Plane support must span multiple spatial tiles and sufficient
horizontal/vertical extent. The interior estimator uses confidence-filtered
points away from the meniscus and robustly removes floating-object residuals.

If distributed planar support remains, report the representative bulk-liquid
plane and record the excluded fraction. If support is compact, multimodal, or
temporally unstable, reject with `insufficient_spatial_support`; do not force a
plane. Truly sloshing/non-planar liquid requires a surface/volume extension and
must not be represented as a high-confidence scalar depth.

## Poor illumination

The default path measures median luminance, dark/saturated ratios, and dynamic
range in roughly two milliseconds at 640x480. Normally exposed images remain
unchanged. A dark frame may receive bounded gamma correction; unusable darkness,
clipping, or low dynamic range is rejected.

Training augmentation remains the primary defense because enhancement can change
transparent boundaries and highlights. If a learned enhancer is later needed,
use a real-time exposure-correction or edge-oriented model and train it jointly
with the downstream task. Large diffusion enhancement is outside the latency and
determinism budget.

Relevant primary sources include [CoTF real-time exposure correction (CVPR
2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Real-Time_Exposure_Correction_via_Collaborative_Transformations_and_Adaptive_Sampling_CVPR_2024_paper.html),
[optimized low-light enhancement for edge vision (CVPRW
2024)](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/html/Sharif_Learning_Optimized_Low-Light_Image_Enhancement_for_Edge_Vision_Tasks_CVPRW_2024_paper.html),
and [zero-reference physical quadruple priors (CVPR
2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Zero-Reference_Low-Light_Enhancement_via_Physical_Quadruple_Priors_CVPR_2024_paper.html).

## SeeGroup deployment decision

The official SeeGroup ViT-L checkpoint has 356.26 million parameters. On the
project RTX 5090, one 640x480 image measured after warm-up as follows:

| Internal input size | Mean latency | P95 latency | Peak allocated memory |
| --- | ---: | ---: | ---: |
| 322 | 39.27 ms | 39.47 ms | 1.77 GB |
| 392 | 51.80 ms | 52.12 ms | 1.91 GB |
| 518 | 86.13 ms | 86.41 ms | 2.25 GB |

These are isolated model measurements, not full camera-to-output latency. The
518 model is easily inside the revised 500 ms end-to-end P95 budget in isolation.
It is nevertheless loaded only for a qualified scene profile or automatic trigger;
model initialization is excluded by keeping the service resident. The recommended sequence is:

1. reproduce the official LayeredDepth validation metrics locally;
2. use full SeeGroup as an offline teacher and difficult-frame reference;
3. distill layer candidates into the 18,892-parameter `RayLayerHead` attached to
   the existing RGB-D decoder;
4. optionally invoke full SeeGroup only for ambiguous/rejected frames if an
   end-to-end benchmark still stays below the deployment limit.

A local 300-image LayeredDepth validation subset reproduced overall pair/trip/
quad accuracies of 83.17% / 75.71% / 72.46%. This is a subset check rather than
the complete official benchmark and must not be compared as a new state of the
art result. Its fake-layer tuple accuracy was approximately zero, reinforcing
the decision to select layers using metric geometry and to reject unresolved
ambiguity instead of trusting every predicted interface.

SeeGroup's official [CVPR 2026
paper](https://openaccess.thecvf.com/content/CVPR2026/html/Wen_SeeGroup_Multi-Layer_Depth_Estimation_of_Transparent_Surfaces_via_Self-Determined_Grouping_CVPR_2026_paper.html)
reports that self-determined grouping improves quadruplet relative-depth
accuracy from 61.34 to 70.09. [LayeredDepth (ICCV
2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wen_Seeing_and_Seeing_Through_the_Glass_Real_and_Synthetic_Data_ICCV_2025_paper.pdf)
provides the real and synthetic multi-layer supervision. Neither source claims
centimeter-level metric liquid-height accuracy, so calibration remains mandatory.

## Dataset use

- DTLD/TCLD: direct container-plus-liquid contact-line and lighting benchmark;
  use for tail-aware contact perception and selective confidence.
- LayeredDepth and LayeredDepth-Syn: multi-layer ordering and distillation;
  never use their real relative depths as metric height ground truth.
- TransCG, DREDS/STD, ClearGrasp, TODD: RGB-D corruption, normals, transparent
  masks, and non-Lambertian transfer.
- Phys-Liquid: deformable liquid, lighting, rotation, and temporal auxiliary
  supervision. Full raw assets are needed; the 2.7 MB Hugging Face parquet file
  contains only the image column and is not a complete physical-label download.
- TRADE: optional 4.3 GB real stress set for containers, fill levels, refraction,
  reflection, and strong light. Its 122.5 GB simulation set is lower priority
  than current LayeredDepth-Syn completion.
  `scripts/build_trade_manifest.py` creates deterministic scene-disjoint splits,
  preserves per-object fill fractions, and optionally records exposure metrics.
  The first extracted real release contains 2,856 paired RGB-D frames across 34
  scenes, including 960 frames from scenes with at least one partially filled
  object. It is a robustness/relative-fill benchmark; fill fractions are not
  direct metric liquid-depth labels.

Dataset licenses are tracked separately. Non-commercial datasets may support
research and a software-copyright demonstration but must be reviewed before any
commercial industrial deployment.

## Promotion gates

Promote an experiment only when the same held-out sequences show:

- lower difficult-scene MAE/RMSE and P95 error;
- calibrated accepted-sample coverage and false-acceptance rate;
- no unacceptable regression on opaque/colored liquids;
- stable temporal output and explicit rejection of unsupported geometry;
- warm full camera-to-output P95 latency no greater than 500 ms on the target profile,
  reported separately for standard and specialist routes;
- metric accuracy evaluated against the industrial distance-band profile, not
  inferred from LayeredDepth ordering or DTLD pixel error.

The first DTLD tail-loss pilot was not promoted. Against the same 2,577-image
validation stride, the frozen model achieved 23.22 px MAE / 88.25 px P95 while
the best tail-loss checkpoint achieved 23.61 px / 90.46 px. The experiment did
slightly improve contact IoU and confidence-error correlation, but worsened the
primary and tail errors. These artifacts remain outside Git for reproducibility;
the production checkpoint is unchanged.
