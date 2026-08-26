# Transparent and reflective liquid-depth research plan

## Feasibility conclusion

The existing acquisition-to-depth architecture is feasible and should remain the production skeleton. Its classical
segmentation and RANSAC path is a valuable deterministic baseline, but it assumes that depth samples inside the liquid
mask belong to the liquid surface. This assumption fails for clear liquid, transparent containers, glare, and strong
specular reflection: an active RGB-D camera may return the background, a refracted path, multipath interference, or no
sample at all. A stronger plane fitter cannot recover geometry that was never measured correctly.

The research system therefore keeps the stable I/O and calibrated geometry, and inserts learned restoration before
plane fitting:

```text
synchronized RGB + raw depth + intrinsics
              |
              v
liquid/container/invalid-depth segmentation and confidence
              |
              v
RGB-D metric depth restoration + surface normal + uncertainty
              |
              v
confidence-filtered plane/meniscus geometry + bottom calibration
              |
              v
temporal and physical consistency checks
              |
              v
depth_cm + uncertainty + accepted/rejected + diagnostics
```

This is executable incrementally: `depth_refinement.backend: identity` reproduces the current baseline, while a trained
model can be exported behind the five-channel TorchScript contract without changing capture, calibration, or output
consumers.

## Why difficult scenes fail

- Transparent liquid has weak intrinsic texture. RGB mostly contains the background transformed by refraction.
- Active stereo/structured light can match the pattern on the container bottom or background instead of the interface.
- ToF sensors can mix direct and reflected paths. A plausible nonzero depth is not necessarily a valid surface return.
- Saturated highlights destroy both RGB cues and active-pattern correspondence.
- One image ray may legitimately contain several visible interfaces: air/glass, glass/liquid, liquid/glass, and the
  background. A conventional single-depth target is physically ambiguous.
- The meniscus is curved near the wall, while the interior is approximately planar only in static conditions.

These failure modes require an explicit validity/confidence representation. Zero-depth masking alone is insufficient.

## Recommended model sequence

### R0: controlled baseline and evaluation

Freeze the current classical result as `baseline-classical-v1`. Create sequence-level train/validation/test splits by
container, liquid, lighting, camera pose, and capture session. Report end-to-end MAE/RMSE, 95th-percentile error,
accepted-sample coverage, scenario metrics, and temporal jitter. Never randomly split adjacent video frames.

### R1: RGB-D restoration baseline (highest priority)

Benchmark three released families before designing a new network:

1. TransCG DFNet on real raw/ground-truth RGB-D.
2. DREDS SwinDRNet on synthetic active-sensor corruption and real specular/transparent scenes.
3. ClearGrasp or RGB-D Local Implicit Function as normal/boundary or local-geometry baselines.

Adapt the best model to input `[RGB, metric raw depth, validity]` and output `[metric refined depth, confidence]`. Fine
tune on project captures with ground truth liquid height. Treat wrong nonzero sensor returns as corrupted, not trusted.

### R2: project-specific multi-task network

Use a shared image encoder and separate heads for:

- liquid-surface/container mask and boundary;
- restored metric depth or plane offset;
- surface normal;
- heteroscedastic uncertainty/confidence;
- optional transparent/specular material class.

Recommended losses are robust metric depth loss on labeled pixels, gradient/edge loss, cosine normal loss, mask Dice +
focal loss, bottom-parallel plane loss, and negative log-likelihood for predicted uncertainty. Sample batches by scenario
so easy opaque/colored liquid does not dominate clear-liquid and glare cases. Synthetic rendering should randomize
refractive index, absorption, roughness, container thickness, background, exposure, emitter pattern, and sensor noise.

### R3: SeeGroup auxiliary branch

[SeeGroup (CVPR 2026)](https://github.com/princeton-vl/SeeGroup) models all surface events along a ray rather than forcing
one depth. A recurrent decomposition module extracts self-determined feature components. Each component predicts a
Laplace center and scale; their point-process intensity is a maximum mixture. The likelihood of a set of ground-truth
depths is a product of intensities and is therefore permutation invariant. This avoids forcing a coherent physical
surface into a fixed front-to-back map when transparent layers overlap.

Use it as an auxiliary representation, not the immediate centimeter output:

1. pretrain/evaluate on LayeredDepth-Syn/LayeredDepth;
2. add RGB-D validity and raw depth as extra conditioning;
3. identify the liquid-interface peak with segmentation, expected gravity, and temporal continuity;
4. align scale using reliable raw depth outside corrupt regions and the calibrated container bottom;
5. feed the selected metric interface and uncertainty to the existing plane/height estimator.

The released SeeGroup model is monocular and scale-invariant, LayeredDepth real annotations are relative, and the paper
reports over-prediction of layers and out-of-distribution failures. It must not bypass metric calibration or quality
gating. Its training recipe used 250k steps on four L40 GPUs, so first reproduce inference and validation on the RTX 5090;
then fine-tune a smaller adapter before attempting full training.

### R4: temporal and physical fusion

For video, estimate depth per frame and fuse plane offset with a robust Kalman filter or sliding-window optimizer. Use
gravity/IMU when available; a static liquid surface should be perpendicular to gravity and nearly parallel to the
calibrated bottom. Reject abrupt jumps, excessive surface tilt, low support, or multimodal ambiguity. Estimate the
interior plane after eroding the meniscus boundary, while retaining a separate boundary/meniscus feature for cases where
interior depth is absent.

### R5: hardware escalation only if passive cues saturate

If model uncertainty remains high in the same scenarios, collect complementary cues instead of hiding the failure:

- crossed or rotating polarization for specular/transparent segmentation;
- stereo or a short-baseline second view;
- multiple exposure/illumination states;
- a second sensor modality such as ultrasonic point height for calibration checks.

[Deep Polarization Cues (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Kalra_Deep_Polarization_Cues_for_Transparent_Object_Segmentation_CVPR_2020_paper.html)
shows why polarization separates transparent-object signatures that intensity misses. [SPIDeRS (CVPR
2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Ichikawa_SPIDeRS_Structured_Polarization_for_Invisible_Depth_and_Reflectance_Sensing_CVPR_2024_paper.html)
is a stronger but custom-hardware option. These are later hardware branches, not prerequisites for R1-R3.

## Dataset decisions

| Dataset | Match to current task | Decision |
| --- | --- | --- |
| DTLD (ECCV 2024) | 27,678 RealSense D435 RGB-D frames with contact-line annotations, masks, camera/pose metadata, and per-instance liquid height in millimeters. | **Highest priority and downloaded first.** This is a direct end-to-end liquid-height stress benchmark, but its short 15-96 mm range cannot alone certify the intended operating range. |
| Phys-Liquid (AAAI 2026) | Physics-informed transparent-liquid images and 3-D meshes across containers, lighting, colors, and rotations. | **Medium-high priority.** Use for liquid mask/geometry/volume auxiliary pretraining after DTLD; it does not replace real metric validation. |
| TransCG | Real RGB-D, raw and refined metric depth, masks/normals; 57,715 images. Same input/output family as this project. | **High priority.** Download metadata and scenes 1-10 pilot first; expand only after loader and baseline pass. |
| DREDS/STD | Synthetic sensor-corrupted RGB-D plus real specular/transparent scenes and released SwinDRNet. | **High priority.** Best source for material and sensor-noise robustness; non-commercial license. |
| ClearGrasp | 50k+ synthetic images with masks/normals/boundaries and 286 real RGB-D tests. | **High priority.** Good auxiliary pretraining/benchmark; limited liquid-specific content. |
| TODD/TranspareNet | Nearly 15k real transparent-object RGB-D images; paper explicitly includes vessels with partially filled fluids. | **High priority if the official download remains available.** Very close semantic match. |
| LayeredDepth / LayeredDepth-Syn | Multi-layer transparent RGB benchmark and synthetic metric layers; direct basis for SeeGroup. | **Medium priority.** Useful for the multi-layer branch, not direct metric liquid-height supervision. |
| Booster | Accurate high-resolution stereo disparity, transparent/specular material masks. | **Medium priority.** Excellent stress benchmark if stereo is added; registration and large storage make it non-immediate. |
| Mirror3D | Mirror masks and plane parameters from indoor RGB-D context. | **Low priority.** Useful for context-to-plane auxiliary learning, but mirrors are not refractive liquid surfaces and may cause negative transfer. |
| HCI 4D Light Field | Synthetic light-field disparity benchmark. | **Do not download now.** Input is many sub-aperture views, unlike the current single RGB-D camera. |
| UrbanLF / Non-Lambertian-LF | Light-field semantic/depth data with a dedicated non-Lambertian subset. | **Hardware-gated.** Use only if a light-field/multi-view acquisition branch is adopted. |

Primary sources: [TransCG](https://graspnet.net/transcg), [DREDS](https://github.com/PKU-EPIC/DREDS),
[ClearGrasp](https://sites.google.com/view/cleargrasp/data), [Seeing Glass / TODD](https://proceedings.mlr.press/v164/xu22b.html),
[LayeredDepth](https://github.com/princeton-vl/LayeredDepth), [Booster](https://cvlab-unibo.github.io/booster-web/),
[Mirror3D](https://openaccess.thecvf.com/content/CVPR2021/html/Tan_Mirror3D_Depth_Refinement_for_Mirror_Surfaces_CVPR_2021_paper.html),
and [UrbanLF](https://github.com/HAWKEYE-Group/UrbanLF).

## Experimental acceptance criteria

The industrial requirement is tracked in
`configs/accuracy_profile_industrial_v1.yaml`. It uses
`max(absolute millimeter floor, relative percentage * true liquid depth)`
because a pure percentage becomes physically meaningless close to zero. The
primary bands are 0.2-1 m, 1-5 m, and 5-10 m; 20-200 mm is retained as a
near-zero stress band for DTLD.

From 1-10 m the research target is approximately 1% mean error, with P95 at
approximately 2%. Close range uses tighter millimeter goals where feasible. A
separate deployment gate is deliberately wider, especially at 5-10 m, and may
only be enabled after the exact sensor, mode, pose, and container geometry pass
traceable qualification. Camera standoff is evaluated independently from the
liquid-depth measurand.

Every report includes MAE, RMSE, signed bias, P95, temporal jitter, accepted
coverage, and false acceptance, split by container, viewpoint, lighting, liquid
appearance, glare, and occlusion. A low error obtained by rejecting most frames
is not acceptable. VDI/VDE 2634-style flatness/length checks and ASTM E2938
relative-range methodology guide sensor qualification; neither standard
prescribes this application's permissible error.

Every candidate must be compared with the frozen baseline using the same split and calibration. Promotion requires:

- lower overall and difficult-scenario MAE/RMSE, not just higher mask IoU;
- reported accepted coverage and error on accepted samples (selective risk);
- no regression on opaque/colored liquid beyond an agreed tolerance;
- stable temporal output and no leakage between adjacent sequences;
- latency and GPU memory measured at deployment resolution;
- calibrated confidence: high-error transparent/glare frames should be rejected rather than silently accepted;
- an ablation separating segmentation, depth restoration, geometry, and temporal gains.

## Executable research runbook

```bash
conda activate /root/autodl-tmp/envs/liquid-depth
cd /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV

# Verify runtime and all existing captures.
python scripts/audit_system.py --require-cuda --data-root /root/autodl-tmp/liquid-depth-data
python scripts/validate_frames.py /root/autodl-tmp/liquid-depth-data/legacy_rgbd \
  --output /root/autodl-tmp/liquid-depth-artifacts/audits/legacy_frames.json

# Inspect/download approved research foundations.
python scripts/download_research_data.py --list
python scripts/download_research_data.py \
  --dataset transcg_info transcg_pilot transcg_code seegroup_code layereddepth_code layereddepth \
  --accept-license --extract

# Run the stable baseline and evaluate scenario-aware end-to-end error.
liquid-depth --config configs/pipeline.yaml batch \
  --input-dir /root/autodl-tmp/liquid-depth-data/legacy_rgbd \
  --bottom-plane /root/autodl-tmp/liquid-depth-artifacts/calibration/bottom_plane.json \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/evaluation/baseline
python scripts/evaluate_liquid_depth.py \
  --ground-truth legacy/known_depths_20260517.csv \
  --predictions /root/autodl-tmp/liquid-depth-artifacts/evaluation/baseline \
  --output /root/autodl-tmp/liquid-depth-artifacts/evaluation/baseline_metrics.json

# Train/export the segmentation baseline after reviewed labels exist.
python scripts/train_segmentation.py --manifest /path/to/manifest.csv --output-dir /path/to/run
python scripts/export_segmentation.py --checkpoint /path/to/run/best.pth --output weights/liquid_surface.ts
```

Large datasets and external source trees remain outside Git. Their registry, provenance marker, loaders, experiment
configs, and metrics belong in Git; downloaded bytes and model weights do not.
