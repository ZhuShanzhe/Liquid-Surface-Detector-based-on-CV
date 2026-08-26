# Executable algorithm roadmap

This document is the implementation contract for the transparent/specular liquid-depth program. The production path remains a stable RGB-D frame contract; research models are interchangeable behind metric depth and confidence outputs.

## Stage gates and current implementation

| Stage | Deliverable | Promotion gate |
| --- | --- | --- |
| R0 | Frozen classical RGB-D geometry baseline | Reproducible metrics and scenario/sequence split |
| R1 | DFNet, SwinDRNet, and ClearGrasp comparison; RGB-D LIDF optional | Best difficult-scene depth error at useful accepted coverage |
| R2 | Project multi-task RGB-D model | Beats R1 end-to-end liquid-depth error without easy-scene regression |
| R3 | Interior/meniscus physics and SeeGroup auxiliary study | Improves ambiguity handling and calibrated rejection |
| R4 | Robust temporal fusion | Lower static jitter without lagging real level changes |
| R5 | Polarization, multi-exposure, or short-baseline stereo | Only if synchronized single RGB-D remains uncertainty-limited |

R0 regression is frozen. DFNet, SwinDRNet, and ClearGrasp satisfy the R1 gate, with SwinDRNet selected as the deployment baseline. LIDF is optional and does not block R2. The R2 model, R3 geometry hooks, R4 filter, and common evaluators are implemented.

## Server locations

    repository  /root/autodl-tmp/Liquid-Surface-Detector-based-on-CV
    environment /root/autodl-tmp/envs/liquid-depth
    research    /root/autodl-tmp/liquid-depth-data/research
    artifacts   /root/autodl-tmp/liquid-depth-artifacts

Datasets, checkpoints, extracted archives, logs, and environments never enter Git.

## R1: reproduce and select a depth-restoration baseline

The official revisions and file contracts are pinned in configs/baselines.yaml. Audit readiness:

    conda activate /root/autodl-tmp/envs/liquid-depth
    python scripts/audit_baselines.py \
      --output /root/autodl-tmp/liquid-depth-artifacts/audits/baseline_readiness.json

Download public official files with resumable aria2c, after reviewing their licenses:

    python scripts/download_research_data.py \
      --dataset dreds_std dreds_swindrnet_weights cleargrasp_eval cleargrasp_train todd \
      --accept-license
    python scripts/prepare_research_data.py \
      --dataset dreds_std dreds_swindrnet_weights cleargrasp_eval cleargrasp_train todd

Extraction is a separate explicit operation and happens only after size/checksum/archive tests:

    python scripts/prepare_research_data.py \
      --dataset cleargrasp_eval todd --extract

Run each official implementation through an isolated adapter. Do not downgrade the RTX 5090 project environment to historical PyTorch/CUDA versions. LIDF CUDA compatibility is available through `scripts/setup_lidf.sh`, but its unavailable legacy checkpoint is not a promotion dependency.

All candidates produce the same per-frame record and are ranked by:

- transparent-mask MAE/RMSE in meters and boundary RMSE;
- valid prediction coverage and confidence-error calibration;
- end-to-end liquid-depth MAE/RMSE in centimeters;
- accepted-sample coverage and rejection reasons;
- latency and peak GPU memory at deployment resolution.

Use scripts/evaluate_depth_restoration.py for common pixel metrics. A baseline is selected on the held-out project validation sequences, not on its own training benchmark alone.

## R2: project multi-task network

The canonical five-channel input is ImageNet-normalized RGB, raw depth divided by max_depth_m, and a raw-depth validity channel. The network jointly predicts:

- liquid mask logits;
- restored metric depth in meters;
- unit surface normal;
- log variance (heteroscedastic uncertainty);
- confidence.

The CSV manifest contract is:

    rgb_path,raw_depth_path,target_depth_path,mask_path,normal_path,split,sequence_id,difficulty_tags,depth_scale_to_m

Build the verified research pretraining manifest directly from the extracted datasets:

    python scripts/build_research_multitask_manifest.py \
      --output /root/autodl-tmp/liquid-depth-data/research/manifests/research_multitask.csv

The generated manifest adds `dataset`, `normal_channel_order`, and `corrupt_depth_in_mask`. Missing normal maps are derived from metric target depth; multi-channel depth EXRs select the channel containing valid depth. ClearGrasp synthetic depth is corrupted only inside the transparent mask so the network cannot learn a trivial depth copy.

difficulty_tags is a semicolon-separated subset of:

    transparent;translucent;glare;saturated_highlight;container_edge;ordinary

Training augmentation explicitly synthesizes exposure shifts, saturated glare blobs, sensor-depth dropout under highlights, and noise. These object datasets pretrain transparent/non-Lambertian restoration; they do not replace the later liquid-surface fine-tuning set with true liquid masks and liquid-plane labels.

Split by whole capture sequence/container/liquid/lighting session. Never split adjacent video frames. Training uses inverse-frequency scenario sampling so ordinary liquid cannot dominate difficult cases.

    python scripts/train_multitask.py \
      --manifest /root/autodl-tmp/liquid-depth-data/manifests/multitask.csv \
      --output-dir /root/autodl-tmp/liquid-depth-artifacts/training/multitask-v1
    python scripts/export_multitask.py \
      --checkpoint /root/autodl-tmp/liquid-depth-artifacts/training/multitask-v1/best.pth \
      --output /root/autodl-tmp/liquid-depth-artifacts/models/multitask-v1.ts

The loss combines Dice/BCE mask loss, uncertainty-weighted metric depth loss, cosine normal loss, depth-gradient consistency, and expected-plane-normal physics loss.

## R3: liquid geometry and physical constraints

Only the eroded liquid interior contributes to the primary plane. The boundary band is saved as a separate meniscus mask and may later feed a curved-surface head. Quality gating uses:

- plane support and robust residual;
- segmentation and restored-depth confidence;
- liquid/bottom plane parallelism;
- gravity direction when an IMU/extrinsic calibration is available;
- plausible change relative to historical liquid depth.

Every result includes confidence, accepted, and machine-readable rejection_reasons. A rejection is a valid system output and must not be silently replaced with an untrusted number.

SeeGroup is an auxiliary multi-layer representation for ambiguous transparent rays. It does not directly replace metric calibration: select the liquid-interface mode using mask, gravity, bottom geometry, and temporal continuity, then align it to reliable metric RGB-D support.

## R4: video fusion

Temporal fusion is opt-in for ordered frames:

    liquid-depth --config configs/pipeline.yaml batch \
      --input-dir /path/to/ordered/frames \
      --bottom-plane /path/to/bottom_plane.json \
      --output-dir /path/to/results \
      --temporal

The robust Kalman filter scales measurement noise by confidence and rejects low-confidence, excessive-jump, or statistically inconsistent innovations. Evaluation must report raw and filtered jitter, response lag to genuine level changes, accepted coverage, and reason counts.

## R5: evidence required before hardware changes

Escalate beyond one synchronized RGB-D camera only when all of the following hold:

1. R1/R2 uncertainty is calibrated and consistently high for the same physical cases.
2. Additional RGB-D training data and exposure randomization no longer improve held-out error.
3. Failures correspond to missing/ambiguous sensor evidence, not label or calibration error.
4. The added modality measurably improves difficult-scene error and accepted coverage.

Test multi-exposure first if saturation dominates, polarization if specular separation dominates, and short-baseline stereo if metric correspondence remains recoverable from a second view.
