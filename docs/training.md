# Training

## Label policy and sequence splits

Automatically generated legacy masks are proposals, not ground truth. Review and correct them before training. Split by complete physical capture sequence/container/liquid/lighting session, never by neighboring video frame.

The held-out test set must include transparent and translucent liquid, saturated highlight, glare, container edges, different containers/camera poses, occlusion, and invalid or plausible-but-wrong raw depth.

## Segmentation baseline

Create a CSV manifest:

    image_path,mask_path,split,scenario
    images/frame_001.png,masks/frame_001.png,train,clear_liquid_daylight
    images/frame_002.png,masks/frame_002.png,val,colored_liquid_reflection

Train the modular DeepLabV3-ResNet50 baseline:

    python scripts/train_segmentation.py \
      --manifest /root/autodl-tmp/liquid-depth-data/labels/manifest.csv \
      --output-dir /root/autodl-tmp/liquid-depth-artifacts/training/segmentation-baseline

## Project multi-task RGB-D model

The canonical CSV fields are:

    rgb_path,raw_depth_path,target_depth_path,mask_path,normal_path,split,sequence_id,difficulty_tags,depth_scale_to_m

Paths may be absolute or relative to the manifest. Depth can be NPY or an image; depth_scale_to_m explicitly converts stored units to meters. If omitted, a median greater than 10 is interpreted as millimeters. Normal maps are HxWx3 NPY/images, either float unit vectors or uint8 values mapped from 0..255 to -1..1.

difficulty_tags is semicolon separated. Use one or more of:

    transparent
    translucent
    glare
    saturated_highlight
    container_edge
    ordinary

The loader builds the five-channel input RGB + normalized raw depth + validity. Training uses inverse-frequency difficulty weights and jointly supervises liquid mask, metric restored depth, normals, and heteroscedastic uncertainty.

    python scripts/train_multitask.py \
      --manifest /root/autodl-tmp/liquid-depth-data/manifests/multitask.csv \
      --output-dir /root/autodl-tmp/liquid-depth-artifacts/training/multitask-v1 \
      --epochs 80 --batch-size 8 --image-size 640,360

Export the best checkpoint to the inference contract:

    python scripts/export_multitask.py \
      --checkpoint /root/autodl-tmp/liquid-depth-artifacts/training/multitask-v1/best.pth \
      --output /root/autodl-tmp/liquid-depth-artifacts/models/multitask-v1.ts

Configure depth_refinement.backend as torchscript and set the exported model path. The adapter consumes depth_m and confidence (or derives confidence from log_variance) without changing camera acquisition or geometry.

## Required evaluation

Report metrics globally and by difficulty tag:

- liquid-mask IoU;
- transparent-mask depth MAE/RMSE and boundary RMSE in meters;
- valid prediction coverage;
- confidence-error correlation and selective risk/coverage;
- final liquid-depth MAE/RMSE in centimeters;
- accepted coverage, rejection reasons, latency, and GPU memory;
- raw and temporally filtered jitter on static videos.

A model is promoted only if it improves difficult-scene depth and end-to-end liquid height without an unacceptable regression on ordinary liquid.

## DTLD liquid-height adaptation

DTLD is the primary end-to-end benchmark because it supplies RealSense D435 RGB-D,
object and visible masks, contact-line control points, camera/pose metadata, and
per-instance liquid height in millimeters. Build the project manifest after the
official archive is extracted:

    python scripts/build_dtld_manifest.py \
      --root /root/autodl-tmp/liquid-depth-data/research/dtld/extracted/DTLD_dataset \
      --split-map configs/dtld_scene_split_v1.json \
      --output /root/autodl-tmp/liquid-depth-data/research/manifests/dtld_v1.csv

The release does not designate official train/validation/test subsets. Use the
tracked scene-level split map and never randomly split adjacent frames from a
capture; doing so would leak nearly identical video frames into evaluation.
The verified v1 manifest contains 90,734 labeled instances: 61,203 train,
12,881 validation, and 16,650 test. The upstream archive contains one
zero-filled depth file and 46 zero-filled visible-mask files despite passing
ZIP CRC validation. The builder excludes the four affected depth instances and
records affected optional visible masks as absent.
Use the object-domain multi-task checkpoint only as initialization, keep
SwinDRNet as the restoration control, and supervise contact line/liquid height
directly. At an approximately 1 m working distance, promotion requires liquid
height MAE <= 1 cm and must also report RMSE, P95 absolute error, within-1-cm
ratio, accepted coverage, and per-container/per-lighting results.

The first DTLD perception baseline predicts a contact-line heatmap and a
pose-conditioned metric height with heteroscedastic uncertainty. It uses an
instance crop supplied by the container detection/pose stage; no ground-truth
liquid mask or contact point is provided as model input. Start with a
scene-subsampled pilot:

    python scripts/train_dtld_height.py \
      --manifest /root/autodl-tmp/liquid-depth-data/research/manifests/dtld_v1.csv \
      --output-dir /root/autodl-tmp/liquid-depth-artifacts/training/dtld-contact-height-v1 \
      --epochs 12 --batch-size 32 --train-stride 10 --val-stride 10

Remove the stride arguments for the final run only after the pilot reduces
scene-held-out MAE. This baseline isolates contact-line perception; the final
geometry head must convert the detected contact curve using calibrated container geometry.
