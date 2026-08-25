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
