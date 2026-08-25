# Segmentation training

Do not train on the automatically generated legacy masks as if they were ground truth. Review and correct masks first,
then create a CSV manifest:

```csv
image_path,mask_path,split,scenario
images/frame_001.png,masks/frame_001.png,train,clear_liquid_daylight
images/frame_002.png,masks/frame_002.png,val,colored_liquid_reflection
```

Split by physical capture sequence/container, not random neighboring frames, to prevent leakage. Keep a held-out test
set covering difficult lighting, transparent and colored liquids, container shapes, camera angles, partial occlusions,
and missing/noisy depth.

```bash
python scripts/train_segmentation.py \
  --manifest /root/autodl-tmp/liquid-depth-data/labels/manifest.csv \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/training/baseline
```

The baseline is DeepLabV3-ResNet50. It is intentionally modular: stronger architectures can replace it while the
inference pipeline continues to consume a binary mask and confidence map.

