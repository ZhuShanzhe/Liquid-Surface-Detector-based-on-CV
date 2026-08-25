# System architecture

```text
Orbbec Gemini 2
      |
      v
ROS 2 camera driver -> synchronized RGB/depth capture -> frame directory contract
                                                        |
                                                        v
                        liquid mask (classical or neural segmenter)
                                                        |
                                                        v
                          masked depth -> robust 3-D plane fit
                                                        |
                                                        v
                 bottom-plane calibration -> liquid depth + quality metrics
```

## Deployment boundary

The camera driver and capture node run on the machine physically attached to the USB camera. The training,
offline evaluation, and model optimization environment runs on the GPU server. The frame directory contract is the
boundary between them, so the algorithms are independent from ROS messages and can also process recorded data.

## Robustness roadmap

1. Curate train/validation/test splits by container, liquid appearance, illumination, viewpoint, reflections, and
   occlusion. Keep raw frames immutable and masks separately reviewed.
2. Replace the classical HSV segmenter with a segmentation model while retaining the same `Segmenter` interface.
3. Fuse model confidence with depth validity and plane-fit residuals. Reject uncertain estimates instead of emitting
   plausible but incorrect depths.
4. Evaluate per scenario in addition to aggregate IoU and depth MAE. Include temporal stability for video streams.
5. Export the selected model as TorchScript or ONNX for the acquisition-side runtime.

