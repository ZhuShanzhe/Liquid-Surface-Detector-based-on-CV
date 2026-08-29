# RGB-D camera five-distance plane qualification

This test identifies scale and offset for one physical camera and one depth mode.
It qualifies diffuse-plane depth only; it is not an end-to-end transparent-liquid
accuracy certificate.

## Fixed protocol

- Reference distances: 0.3, 1, 3, 5, and 8 m.
- Capture 60 frames at each distance. The first 30 fit the correction and the
  untouched last 30 validate it.
- Analyze the center 50% ROI and reject a frame below 80% valid depth.
- Tolerance is the greater of 3 mm and 1% of the reference distance.
- Report per-frame and five-frame median results. At 30 FPS the median window is
  about 0.17 s.
- Out-of-specification distances are stress tests and are excluded from fitting.
- Deploy only when a real-camera report says qualification_gate.deployable=true.

Measure from the camera optical center to the target. Reference uncertainty should
be below one quarter of the applicable tolerance. Repeat after changing the camera,
lens, resolution, exposure, depth mode, or installation.

## Simulation result

On 2026-08-29 the protocol was run on 100 independently sampled virtual units for
each conservative market error envelope.

| Envelope | Raw median MAE | Calibrated five-frame median MAE / P90 | AbsRel median / P90 | Pass median / P10 | Sites with pass rate at least 90% |
|---|---:|---:|---:|---:|---:|
| 2% stereo | 37.51 mm | 16.48 / 20.79 mm | 0.556% / 0.739% | 83.3% / 73.3% | 35% |
| Long-baseline stereo | 18.12 mm | 6.69 / 8.46 mm | 0.247% / 0.332% | 100% / 93.3% | 97% |
| Typical ToF | 14.50 mm | 7.48 / 9.46 mm | 0.370% / 0.505% | 94.4% / 82.8% | 52% |

The ToF aggregate includes only 1, 3, and 5 m. Its 0.3 and 8 m stress points are
outside the modeled range. A mean AbsRel below 1% does not by itself qualify a
device; the close-range absolute floor and per-point pass rate remain gates.

Reports are stored on the server under:

    /root/autodl-tmp/liquid-depth-artifacts/evaluation/camera-plane-qualification/

market_profiles_100_sites.json contains the 100-unit aggregate. The three other
JSON files contain auditable per-distance representative-unit results.

## Physical capture

Start the camera driver and frame saver:

    ros2 launch orbbec_camera gemini2.launch.py
    ros2 run liquid_depth_camera capture_node --ros-args -p output_dir:=/data/camera_qualification/incoming

Run the interactive collector:

    python scripts/capture_camera_plane_qualification.py --incoming /data/camera_qualification/incoming --output /data/camera_qualification/run_001 --frames-per-distance 60

The collector creates directories such as:

    run_001/0.3m/FRAME_ID/depth.npy
    run_001/1m/FRAME_ID/depth.npy
    run_001/3m/FRAME_ID/depth.npy
    run_001/5m/FRAME_ID/depth.npy
    run_001/8m/FRAME_ID/depth.npy

For a camera whose raw unit is one millimeter, evaluate with:

    python scripts/evaluate_camera_plane_qualification.py --capture-root /data/camera_qualification/run_001 --depth-scale-to-m 0.001 --calibration-frames 30 --validation-window-frames 5 --output artifacts/camera_qualification_real.json

The report includes validity, robust spatial sigma, MAE, AbsRel, pass rate,
scale/offset, qualification_gate, and camera_depth_correction.

A simulated correction has status=simulation_only and must never be deployed. If
a real run fails, inspect the depth mode, exposure, target, operating range, and
reference measurement instead of forcing the correction.

## Runtime integration

The runtime accepts this optional profile section:

    camera:
      depth_correction:
        scale: 1.0
        offset_m: 0.0
        status: not_verified

Correction applies only to valid pixels and cannot turn holes into fabricated
depth. It improves the metric depth input seen by the model. Final liquid depth
still requires surface perception, container geometry, site liquid-level
calibration, temporal filtering, rejection, and a small real-liquid acceptance
set.
