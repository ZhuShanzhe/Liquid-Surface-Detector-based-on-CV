# V72 far-shallow structured-light quantization audit (2026-10-14)

Status: opt-in simulation research route; not field-metrology qualified.

## Failure mode and correction

- At long range, adjacent 1/16-pixel disparity bins can correspond to hundreds of millimeters of axial depth. A local height window may choose one bin as a false surface.
- V71 checks for two well-supported adjacent quantized bins. If the likelihood fit is invalid, it preserves the V64 numeric estimate and records an ambiguity flag; it does not force zero or clip to the vessel height.
- V72 fits a continuous liquid-height parameter from all quantized container returns using a Gaussian bin-integrated disparity likelihood conditioned on the SDK maximum-return cutoff. No truth depth, oracle mask, RGB label, or future frame enters inference.
- The correction requires a measured camera profile: disparity step and pre-quantization disparity noise. The current evaluation uses the simulation profile (1/16 px, 0.26 px). Do not assume these values for a real camera.
- Applicability: partial-depth-loss route, structured light, 8 m or farther camera-to-bottom distance, preliminary height between 0.05 m and 25% of that distance, no dry-bottom vote, and sufficient adjacent-bin support. Outside this scope the existing route remains unchanged.

## Evaluation protocol

Actual liquid height above the interior bottom is the measurand. Pass tolerance is max(20 mm, 2% of true height). All eligible native and controlled-loss frames are counted; no output-coverage gate, zero snap, or vessel-height clipping. V64 and V67 run on the identical frames.

| Dataset | Eligible frames | Excluded | V64 MAE / pass | V67 MAE / pass | V72 MAE / pass |
|---|---:|---:|---:|---:|---:|
| V67 prior holdout | 4,687 | 0 | 16.3 mm / 95.5% | 12.5 mm / 95.5% | 12.3 mm / 95.8% |
| V69 pose holdout | 3,001 | 36 | 14.1 mm / 93.8% | 12.8 mm / 93.9% | 9.7 mm / 95.0% |
| V70 geometry shift | 3,007 | 36 | 15.4 mm / 92.4% | 13.7 mm / 92.4% | 11.3 mm / 93.8% |
| New independent seed 202610141101 | 2,322 | 1 | 12.8 mm / 95.8% | 10.5 mm / 95.8% | 8.5 mm / 97.0% |

Numeric coverage was 100% in every eligible set. V69/V70 excluded 36 geometry-incompatible frames each; the independent set excluded one native-loss-above-50% frame. These exclusions must not be treated as passes.

## Limitations

- The synthetic sensor model and the correction share the same disparity-noise family; an independent seed is not an independent physical sensor validation.
- V67 case 23 (true height about 0.866 m) worsened from 35.6 to 46.7 mm MAE, despite aggregate improvement. The near-empty V69 case 14 remains about 66 mm MAE. These are not solved by quantization correction alone.
- No temporal fusion is active. Real-device disparity histogram, profile calibration, and repeatability are required before enabling this route by default.

The current consolidated reproducer is `scripts/evaluate_supported_three_scenes.py`.
The final V72c development records remain under `/root/autodl-tmp/liquid-depth-artifacts/evaluation/v72c-*`.
