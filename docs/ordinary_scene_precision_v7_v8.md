# Ordinary-scene liquid-level precision: V7 and V8

## Acceptance target

The engineering-product gate for ordinary scenes is: liquid-level AbsRel <= 1.5%, within `max(5 mm, 2%)` >= 90%, output coverage >= 95%, and evaluable accepted outputs >= 99%. All numbers below are from the independent synthetic test split and are not hardware qualification claims.

## V7 learned calibration and tail-loss ablation

V7 adds an identity-initialized per-frame scale/bias head, camera-range-balanced sampling, direct tolerance-overrun loss, worst-10% CVaR loss, ordinary-sample weighting, and checkpoint selection biased toward ordinary liquid-level tolerance.

The best ordinary validation pass rate was 82.09%, but the independent test split did not improve reliably. Ordinary validation and test each contain only 67 frames, so the per-frame calibration head overfit the validation distribution. A global affine fit also failed to reach 90% on test. V7 is retained as an ablation result and is not the promoted ordinary route.

## V8 robust sensor-anchor route

Error auditing showed that the robust median of valid raw RGB-D returns inside the predicted liquid mask is substantially better than full learned reconstruction in ordinary scenes. V8 retains V6 mask, completion, and uncertainty predictions, while adding an ordinary-only bias anchor:

1. select valid raw-depth points inside the predicted liquid mask;
2. compute robust medians of raw and restored depth over the same support;
3. use their difference as a frame-level bias correction;
4. scale the correction by valid-support coverage, falling back to learned restoration as support disappears;
5. clamp the correction to prevent a large shift from abnormal returns.

Enable this route only when diagnostics classify the scene as ordinary and raw RGB-D returns are reliable. Transparent, glare, multilayer, and large-depth-failure scenes must continue to use their learned repair or explicit rejection routes.

## Independent synthetic test result

| Route | AbsRel | Within `max(5 mm, 2%)` | Coverage | Evaluable acceptance | Ordinary gate |
|---|---:|---:|---:|---:|---|
| V6 learned reconstruction | about 1.47% | about 76.1% | 100% before strict thresholding | 100% | Fail |
| V8 ordinary sensor anchor | 0.506% | 95.38% | 97.01% | 100% | Pass |

Strict V8 report: `/root/autodl-tmp/liquid-depth-artifacts/evaluation/scenario-confidence-v8-ordinary-engineering.json`.

Exported model: `/root/autodl-tmp/liquid-depth-artifacts/models/universal-liquid-v8-ordinary-sensor-anchor.ts`.

## Reproduction

```bash
bash scripts/run_v7_ordinary_precision_pipeline.sh
bash scripts/run_v8_ordinary_sensor_anchor_pipeline.sh
```

The next qualification step is to repeat the anchor audit with the target RGB-D camera, containers, exposure settings, and temperature bands, then lock the ordinary/complex route thresholds. V8 must not be marked hardware-qualified until independent physical tests pass.
