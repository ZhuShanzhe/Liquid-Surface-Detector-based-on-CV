# Ordinary far-shallow transfer ablation (2026-10-14)

## Question and method

Can the partial-depth-loss estimator improve the ordinary diffuse route,
especially at long camera distance and shallow liquid height? Both estimators
were run on the identical native RGB-D frames with the same installed dry
container geometry. The candidate was the existing V64 dry-layer quantile
route, plus V67 elastic evidence and V72 quantized-disparity correction
where applicable. It did not use the truth height, surface mask, zero snap,
or vessel-height clipping during inference. A measured structured-light
camera profile would be needed before using V72 on hardware; the test used
the existing synthetic 10 m / 1/16 px / 0.26 px profile.

The score is actual liquid height above the interior bottom. Pass means
absolute error no greater than max(20 mm, 2% of true height). Missing
outputs count as failures. "Far shallow" means camera-to-bottom distance
at least 6 m and true height at most 0.5 m, used only for offline scoring,
never to select the estimator.

## Paired results

| Set and subset | Frames | Ordinary MAE / pass | Partial-method MAE / pass |
|---|---:|---:|---:|
| Prior independent seed, all ordinary | 396 | 27.83 mm / 85.61% | 8.29 mm / 95.20% |
| Prior seed, far shallow | 28 | 95.65 mm / 42.86% | 22.28 mm / 57.14% |
| New seed 202610141731, all ordinary | 396 | 23.34 mm / 83.08% | 8.27 mm / 94.44% |
| New seed, far shallow | 24 | 25.93 mm / 66.67% | 19.84 mm / 62.50% |

Numeric coverage stayed at 392/396 in both routes on both sets. Four
structured-light frames per set still lacked usable empty-container
geometry; changing the return estimator cannot repair initialization.

On the new seed, the 0-0.1 m band (72 frames) improved from 12.3 mm /
77.8% to 8.1 mm / 87.5%, and the 0.1-0.5 m band (20 frames) improved
from 14.1 mm / 65.0% to 2.8 mm / 100%. At 6-10 m liquid height the
pass rate stayed 96.3%. However, the long-distance shallow subset is
small and clustered by only six configurations, so its 24 frames are
not 24 independent installations.

## Why the remaining far-shallow frames fail

In the new seed, active-stereo case 2 is truly empty. Its ordinary
estimates are about 6-7 mm, while the transferred quantile yields
9-35 mm; two frames now exceed the 20 mm tolerance. In the earlier
seed the same dry case produced transferred estimates up to 129 mm.
Case 11 has about 70 mm of liquid: the broad ordinary median stays near
zero because bottom/wrong returns dominate, while the transferred
estimate reaches roughly 89-110 mm. This is a large improvement in
absolute error, but several frames still miss the 20 mm tolerance.
Structured-light case 14 has roughly 43 mm of liquid in the new seed;
the dry-layer vote selects about -10 mm and remains a failure. Its
50 mm peak-discovery grid and the sensor's far-range disparity noise
cannot reliably separate zero from a few centimeters.

These are not simply missing depth points. The dry and wet return
distributions overlap and include broad wrong-echo components. A
single-frame depth-only rule cannot infer which near-zero layer is
the true surface in every configuration.

## Decision

Product-route decision (2026-10-14): promote the shared robust-return
estimator to the supported ordinary route after explicit project direction.
Its large aggregate gain on both seeds is the reason for promotion; this
does not claim the far-shallow tolerance target is solved. The formal route
is implemented in src/liquid_depth/supported_scene_meter.py and the current
scorecard is docs/supported_three_scene_summary_20261014.md.

The remaining research priority is a truth-free dry-reference likelihood
and 2-4 stationary frames, with an explicit ambiguity flag when empty and
very shallow liquid remain indistinguishable. Validate on new containers
and poses before claiming field metrology qualification.

Records: /root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-partial-method-ablation-20261014/
and /root/autodl-tmp/liquid-depth-artifacts/evaluation/ordinary-partial-method-newholdout-20261014/.
Raw dry/wet synthetic scenes for the new seed remain under
/root/autodl-tmp/liquid-depth-data/synthetic/uniform-{native,empty}-ordinary-transfer-20261014/.
This is simulation evidence, not field metrology qualification.
