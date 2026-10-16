# Consolidated scene simulation report: three supported routes and uneven-surface research (2026-10-14)

## Scope and implementation

The public entry point is `liquid_depth.FormalSceneMeter`, implemented in
`src/liquid_depth/supported_scene_meter.py`. Ordinary diffuse liquid and
partial depth loss share the empty-container geometry, camera calibration,
and container ROI. The ordinary route now uses the same robust return-layer
estimator as partial loss; its scene label and diagnostics remain distinct.
Homogeneous transparent or mildly translucent liquids use one empty image plus
four known-height prior images and rectified stereo.
For structured-light ordinary and partial loss, an explicit camera profile
activates V72 with internal V67/V64 fallbacks. No truth-dependent routing,
zero snap, vessel-height clipping, or numeric-coverage threshold is used.
Special scenes and loss above 50% are outside this supported scorecard.
All routes remain `field_metrology_qualified=false`.

## Paired simulation replay

Measurand: actual liquid height relative to the interior bottom.
Pass tolerance: max(20 mm, 2% of true liquid height).
Seed 202610141101: six configurations x 24 frames per homogeneous condition;
99 RGB-D cases x up to four repeats; native and controlled 10%-50% loss frames.
The truth surface mask is used for offline deletion and scoring only.

| Scene | Frames | Numeric coverage | All-frame pass | MAE | P95 absolute error |
|---|---:|---:|---:|---:|---:|
| Homogeneous transparent | 144 | 100.00% | 100.00% | 4.83 mm | 12.51 mm |
| Mildly translucent | 144 | 100.00% | 100.00% | 5.22 mm | 12.66 mm |
| Ordinary diffuse | 396 | 98.99% | 95.20% | 8.29 mm | 29.22 mm |
| Partial depth loss 0%-50% | 2,322 | 100.00% | 96.99% | 8.49 mm | 28.82 mm |

The ordinary row is the formal shared robust-return route, not the former
broad container median. On a second, newly generated RGB-D seed
(202610141731), ordinary diffuse has 396 frames, 98.99% numeric
coverage, 94.44% all-frame pass, 8.27 mm MAE, and 27.85 mm P95.
The same new-seed partial-loss replay contains 2,332 eligible frames,
100% numeric coverage, 95.28% pass, and 8.46 mm MAE. The transparent
routes were not regenerated on that second seed.
The transfer decision considered both synthetic seeds. Their comparison
is a reproducible robustness check, not a blinded final holdout or
validation of a physical camera.

The 288 homogeneous replay outputs are identical to the former interface
(maximum paired difference: 0 m). The 2,322 partial-loss outputs match the
frozen V72c record frame by frame (maximum paired difference: 0 m).
Full frozen homogeneous tests, which were not regenerated here, report:
transparent 2,672 frames, 100% coverage, 99.96% pass, 5.01 mm MAE,
14.54 mm P95; mildly translucent 5,344 frames, 100% coverage, 99.93% pass,
6.41 mm MAE, 19.46 mm P95. Do not extrapolate the 144-frame 100% results
to these full populations.

## Remaining errors and mechanisms

Ordinary diffuse V3: active stereo 132 frames, 93.94% pass, 8.16 mm MAE;
structured light 132 frames, 96.97% coverage, 91.67% pass, 10.26 mm MAE;
ToF 132 frames, 100% pass, 6.49 mm MAE. The unchanged four missing
ordinary outputs come from structured-light case 86: its empty-geometry
points fall outside the image before the return estimator runs. The new
seed has the same four-frame geometry failure in ToF case 98.

The ordinary 0-0.1 m band passes 86.1% in the primary seed and 87.5%
in the new seed. Yet the far-shallow subset (camera-to-bottom >=6 m,
height <=0.5 m) does not show reproducible tolerance improvement:
28 primary-seed frames improve from 42.86% to 57.14% pass, whereas
24 new-seed frames decline from 66.67% to 62.50% pass. Both subsets
have only six or seven distinct scene configurations; repeat frames
are correlated. The V64 quantile can turn dry wrong echoes into a
positive empty-container height (up to 129 mm in one primary-seed
active-stereo frame). Conversely, V63's 50 mm layer-discovery grid
and far-range disparity noise can select a near-zero dry layer when
true liquid height is roughly 28-43 mm. Empty and very shallow return
distributions overlap, so a single-frame depth-only candidate lacks
decisive evidence. These errors and geometry initialization remain
the principal ordinary-route limitations.

Of 396 ordinary frames, 19 fail in the primary seed and 22 in the new
seed; each set's failures come from only eight configurations. Four
per set are geometry-initialization failures. Beyond the empty/shallow
cases above, structured-light cases 23, 32, and 41 retain roughly
35-100 mm errors at about 0.8-2.3 m liquid height. This pattern is
consistent with residual disparity or wrong-return bias, not proof of
one specific mechanism. Active-stereo case 20 has near-threshold
errors. Per-case counts and predictions are in the V3 frame records.

Homogeneous transparent/translucent: the five-prior model assumes a
single stable liquid, observable bottom texture, rectified stereo,
and a fixed or recalibrated near-top camera. In the full frozen test,
one transparent and four translucent frames fail tolerance, all
at empty or approximately 7.5 cm liquid height. Multilayer glass,
strong scattering, bottom invisibility, and real IR matching bias
are not validated by this scorecard.

Partial depth loss: active stereo 789 frames, 95.06% pass, 8.54 mm MAE;
structured light 741 frames, 95.82% pass, 10.50 mm MAE;
ToF 792 frames, 100% pass, 6.56 mm MAE. Seventy frames fail tolerance:
39 active stereo and 31 structured light. For 0-0.1 m liquid height,
pass is 88.3%; for 40%-50% missing returns, pass is 96.2%.
Active-stereo case 2 reports false positive empty height of 22-134 mm
from wrong returns. Structured-light case 14 votes for a dry layer
near -11 mm when the true height is about 28 mm. Cases 23 and 32
retain residual far-range disparity errors after V72. Case 86 has
unusable empty geometry and is excluded from partial-loss scoring;
its four ordinary frames remain failures in the ordinary denominator.

## Uneven surface and floating-object dual-height route (research)

Historical V45/V10 measurements follow in this section. They are retained for
method chronology, not as the current uneven-scene score. The merged V11 route
and corrected E/F/H/I test data are in
[the current uneven-surface report](uneven_surface_merged_v11_20261015.md).

This section extends the consolidated report; it is not a fourth
field-qualified or supported `FormalSceneMeter` scene. The default
`UnevenSurfaceMeter` route now uses the matched dry-frame shallow correction
and evidence repair whenever an aligned dry image is provided. The old
robust-return-only result is retained only as a reproducible ablation
via `--no-dry-paired-shallow`. The representative liquid height is
bottom-relative area-mean height; the separate peak is the highest
supported visible wave/float return. No zero snap or height clipping.

Matched V45 independent simulation: 99 installations, two subscenes
(waves only; waves plus three floating-depth proxies), four frames each.
All 792 frames are scored; each of 11 nominal liquid-depth bands has
72 frames. Camera-to-bottom distance is at most 10 m. Passing tolerance
is max(20 mm, 2% of true bottom-relative height), identical to the
three supported-scene scorecard. P95 and MAE exclude missing outputs;
all-frame pass counts them as failures. Wet and dry images share pose
and vessel geometry. An earlier V45 run used a mismatched V34 dry
reference and is invalid; it is explicitly superseded.

| V45 group / output | Frames | Numeric coverage | All-frame pass | MAE | P95 absolute error |
|---|---:|---:|---:|---:|---:|
| All frames / liquid mean | 792 | 100.00% | 93.81% | 19.94 mm | 48.11 mm |
| All frames / visible peak | 792 | 97.22% | 82.32% | 40.21 mm | 105.77 mm |
| Wet waves / liquid mean | 360 | 100.00% | 93.61% | 28.65 mm | 55.43 mm |
| Wet waves / visible peak | 360 | 97.78% | 79.17% | 60.50 mm | 288.65 mm |
| Wet waves + floats / liquid mean | 360 | 100.00% | 96.11% | 13.08 mm | 39.22 mm |
| Wet waves + floats / visible peak | 360 | 100.00% | 90.83% | 21.03 mm | 54.20 mm |
| Dry control / liquid mean | 72 | 100.00% | 83.33% | 10.70 mm | 43.75 mm |

With the matched empty reference, the prior uneven route scored
91.04% liquid pass, 23.24 mm MAE, and 67.79 mm P95 on these same
792 frames. The selected route scores 93.81%, 19.94 mm, and 48.11 mm.
The peak output is identical frame by frame because this change only
corrects representative shallow liquid height.

### Liquid-height stratification on V45

| True height band | Frames | Numeric coverage | Liquid pass | Liquid MAE | P95 |
|---|---:|---:|---:|---:|---:|
| Dry, 0 m | 72 | 100% | 83.3% | 10.7 mm | 43.8 mm |
| 0.01-0.1 m | 72 | 100% | 90.3% | 11.5 mm | 54.9 mm |
| 0.1-1 m | 72 | 100% | 84.7% | 14.0 mm | 71.2 mm |
| 1-2 m | 72 | 100% | 84.7% | 15.1 mm | 54.2 mm |
| 2-3 m | 72 | 100% | 93.1% | 15.8 mm | 54.6 mm |
| 3-4 m | 72 | 100% | 97.2% | 17.0 mm | 41.5 mm |
| 4-5 m | 72 | 100% | 100.0% | 12.5 mm | 43.9 mm |
| 5-6 m | 72 | 100% | 98.6% | 88.7 mm | 74.5 mm |
| 6-7 m | 72 | 100% | 100.0% | 11.8 mm | 29.7 mm |
| 7-8 m | 72 | 100% | 100.0% | 11.3 mm | 27.6 mm |
| 8-9 m | 72 | 100% | 100.0% | 11.0 mm | 15.9 mm |

The dry, 0.01-0.1 m, and 0.1-1 m bands improve from the matched
baseline's 77.8%, 68.1%, and 81.9% pass to 83.3%, 90.3%, and
84.7%. All 576 frames at 1-9 m are unchanged. The 5-6 m MAE
is dominated by one structured-light wrong-layer output: true
5.141 m versus predicted 0.018 m (case 68, frame 2). It remains
visible rather than being clipped or removed; P95 alone misses it.

Across all 264 frames per simulated sensor, liquid MAE/pass is
14.82 mm/90.91% for active stereo, 38.16 mm/90.53% for
structured light, and 6.85 mm/100% for ToF. The structured-light
mean is strongly affected by the single wrong-layer case above.

### Remaining errors, coverage, and applicability

The dry-pair correction was applied to 202/792 frames. It requires
bottom-return support and preliminary/candidate agreement within
0.20 m, without reading the true height. Some true shallow frames
remain outside its entry range when the preliminary estimate is
just above 1 m. An empty active-stereo frame retains a 111 mm
false height, and a 19 mm-true shallow frame is estimated at
233 mm because the contradictory paired candidate is rejected.
Eleven liquid outputs are negative and remain visible as-is.

The peak is unavailable in 22/792 frames, and 49 frames flag a
peak below the representative liquid estimate. There are 118
numeric peak outputs beyond tolerance, plus 22 missing peak
outputs. Sparse apex returns and wrong-return clusters remain
unresolved; the peak should not yet be used as a safety clearance.

The synthetic float depth was injected geometrically without RGB
rerendering. The four frames in each installation are correlated;
this is an independent seed, not 792 independent field trials.
The correction needs stable dry/wet registration and a matched
empty-vessel image. Camera movement requires re-registration or
recalibration. No physical camera or industrial liquid validation
has been completed; `field_metrology_qualified` remains false.

Implementation: `src/liquid_depth/uneven_surface_meter.py` and
`scripts/evaluate_uneven_dual_height.py` (dry pairing is default).
Selected V45 records:
`/root/autodl-tmp/liquid-depth-artifacts/evaluation/uneven-dual-current-v45-20261014/`.
Matched prior-route comparator: `/root/autodl-tmp/liquid-depth-artifacts/
evaluation/uneven-dual-matched-baseline-20261014/`.
Development-seed comparison and failure details:
`docs/uneven_surface_shallow_dry_paired_20261014.md`.


### Follow-up echo-layer and peak repair trial (optional, 2026-10-14)

The numbers in this subsection preserve the historical 20 mm / 2% peak
criterion. Current highest-point acceptance is 50 mm / 5% while mean
height remains 20 mm / 2%; see
[the V10 uneven peak report](uneven_peak_50mm_5pct_v10_20261014.md).

The then-optional evidence repair was evaluated separately from the default
route above. It vetoes contradicted near-zero layers, uses spatial dry-pair
evidence for shallow disagreements, and repairs peaks with supported local
means. Sparse wider-neighborhood values carry an observability flag.
Four new seeds were generated; final C/D comparisons preserve missing
sensor returns in float regions instead of filling them from ideal depth.
These strict comparisons include 2,880 wet frames from 180 installations
plus separately evaluated dry controls. Frame repeats remain correlated.

| Strict C/D output | Baseline coverage / pass | Repair coverage / pass | Baseline MAE / P95 | Repair MAE / P95 |
|---|---:|---:|---:|---:|
| Liquid mean | 100% / 91.25% | 100% / 91.35% | 22.95 / 75.98 mm | 17.60 / 75.94 mm |
| Visible peak | 98.26% / 79.79% | 100% / 86.39% | 53.06 / 206.96 mm | 27.83 / 78.26 mm |

Maximum liquid error changes from 5,372 mm to 233 mm. The diagnosed
V45 5.141 m case changes from 0.018 m to 5.202 m (60.8 mm error).
However, strict C/D far-shallow pass remains 42.19%; no consistent
accuracy gain is established for that subset. One 1,139 mm peak error
remains in D and is flagged. Empty-control peak accuracy can worsen.
At the time, `--evidence-repair` stayed opt-in; the merged V11 research route
now enables it by default after the later E/F/H/I comparison.

CPU algorithm-only timing at 320x180: median per-case 24.85 ms, worst
sampled P95 32.87 ms; all 564 tests pass. Capture and transport are excluded.
Detailed plan, per-depth metrics, failure analysis and reproduction:
[echo-layer and peak repair trial](uneven_surface_echo_peak_repair_20261014.md).
Machine summary: evaluation/uneven-repair-20261014/comparison-final-v4.json
under /root/autodl-tmp/liquid-depth-artifacts.

The current uneven-scene issue inventory, excluding deferred far/shallow wet
frames, and the cross-scene far/shallow backlog are in
[remaining issues and deferred work](uneven_remaining_and_cross_scene_far_shallow_20261014.md).

## Subsequent special-scene work

Preserve special-scene research code and raw datasets separately from
the supported three-scene interface. Low light, glare, standalone
floating-object classification, and multilayer transparency remain
outside this scorecard. Evaluate each route by depth, error tail,
numeric coverage, and latency on independent cases before promotion.
Loss above 50% remains outside the promised adaptation range.

Reproducer: `scripts/evaluate_supported_three_scenes.py`.
Formal V3 evaluation directory:
`evaluation/supported-three-scenes-v3-independent-20261014/`.
New-seed ordinary/partial evaluation directory:
`evaluation/supported-three-scenes-v3-newholdout-20261014/`.
Both are under `/root/autodl-tmp/liquid-depth-artifacts/`.
V2 records remain as a historical paired comparator.
Full homogeneous frozen source:
`evaluation/transparent-sr-v33-20261012/test-raw.json`.

## Consolidation and cleanup

The V1 dispatcher/evaluator/report and V52-V67 intermediate partial-loss
evaluation directories, evaluators, and superseded reports were removed.
The product panel runtime was not changed by this consolidation. The
uneven-surface research route was updated separately as documented above.
Retained evidence includes the full frozen homogeneous source, the four
V72c comparative records, and independent three-scene V3 records.
V56/V63/V67 algorithm modules remain as internal fallbacks, not separate
product entry points. Removed simulation records can be regenerated from raw
datasets; raw datasets were not deleted.

Follow-up: the ordinary far-shallow transfer ablation is reported in
`docs/ordinary_far_shallow_partial_method_ablation_20261014.md`.
The shared estimator is now the supported ordinary route; its
far-shallow regression and false-empty-return causes remain documented.
