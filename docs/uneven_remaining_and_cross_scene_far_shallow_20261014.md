# Uneven-surface remaining issues and deferred far/shallow work (2026-10-14)

Historical V4 issue inventory: its peak pass percentages use the old
`max(20 mm, 2%)` tolerance. The current highest-point acceptance
`max(50 mm, 5%)` and V10 results are in
[the updated peak report](uneven_peak_50mm_5pct_v10_20261014.md).
The mean-height tolerance remains `max(20 mm, 2%)`.

## Evaluation contract

These are synthetic, bottom-relative liquid heights, not sensor range. The
score is an all-frame pass under absolute error <= max(20 mm, 2% of the true
height); a missing number fails. Mean liquid height and the highest supported
visible height are separate outputs. The latter includes the wave or floating
object. No numerical zero snap or height clipping is applied.

The primary uneven evidence is the final opt-in echo/peak repair on independent
strict holdouts C and D, where float overlays preserve missing raw sensor
returns. Each has 99 installations, two subscenes and eight repeats. Repeated
frames from one installation are correlated. The default production path is
unchanged and this uneven route is research-only. Exact records:
`evaluation/uneven-repair-{c,d}-strict-final-v4-20261014/records.jsonl`
under `/root/autodl-tmp/liquid-depth-artifacts/`.

For this issue inventory, "deferred far/shallow" means camera-to-bottom >=6 m
and 0 < true liquid height <=1 m. This removes 192 wet frames from the 2,880
strict C/D wet frames. Dry controls are shown separately. The older ordinary
experiment used height <=0.5 m including dry; its percentages are not pooled
with the uneven subset.

## Uneven scene after excluding far/shallow

| Subset | Frames | Mean height MAE / P95 / pass | Highest height MAE / P95 / pass |
|---|---:|---:|---:|
| All remaining wet | 2,688 | 14.13 / 54.01 mm / 94.87% | 25.63 / 68.43 mm / 89.55% |
| 0.01-0.1 m | 192 | 4.35 / 12.69 mm / 98.96% | 10.10 / 42.12 mm / 82.29% |
| 0.1-1 m | 192 | 4.96 / 14.26 mm / 97.40% | 12.54 / 33.89 mm / 74.48% |
| 1-2 m | 288 | 15.91 / 59.74 mm / 81.60% | 23.97 / 79.84 mm / 70.14% |
| 2-3 m | 288 | 26.60 / 96.89 mm / 84.03% | 54.75 / 94.55 mm / 69.10% |
| 3-4 m | 288 | 17.17 / 118.03 mm / 93.06% | 27.61 / 81.09 mm / 93.06% |
| 4-9 m | 1,440 | varies by band; each >=95.83% pass | varies by band; each >=98.96% pass |

The repair improves the non-far wet highest output from 48.40 mm MAE,
163.70 mm P95 and 83.30% pass to 25.63 mm, 68.43 mm and 89.55%.
For mean height it improves MAE from 19.89 to 14.13 mm, while pass changes
from 94.75% to 94.87%. Coverage is 100% for both outputs after repair.

## Ranked issues outside far/shallow

1. **Highest-point selection at 1-3 m is the largest measured gap.** Of 576
   wet frames, 175 fail the highest-point tolerance (69.62% pass), while 99
   fail the mean-height tolerance (82.81% pass). These account for 62% of
   remaining peak failures and 72% of remaining mean failures. Across all
   non-far wet frames, peak has 281 failures (39 configurations); mean has
   138 failures (17 configurations). Peak alone fails in 190 frames even
   when the mean is within tolerance, so repairing the mean does not solve
   peak selection. The 1-3 m repair raises peak pass from 59.20% to 69.62%,
   but barely changes mean pass (82.81% both versions).

2. **Wrong echo layer can create a dangerous numerical peak.** There are 17
   peak errors above 100 mm, from four configurations, all structured-light
   simulations. The largest is 1,139 mm: in D/case 41/repeat 6, the true
   mean/peak are 2.423/2.506 m, but the reported mean/peak are 2.593/1.367
   m. The estimator retained a low echo as the peak while another return
   layer moved the mean upward. It emits `height_order_inconsistent`, yet
   still returns a number. Among 281 out-of-tolerance peaks, 132 preserved
   the original peak, 83 selected a supported lower-layer mean, 64
   suppressed an isolated high echo, and two rescued a missing peak.
   These are observed decisions, not a uniquely identified physical cause.

3. **Dry-container peak false positives persist even at short range.** For
   192 dry frames with camera-to-bottom <6 m, mean height passes 100%
   (2.97 mm MAE), but the highest-point output passes only 59.38%
   (30.80 mm MAE, 78 failures). The optional repair lowers peak MAE from
   41.82 mm but leaves the pass fraction unchanged. A dry/not-dry state
   and a distinct highest-object validity decision are needed before
   treating the peak as an operational alarm.

4. **The floating-object response is partly verified.** In 1,344 matched
   wave-only versus wave-plus-float frame pairs outside far/shallow, the
   mean-height estimate changes by 0.70 mm on average (2.38 mm P95),
   which is desirable. The peak changes by 14.50 mm on average
   (45.76 mm P95); 477 pairs change by >20 mm. However these are three
   simulated floating-depth proxies. Tall, irregular, moving, wet or
   transparent objects, and partial occlusion remain unvalidated. The
   fixed candidate band around the estimated mean can miss a genuine
   object when the mean itself is biased.

5. **Confidence and observability still need calibration.** All 2,688
   remaining wet frames produce both numbers, but 281 peaks and 138 means
   are outside tolerance. Only 33 frames carry the height-order flag;
   19 of those have a bad peak. Eight bad peaks carry the sparse-support
   flag. Numeric coverage is therefore not reliability. The final route
   needs a calibrated per-output error risk and a visible status when the
   two height estimates disagree.

The present simulation tests independent frames of static installations.
Temporal wave motion, changing pose, real material response, and the field
camera pipeline have not been validated. The 320x180 algorithm-only median
latency of 24.85 ms does not include camera capture, transfer or perception.


## Cross-scene far/shallow backlog for one later study

| Route or scenario | Existing evidence | Shared or scene-specific suspected source |
|---|---|---|
| Ordinary diffuse | Older <=0.5 m including dry definition: 28 frames, 22.28 mm MAE / 57.14% pass in seed 1; 24 frames, 19.84 mm / 62.50% in seed 2. Only six or seven distinct installations per seed. | Dry wrong echoes can yield false positive empty height up to 129 mm; disparity-bin selection can choose near-zero dry layer for true 28-43 mm liquid. Four ordinary frames per seed also fail during empty-geometry initialization. |
| Homogeneous transparent/translucent | Latest five-prior full frozen result under the current tolerance: one transparent and four translucent failures, all empty or about 7.5 cm. No harmonized camera>=6 m subset score has been published. | Small wet-versus-dry image displacement, common calibration noise, and dependence on visible bottom texture. |
| Partial depth loss <=50% | Latest formal test: 0-0.1 m band passes 88.3% across distances; no harmonized far-only subset score. Examples include 22-134 mm false empty height and an approximately -11 mm selected dry layer when truth is about 28 mm. | Missing surface evidence amplifies wrong-return and near-zero layer confusion. |
| Uneven surface | Strict C/D wet camera>=6 m, 0<h<=1 m: 192 frames, mean-height 66.24 mm MAE and 42.19% pass; active stereo 21.88%, structured light 4.69%, ToF 100% in this simulation. The repair did not establish a reproducible pass gain here. | Remote shallow surface is poorly separated from bottom return; the structured-light paired-evidence uncertainty commonly exceeds its 35 mm gate. |
| Glare, low light, multilayer and compound scenes | No current-chain harmonized far/shallow score. Older scene studies are not directly comparable to the selected formal routes. | Capture quality, mixed interfaces and evidence scarcity require separate scenario labels. No accuracy claim is made from the present records. |

The synthetic sensor reports can be censored above 10.5 m in
`src/liquid_depth/simulation.py`, while the present uneven evaluator
constructs its shared meter with a 10.0 m maximum return. The 10 m
application limit is being mixed with the raw sensor-return cutoff.
This mismatch must be resolved before attributing far-range failures
to hardware. The current 320x180 stereo/structured-light simulation
also uses short generic baselines, not a verified 10 m camera profile.
The unified study should keep raw returns near 10 m, align calibration
and censoring, then compare each scene on the same independent
camera-distance x liquid-height x sensor grid. Report empty
separately; do not pool repeated frames as independent installations.

## Work order after the far/shallow deferral

First, decouple the 1-3 m highest-point candidate from a potentially
biased mean-height layer, using spatial support and consistency with
the dry reference; score mean and peak separately, including >100 mm
tail errors. Second, add an explicit empty-container and peak-validity
decision, calibrated against the 192 near dry controls. Third, expand
float geometry and motion tests, then calibrate separate confidence
for mean and peak. Any promotion from the opt-in uneven route should
require a new independent holdout and no regression in 4-9 m bands.

