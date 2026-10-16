# Uneven-surface peak tolerance V10, calibration repair V11 (2026-10-14)

## Scope and scoring

The representative bottom-relative liquid height keeps its tolerance
`max(20 mm, 2% of true area-mean liquid height)`. Only the highest visible
wave/float height now uses `max(50 mm, 5% of true highest height)`. A missing
number fails; MAE and P95 use numeric frames only. The highest height remains
a separate physical output: relaxing its tolerance does not alter a prediction.
No hard zeroing, value clipping or oracle truth enters inference.

This is a historical V10/V11 experiment; the [merged V11 report](uneven_surface_merged_v11_20261015.md)
is the current code and test-data reference. At this experiment's date, the
V10 evidence route was opt-in; it is now enabled by default in the research
`UnevenSurfaceMeter`. The supported ordinary, homogeneous transparent/translucent,
and <=50% partial-loss product routes are unchanged. The matched dry image
is required. Float overlays preserve raw missing returns. Each fresh seed
contains 99 installations, two subscenes (wave only and wave plus float
proxies), and eight repeats per subscene. Repeats from one installation are
correlated; no field-camera accuracy claim is made here.

Development used A-H, including diagnosis of E/H geometry failures and a
near-bottom float regression in G. Seed I was generated after the final
numeric choices and remains the primary untouched holdout. Its observations
and original calibration are unchanged by the V11 calibration repair.

## Final untouched holdout I

| Output and subset | Frames | Baseline pass / MAE / P95 | V10 pass / MAE / P95 |
|---|---:|---:|---:|
| Highest, wet outside deferred far/shallow | 1,344 | 90.77% / 52.74 / 146.70 mm | 99.48% / 21.51 / 60.73 mm |
| Highest, 1-3 m liquid | 288 | 77.78% / 117.07 / 725.45 mm | 97.57% / 22.22 / 69.06 mm |
| Highest, 4-9 m liquid | 720 | 98.75% / 35.69 / 77.43 mm | 100% / 22.46 / 58.62 mm |
| Highest, dry with camera-to-bottom <6 m | 96 | 64.58% / 39.31 / 108.09 mm | 100% / 12.19 / 29.74 mm |
| Mean liquid, wet outside deferred far/shallow | 1,344 | 93.53% / 15.86 / 61.55 mm | 93.53% / 15.75 / 61.55 mm |
| Mean liquid, 1-3 m liquid | 288 | 73.96% / 28.00 / 79.83 mm | 73.96% / 27.45 / 79.83 mm |

Seed I has no installation initialization failure. Both outputs are numeric
on all 1,344 non-far wet frames. Seven highest estimates still fail their
new tolerance, all in structured-light case 32. The mean still fails on
87/1,344 non-far wet frames, including 75/288 in the 1-3 m band.

F and H are supporting development checks, not untouched confirmations.
With H's calibration prior repaired before either chain was evaluated, pooled
F/H/I contains 4,032 non-far wet frames. Baseline highest pass/MAE/P95 are
90.82% / 47.04 / 149.69 mm; V10 gives 99.38% / 20.18 / 57.50 mm.
Mean pass is 94.99% baseline versus 94.89% V10; the peak repair does not
establish a mean-height improvement. V10 now returns both values on all
4,032 non-far wet frames. Two baseline peak outputs are still missing for
algorithmic reasons, not for calibration failure.

## V11 calibration-prior repair and retest

The original H case 92 and E case 95 had two rim corners genuinely outside
the 320x180 dry image. Re-clicking invisible corners would be fabricated
input. The repaired route instead uses 60-63 noisy clicks along four *visible*
rim edge segments, the independently measured two rim dimensions, gravity
and camera-to-bottom distance to fit the rectangle; two corners are then
explicitly marked as extrapolated. Median visible-edge fit residuals are
2.27 mm (H92) and 0.97 mm (E95). E case 84 failed a noisy independent
dimension cross-check; a second simulated operator annotation passed. This
retry uses dry-frame calibration validity only, never wet test error.

The evaluator now preflights all 99 installations before scoring any wet frame.
If any prior still fails, it raises an error and creates no evaluation output.
Both H and E pass 99/99 preflight, with zero installation failures in the
reruns. Baseline and V10 use identical repaired priors and unchanged wet
frames: their installation-audit SHA-256 hashes match exactly within each
seed (H: `c3df61195d6b0e2e`; E: `d273ae7f7253183e`, prefixes shown).
H92's recovered 16 frames have mean-height signed errors from
-3.2 to +0.7 mm. The old H/E artifacts with failed priors are superseded,
not silently mixed into the new summary.

| Repaired development set, non-far wet | Frames | Mean: coverage / pass / MAE | Highest: coverage / pass / MAE |
|---|---:|---:|---:|
| H, V10 | 1,344 | 100% / 95.39% / 13.63 mm | 100% / 99.40% / 19.83 mm |
| E, V10 | 1,344 | 100% / 94.20% / 16.40 mm | 100% / 100% / 21.36 mm |

Pass tolerance is max(20 mm, 2% true mean) for liquid height and
max(50 mm, 5% true highest) for the highest point. E is a development
set and is not added to the F/H/I summary.

Partial-rim extrapolation is an observability-limited research route, not
industrial qualification. In a real installation, the operator must inspect
the visible edge labels and independent dimensions; if adequate visible
edges are unavailable, reposition the camera or use another metrology
method. The reconstruction is never allowed to use test liquid truth.


## Changes and their evidence

1. Empty and near-bottom peaks use the matched dry image, not a forced
   numerical zero. When the mean, paired wet-minus-dry evidence and its
   spatial uncertainty agree near the bottom, the visible peak is the
   90th percentile of supported 9x9 local dry-pair differences. A
   coherent elevated candidate is preserved at close range, protecting
   very shallow liquid with a real floating object. A previous version
   suppressed such objects; G case 9 improves from about 0.019 m to
   about 0.054 m for a true 0.070 m peak. H dry controls improve from
   93.75% to 100% after the final dry-pair refinement.

2. If the old peak is far below the mean and narrow support is absent,
   a wider search uses a radius of 0.25 m plus spatial paired
   uncertainty, capped at an additional 0.12 m. This repairs a
   separate supported return layer without using truth at inference.
   The known D/case 41 error above 1 m is removed on replay.

3. At estimated mean height 1-3 m, a reliable dry-pair offset can
   correct a common measurement bias in the peak. It is shrunk by
   `1/(1+(paired_tile_se/0.030 m)^2)`; it applies only with sufficient
   spatial support and paired uncertainty <=0.040 m. The supported
   local candidate now replaces an isolated raw maximum when their
   difference exceeds 0.020 m. These changes account for the large
   1-3 m peak gain, while the mean height estimator itself stays
   essentially unchanged.

4. `highest_evidence_status` is returned as `supported`, `weak`,
   `missing`, or `unassessed`, with explicit warning reasons.
   Numerical estimates are not rejected. This is an observability
   warning, not a calibrated probability. On untouched I, 1,255
   supported frames have zero peak tolerance failures; 89 weak frames
   include all seven failures. Across development F/H plus I after
   calibration repair, supported has 3 failures in 3,748 frames and
   weak has 22 in 284; no calibration-induced missing frames remain.
   The three unflagged failures show why status is not guaranteed confidence.

## Remaining problems and priority

The main remaining numerical problem is **mean liquid height at 1-3 m**:
on I it passes only 73.96%, and on F/H/I pooled it passes 79.98%.
The peak correction cannot repair a biased area-mean estimate.
Analyse the dry-paired area distribution against the return-quantile
and quantized-disparity layers, then test a separate mean estimator
without coupling it to the highest output.

The residual highest failures are concentrated in a few structured-light
configurations with sparse or conflicting layers (seven I frames in
case 32). The former H geometry initialization failure has been repaired;
it no longer counts as a measurement failure. Partial-rim geometry remains
less observable than a fully visible rim and requires field validation.

The sampled floating objects are only three opaque depth domes.
Real object shape/material, liquid motion, image alignment, and
camera-specific return behavior remain unverified. The cross-scene
far-distance shallow-liquid work is deferred to
[its shared backlog](uneven_remaining_and_cross_scene_far_shallow_20261014.md).
The historical opt-in recommendation is superseded by the merged V11 research
route; mean-height tails, geometry initialization and physical-camera
validation remain unresolved.


## Reproduction and verification

All test artifacts are under
`/root/autodl-tmp/liquid-depth-artifacts/evaluation/`.
The main machine-readable pool is
`uneven-peak50-v11-calfix-fhi-summary-20261014.json`; it names the
exact F/H/I baseline and candidate record directories and rejects any
invalid calibration audit. Recreate the pooled summary with:

```bash
python scripts/summarize_uneven_peak50.py \
  --root /root/autodl-tmp/liquid-depth-artifacts/evaluation \
  --output /root/autodl-tmp/liquid-depth-artifacts/evaluation/uneven-peak50-v11-calfix-fhi-summary-20261014.json
```

The evaluator is `scripts/evaluate_uneven_dual_height.py`. Its current default
(or explicit `--evidence-repair`) selects the merged evidence route; use
`--no-evidence-repair` for the historical paired baseline on the same inputs.
The reference generator is
`scripts/generate_uneven_validation.py`.

Ruff passes for edited code; the full repository suite passes
570 tests (14 upstream Torch deprecation warnings). A separate
320x180 server CPU test over 10 installations and 20 repeats each
gives median of per-installation medians 21.09 ms, worst sampled
per-installation P95 30.41 ms and maximum 33.17 ms. This is
algorithm-only time; capture, transfer, model perception and UI
are excluded from the 0.5 s end-to-end target.

