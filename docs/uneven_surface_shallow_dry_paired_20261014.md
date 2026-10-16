# Uneven-surface shallow liquid level: matched empty-frame correction (2026-10-14)

## Scope and measurement

- Research route for opaque diffuse liquid with waves, with or without floating objects.
- Reported liquid height is bottom-relative area-mean interface height.
- Reported highest height is a separate visible-return envelope; this change does not modify it.
- A matched dry depth image, fixed camera pose/intrinsics, and an initialized tank ROI are required.
- No true height, surface mask, forced zero, or output clipping is used at inference.

## Method

The original robust-return liquid estimate is retained as a preliminary value.
For preliminary values in [-0.1, 1.0] m, pair valid pixels of the wet frame
with the aligned dry depth frame. Keep only pixels whose dry world height is
within 30 mm of the calibrated bottom, and require at least 24 pairs.
Use the median of wet-minus-dry world-height differences as the shallow
candidate. Adopt it only when it is in [-0.1, 1.0] m and differs from the
preliminary estimate by at most 0.20 m; otherwise retain the original value.
The threshold is a routing consistency check, not a clamp on the output.
Each frame records support count, candidate, application decision, and reason.

## Evaluation protocol

Matched empty frames were regenerated for the independent V45 wet scenes.
The older V45 result had used a V34 empty frame with different pose/geometry;
its results are invalid for a dry-reference comparison and are superseded.
V34 is a development set, V45 an independent synthetic seed. Each has
99 top-mounted installations x 2 subscenes x 4 frames = 792 frames.
Subscenes are waves only and waves plus three floating-object depth proxies.
Per-set dry, 0.01-0.1 m, and 0.1-1 m bands each contain 72 frames.
Three simulated RGB-D families are represented. Float RGB was not rerendered.
All-frame pass uses max(20 mm, 2% of true bottom-relative liquid height);
missing numeric outputs count as failures. MAE uses numeric outputs only.

## Results: liquid area-mean height

| Set and band | Baseline MAE | Paired MAE | Baseline pass | Paired pass | Numeric coverage |
|---|---:|---:|---:|---:|---:|
| V34 dry | 16.95 mm | 4.19 mm | 77.8% | 100.0% | 100% |
| V34 0.01-0.1 m | 27.94 mm | 14.59 mm | 69.4% | 83.3% | 100% |
| V34 0.1-1 m | 38.02 mm | 23.98 mm | 72.2% | 84.7% | 100% |
| V34 pooled 0.01-1 m | 32.98 mm | 19.28 mm | 70.8% | 84.0% | 100% |
| V45 dry | 25.18 mm | 10.70 mm | 77.8% | 83.3% | 100% |
| V45 0.01-0.1 m | 31.10 mm | 11.55 mm | 68.1% | 90.3% | 100% |
| V45 0.1-1 m | 16.15 mm | 13.96 mm | 81.9% | 84.7% | 100% |
| V45 pooled 0.01-1 m | 23.63 mm | 12.75 mm | 75.0% | 87.5% | 100% |

For both seeds, all 576 frames at 1-9 m are unchanged by the route.
The separate highest-visible-height output is also unchanged in every frame.
The V45 1-9 m baseline has MAE 22.89 mm and all-frame pass 96.7%.
The V45 peak estimates still miss some wave/float apices, especially when
returns are sparse; this method does not solve the peak-height task.

In V45 shallow frames, waves-only MAE/pass changes from 25.2 mm/76.4%
to 15.7 mm/83.3%; waves-plus-floats changes from 22.1 mm/73.6%
to 9.8 mm/91.7%. By simulated sensor (48 shallow frames each),
active stereo changes from 46.8 mm/62.5% to 19.8 mm/81.2%; structured
light from 20.0 mm/62.5% to 15.2 mm/81.2%; ToF from 4.0 mm/100%
to 3.3 mm/100%. These sensor slices are small and simulation-specific.

The largest remaining V45 dry error is 111 mm (active stereo case 2).
One 19 mm-true shallow frame still estimates 233 mm because the paired
candidate disagrees with the preliminary estimate and is rejected.
Several true 0.96 m frames estimate just above 1 m, outside the
paired-route entry gate. None of these values is clipped or hidden.

## Ablations and remaining failure modes

- A fixed depth quantile was unstable across dry and wet shallow frames.
- Unrestricted temporal median improved pass rate but caused a 1-2 m regression;
  it is not enabled in this selected route.
- A 0.25 m agreement gate admitted one severe shallow correction error.
  The selected 0.20 m gate trades a small amount of average improvement
  for fewer large per-frame regressions on the development set.
- V45 dry pass remains 83.3%; isolated wrong returns and bottom noise persist.
- V45 0.1-1 m pass remains 84.7%; small waves and sparse bottom support remain.
- Same-pixel dry/wet comparison requires stable registration. Camera movement,
  refraction, changing tank geometry, or non-diffuse return layers require
  separate routes or a new aligned dry reference.
- Synthetic depth-proxy floats and four correlated repeats per installation
  do not establish field metrology accuracy. Test physical RGB-D scenes next.

## Reproducibility

Code: src/liquid_depth/uneven_surface_meter.py
Evaluator: scripts/evaluate_uneven_dual_height.py (dry pairing is default)
Comparator switch: --no-dry-paired-shallow
Baseline: /root/autodl-tmp/liquid-depth-artifacts/evaluation/
  uneven-dual-v34-matched-development-20261014 and uneven-dual-matched-baseline-20261014
Selected outputs: /root/autodl-tmp/liquid-depth-data/evaluation/
  uneven-dual-dry-paired-v2-v34-20261014 and uneven-dual-dry-paired-v2-v45-20261014
Default-route V45 replay: /root/autodl-tmp/liquid-depth-artifacts/evaluation/
  uneven-dual-current-v45-20261014 (record bytes identical to selected V45)
