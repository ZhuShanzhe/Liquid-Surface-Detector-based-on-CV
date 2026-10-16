# Uneven liquid surface: echo-layer and peak repair trial (2026-10-14)

## Status and solution

The repair is an optional research route:
`meter.measure(depth, evidence_repair_enabled=True)`.
The existing default remains unchanged because empty-control peak error
can worsen and far-shallow pass remains weak. No true height or true
surface mask enters inference; no value is zero-snapped or clipped.

1. Veto contradictory bottom-layer selections across the full range.
   Use pixel-paired dry/wet displacement and the observed return quantile.
   Require >=4 spatial tiles, agreement within max(0.25 m, 5% candidate),
   and paired candidate minus three empirical tile standard errors above
   max(0.25 m, 15% of camera-to-bottom height).
2. Resolve the old fixed shallow agreement boundary using spatial support.
   Allow <=0.35 m disagreement only with tile standard error <=35 mm.
   This is a consistency indicator, not a calibrated confidence interval.
3. Select peak evidence through the calibrated opening near the observed
   liquid level. A supported 9x9 local mean repairs missing peaks, wrong
   low layers, and isolated high echoes. Keep an agreeing original peak.
4. If local evidence is sparse, expand to 15x15 with measured support.
   Attach highest_sparse_support_low_observability; never manufacture a
   peak merely by assigning the representative liquid height.
5. Preserve numerical values and explicit layer/height contradiction flags.

## Data and protocol

Four new seeds contain 99 installations each, two subscenes and eight
noise realizations per installation: 1,584 frames per seed including
144 dry controls. A/B were exploratory independent-seed comparisons.
C used the revised layer guard; D was generated after the final sparse
fallback was frozen. The old V45 seed is a diagnosed comparator.

The legacy float proxy fills holes using ideal target depth. C/D strict
tests use --float-preserve-missing: missing sensor returns remain missing.
Both compared routes use the same input. Float RGB is not rerendered;
wave geometry is static across repeats. This is not real-camera validation.

Tolerance: max(20 mm, 2% true bottom-relative height). Missing outputs
fail all-frame pass; MAE/P95 use numeric outputs. Wet results below exclude
dry controls. The repeated frames and subscenes are correlated.

## Final paired results

| Dataset / output | Frames | Baseline MAE | Repair MAE | Baseline P95 | Repair P95 | Baseline pass | Repair pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| V45 liquid mean | 720 | 20.87 mm | 13.52 mm | 47.18 mm | 43.87 mm | 94.86% | 95.28% |
| V45 visible peak | 720 | 40.54 mm | 21.65 mm | 93.28 mm | 65.02 mm | 85.00% | 88.61% |
| New A/B legacy liquid | 2,880 | 15.72 mm | 15.33 mm | 54.39 mm | 53.73 mm | 92.67% | 92.74% |
| New A/B legacy peak | 2,880 | 36.54 mm | 21.20 mm | 93.13 mm | 66.99 mm | 84.93% | 88.99% |
| New C/D strict liquid | 2,880 | 22.95 mm | 17.60 mm | 75.98 mm | 75.94 mm | 91.25% | 91.35% |
| New C/D strict peak | 2,880 | 53.06 mm | 27.83 mm | 206.96 mm | 78.26 mm | 79.79% | 86.39% |

Strict C/D liquid coverage stays 100%; peak coverage rises 98.26% to 100%.
Maximum liquid error falls 5,372 mm to 233 mm. Maximum peak error remains
1,139 mm in seed D and is marked height_order_inconsistent.
V45's 5.141 m layer case changes from 0.018 m to 5.202 m: error falls
from 5,123 mm to 60.8 mm. Its 5-6 m band MAE improves 88.7 mm to 18.4 mm.

## Selected repair by liquid depth: strict C/D

Each row has 288 frames from 18 installation configurations.

| Actual liquid band | Liquid MAE / P95 | Liquid AbsRel | Liquid pass | Peak MAE / P95 | Peak pass |
|---|---:|---:|---:|---:|---:|
| 0.01-0.1 m | 16.00 / 76.73 mm | 23.73% | 81.94% | 20.90 / 85.85 mm | 68.06% |
| 0.1-1 m | 34.37 / 225.94 mm | 9.50% | 77.08% | 33.28 / 191.02 mm | 64.58% |
| 1-2 m | 15.91 / 62.24 mm | 1.16% | 81.60% | 23.97 / 81.25 mm | 70.14% |
| 2-3 m | 26.60 / 96.89 mm | 1.11% | 84.03% | 54.75 / 94.59 mm | 69.10% |
| 3-4 m | 17.17 / 118.12 mm | 0.51% | 93.06% | 27.61 / 81.79 mm | 93.06% |
| 4-5 m | 17.33 / 75.22 mm | 0.39% | 95.83% | 28.52 / 68.31 mm | 98.96% |
| 5-6 m | 19.01 / 60.20 mm | 0.35% | 100.00% | 25.95 / 67.81 mm | 100.00% |
| 6-7 m | 12.12 / 34.48 mm | 0.19% | 100.00% | 20.67 / 49.55 mm | 100.00% |
| 7-8 m | 8.35 / 22.35 mm | 0.11% | 100.00% | 20.99 / 55.39 mm | 100.00% |
| 8-9 m | 9.17 / 19.74 mm | 0.11% | 100.00% | 21.63 / 59.94 mm | 100.00% |

## Remaining problems and next priorities

- Far-shallow improvement is not established. Strict C/D pass remains
  42.19% across 192 frames; MAE changes 65.88 mm to 66.24 mm.
  Millimeter-scale shifts should not be overinterpreted.
- In D's worst peak case, true peak is 2.506 m and output 1.367 m.
  The representative liquid estimate is biased high at 2.593 m, leaving
  no usable return in the peak evidence band. Peak estimation must gain
  independent layer evidence rather than inherit a biased mean.
- Empty-control peak recovery increases numerical coverage without
  consistently improving correctness. In C, pass changes 55.56% to
  54.17% and MAE 35.22 mm to 40.22 mm; dry liquid results are unchanged.
- The peak gate is tested for 30-60 mm float proxies and sampled waves.
  Taller floating objects may be suppressed by this evidence band.
  Their height range needs explicit component-based validation.
- Highest supported visible height cannot guarantee an unobserved apex.

Next priorities: independent peak-layer selection, an area height field
plus floating-object components, and far-range disparity calibration with
a small number of known-height priors. Calibrate uncertainty against these
failures; keep numerical estimates and reason flags visible during research.

## Runtime, tests, and reproducibility

On server CPU at 320x180, median per-case algorithm time is 24.85 ms,
worst sampled P95 32.87 ms, maximum 39.14 ms. Capture, transport, image
loading and GPU perception are excluded. All 564 tests and targeted lint pass.

Code: src/liquid_depth/uneven_surface_repair.py and uneven_surface_meter.py.
Evaluator: scripts/evaluate_uneven_dual_height.py --evidence-repair.
Strict simulation option: --float-preserve-missing.
Generator: scripts/generate_uneven_validation.py.
Summary: scripts/summarize_uneven_repair.py.
Evidence root: /root/autodl-tmp/liquid-depth-artifacts/evaluation/.
Manifest: uneven-repair-20261014/comparison-final-v4.json.
Timing: uneven-repair-20261014/latency-final-v4.json.
Frame records: uneven-repair-*-final-v4-20261014.
Raw synthetic data and paired historical comparators are retained.
