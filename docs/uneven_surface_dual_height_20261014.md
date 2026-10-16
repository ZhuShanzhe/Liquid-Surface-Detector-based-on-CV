# Uneven Surface: Dual-Height Simulation, 2026-10-14

> SUPERSEDED: The V45 figures below paired wet images with an empty image
> from V34 at a different pose/geometry. They are invalid for calibrated
> dry-reference evaluation. Use the corrected matched-empty baseline and
> shallow optimization in `uneven_surface_shallow_dry_paired_20261014.md`.
> The old figures are retained only as a trace of the discarded run;
> do not cite them as system performance.


## Measurands

- Both heights are relative to the interior empty-tank bottom.
- Liquid level is the area mean of the liquid interface.
- Highest level is the maximum of waves and floating objects.
- Tank walls, rim, and background are excluded from truth.
- The new UnevenSurfaceMeter is research-only, not field-qualified.
- Its liquid output reuses the formal robust-return route as a
  proxy for area mean. The peak uses local-supported returns.
- No forced zero or value clipping is applied.

## Protocol

- Independent V45 synthetic set: 99 installations, four frames each.
- Eleven depth bands from dry to 9 m; three RGB-D sensor types;
  three camera distances. Camera-to-bottom distance is <= 10 m.
- Top-mounted cameras; two subscenes: waves only, and waves
  with three floating objects.
- Waves are a two-direction sinusoidal height field.
- Float proxies are elliptical domes, 30-60 mm above the
  local liquid interface, with radii 8-13% of tank radii.
- Floats determine the true peak in 81 of 90 wet installs.
- Dry-tank controls have no floats.
- Float depth is geometrically injected. RGB is not rerendered;
  inference here uses depth only. Visual float recognition is
  not demonstrated by this test.
- Wet inference does not read true surface or float masks.

## Wet-frame results

Tolerance: max(20 mm, 2% of the true bottom-relative height).
Pass rate uses all frames; missing outputs fail. MAE and P95
use numeric frames only.

| Scene | Frames | Liquid coverage | Liquid MAE | Liquid pass | Peak coverage | Peak MAE | Peak pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| Waves | 360 | 100% | 46.9 mm | 83.6% | 91.4% | 83.6 mm | 75.8% |
| Waves + floats | 360 | 100% | 31.9 mm | 85.6% | 91.4% | 55.8 mm | 81.7% |
| Combined | 720 | 100% | 39.4 mm | 84.6% | 91.4% | 69.7 mm | 78.8% |

Example, case 79 frame 0: liquid truth 6.447 m, estimate
6.450 m; peak truth 6.480 m, estimate 6.479 m.
This example is not representative of aggregate accuracy.

## Depth-stratified pass rate

Each row has 72 frames, pooling both subscenes.

| True liquid band | Liquid pass | Peak pass | Peak coverage |
|---|---:|---:|---:|
| Dry, 0 m | 41.7% | 58.3% | 100% |
| 0.01-0.1 m | 37.5% | 50.0% | 69.4% |
| 0.1-1 m | 33.3% | 34.7% | 55.6% |
| 1-2 m | 90.3% | 61.1% | 88.9% |
| 2-3 m | 88.9% | 76.4% | 100% |
| 3-4 m | 98.6% | 76.4% | 100% |
| 4-5 m | 100% | 94.4% | 100% |
| 5-6 m | 97.2% | 97.2% | 100% |
| 6-7 m | 100% | 100% | 100% |
| 7-8 m | 100% | 97.2% | 100% |
| 8-9 m | 100% | 100% | 100% |

## Failure analysis

- All 99 empty-container installations initialized.
- Liquid estimates are numeric in all 792 frames, but this
  includes physically impossible negative and above-rim values.
- The peak is missing in 62/792 frames: 14 have fewer than
  nine valid ROI returns; 48 have points but no sufficiently
  supported 3x3 neighborhood.
- In 111/792 frames the estimated peak is over 5 mm below
  the estimated representative liquid level. The dual output
  flags this contradiction; it does not silently overwrite it.
- Dry and shallow scenes confuse bottom, wall, and surface
  returns. Case 14 frame 2 predicts 1.590 m for a 0.061 m
  liquid mean using only a few container candidates.
- Structured-light layer voting can choose the dry layer
  wrongly under waves: case 68 frame 2 predicts 0.031 m
  for a 5.141 m mean. This is a tail failure, not a small bias.
- ToF shallow dry-pairing can yield a negative height:
  case 8 dry control gives -0.595 m. This remains visible
  because the protocol forbids clipping.
- Highest-point estimates can miss wave/float apices when
  returns are sparse, or include wrong-return clusters.
- These outcomes do not meet a robust industrial dual-height
  claim, especially below 1 m and for peak safety clearance.

## Next method changes, not yet qualified

1. Separate bottom, liquid, float, wall, and invalid returns
   using spatial/temporal evidence, not only a depth quantile.
2. Estimate a liquid height field over calibrated tank cells;
   report its area mean independently of the object peak.
3. Detect small connected high-evidence regions for the peak,
   with an explicit visibility/uncertainty flag when no apex
   can be observed.
4. Guard dry-layer voting and ToF dry pairing with consistency
   checks before re-evaluating on a new independent seed.
5. Rerender floating-object RGB and sensor returns together;
   validate actual RGB-D alignment and real camera data.

Code: src/liquid_depth/uneven_surface_meter.py and
scripts/evaluate_uneven_dual_height.py.
Results: uneven-dual-v5-20261014/summary.json and records.jsonl
under /root/autodl-tmp/liquid-depth-artifacts/evaluation.
