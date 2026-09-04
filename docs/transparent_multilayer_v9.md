# Transparent and translucent multi-layer depth V9.1

## Goal

The specialist route resolves several physical interfaces on one camera ray
instead of forcing every transparent pixel into one depth. The production
targets remain metric liquid-surface depth and final liquid height in the
project's 0.1–10 m operating range.

The ordinary V8 route remains unchanged. V9.1 is loaded only for a qualified
transparent, translucent, or multi-layer scene and must pass the same selective
confidence and 500 ms warm end-to-end latency gate before product activation.

## Why this is a SeeGroup adaptation, not a direct copy

[SeeGroup (CVPR 2026)](https://github.com/princeton-vl/SeeGroup) treats the
depths observed along each pixel ray as an unordered point process. Its recurrent
decoder extracts one Laplace component at a time, and its maximum-intensity
likelihood lets the network decide which surfaces belong to which component.
The official model is monocular and its LayeredDepth real benchmark is mainly
relative-depth supervision. It does not by itself provide calibrated metric
liquid height.

This project keeps those three useful ideas:

1. recurrent component extraction with latent evidence erasure;
2. unordered components during training, with sorting only for auxiliary
   gradient continuity and physical interpretation;
3. bidirectional maximum-intensity Laplace loss, so every target layer is
   explained and every active prediction is supported.

It adds RGB-D metric encoding, component existence, learned liquid-interface
identity, a V8 single-depth metric prior, physical rejection, and the project's
metric accuracy gates.

## Ray labels

Each synthetic top-view ray can contain up to four events:

- transparent container wall or rim;
- liquid free surface;
- a second transparent wall;
- container interior bottom behind the liquid.

The V3 labels omitted the interior bottom from most top-view rays. V4 regenerates
only analytic layer labels from the already archived camera and container
metadata; RGB renders and the established train/validation/test sequence split
are unchanged. Derived arrays are compressed and stored separately, preserving
the original V3 data.

## Model and output contract

The V8 U-Net decoder supplies a full-resolution feature map to a shared recurrent
four-component head. Every component predicts:

- metric center depth in 0.1–10 m;
- Laplace scale (aleatoric uncertainty);
- existence probability;
- probability of being the liquid interface.

The primary output remains unordered. The runtime returns the selected liquid
depth only when semantic interface probability, component confidence, and the
calibrated V8 metric prior agree. It otherwise rejects the pixel/frame rather
than silently choosing the nearest transparent layer.

## Training sequence

1. Generate V4 layer labels with
   `scripts/upgrade_multilayer_labels.py`.
2. Initialize the metric backbone from V8.
3. Train only the recurrent ray head for four epochs.
4. Jointly fine-tune for six epochs at 5% of the ray-head learning rate.
5. Evaluate layer-set MAE/AbsRel, per-layer tolerance recall, multi-layer tuple
   accuracy, layer-count MAE, selected liquid-depth accuracy, and accepted
   coverage.
6. Calibrate confidence by scenario and validate warm end-to-end latency.
7. Keep the specialist disabled until all promotion gates pass.

The reproducible entry point is
`scripts/run_v9_transparent_multilayer_pipeline.sh`. It regenerates labels only
when absent, trains V9.1, and writes a confidence-calibration report.

## V9.1 synthetic validation result

The scene-disjoint validation split contains 481 transparent, translucent,
multi-layer, and compound scenes. At the conservative confidence threshold
0.50, the selected liquid interface achieved:

- liquid AbsRel: **2.92%**;
- liquid MAE: **15.86 mm** over the mixed 0.1–10 m range;
- `max(5 mm, 2%)` pass rate: **53.76%**;
- accepted liquid-pixel coverage: **52.09%**;
- frame output coverage: **69.23%**;
- evaluable acceptance rate: **99.40%**.

Using at least 64 accepted points and the robust median signed-depth residual as
a top-view surface-level proxy gives 2.69% AbsRel, 22.66 mm MAE, 53.11%
tolerance pass rate, and 66.94% frame coverage. This validates the proposed
"recover reliable points, then reconstruct the surface" direction, but is not a
substitute for the final camera-coordinate plane fit against a calibrated bottom.

The full unordered ray set achieved 8.61% layer-set AbsRel, 39.51% per-layer
tolerance recall, 13.74% multi-layer tuple accuracy, and 1.00 layer-count MAE.
Relative to the first V9 experiment, per-layer recall rose from 30.67% and tuple
accuracy from 5.16%. Relative to the unchanged single-depth output in the same
validation run, selected-interface AbsRel fell from 10.66% to 2.92%.

Therefore V9.1 passes the **simulation-candidate** gates for the final selected
liquid interface. It does not make every optical interface engineering-grade:
full multi-layer reconstruction remains an auxiliary research output. The resident
model plus interface selector measured 5.05 ms P95 over 300 RTX 5090 repetitions
at 320x180, well below the 500 ms budget; this excludes camera I/O, preprocessing,
plane fitting, and UI. The route stays disabled in production until real RGB-D
domain validation, final calibrated plane-level height evaluation, false-accept
testing, and full warm end-to-end P95 latency pass.

Rejection codes are part of the output contract: `0` accepted, `1` metric-prior
disagreement, and `2` confidence below the configured threshold.

## Promotion gates

A V9.1 checkpoint is eligible for a product trial only if, on scene-disjoint
transparent/semtransparent validation:

- selected liquid AbsRel is at most 3%;
- `max(5 mm, 2%)` pass rate is at least 50%;
- accepted output coverage is at least 30%;
- evaluable output rate is at least 90%;
- multi-layer recall and tuple accuracy improve over the V8 single-depth proxy;
- ordinary route results do not regress because V8 remains the default route;
- warm camera-to-output P95 latency is at most 500 ms;
- unsupported ambiguity is accompanied by a confidence and rejection reason.

These are simulation qualification gates, not final industrial certification.
A small traceable real-camera set is still required for metric calibration,
sensor-domain validation, and final acceptance.
