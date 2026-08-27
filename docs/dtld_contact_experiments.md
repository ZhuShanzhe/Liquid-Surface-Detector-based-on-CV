# DTLD contact-perception experiments

This note records scene-disjoint DTLD experiments for the contact-line-to-container
geometry route. All promotion decisions use held-out curve error, not training loss.
The v1 manifest has 61,203 train, 12,881 validation, and 16,650 test instances.

## Promoted baseline

The promoted checkpoint is:

`/root/autodl-tmp/liquid-depth-artifacts/training/dtld-crm-resnet34-pretrained-full-v9/best.pth`

It uses an ImageNet-pretrained ResNet34 encoder, a learned five-channel RGB-D
adapter, CRM, multiscale contact/Bezier decoding, heteroscedastic curve confidence,
and the separate robust contact-to-CAD geometry solver. All three validation
selectors (mean, P95, and worst-object mean) chose epoch 8 of the 12-epoch run.

Validation at the selected epoch:

| Metric | Value |
| --- | ---: |
| Curve MAE | 23.23 px |
| Curve P95 | 86.58 px |
| Worst-object curve MAE | 32.55 px |
| Contact IoU | 0.217 |
| ALI residual RMSE | 0.095 |
| Confidence-error correlation | -0.576 |

Full scene-disjoint test results:

| Scope | Samples | Curve MAE | Median | P95 | Contact IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 16,650 | 22.54 px | 12.25 px | 83.51 px | 0.169 |
| Object 15 | 4,161 | 21.86 px | 8.79 px | 91.95 px | 0.183 |
| Object 16 | 4,163 | 35.02 px | 18.06 px | 109.44 px | 0.118 |
| Object 17 | 4,163 | 15.72 px | 4.31 px | 76.20 px | 0.377 |
| Object 19 | 4,163 | 17.56 px | 15.31 px | 27.34 px | 0.067 |

Against the previous scratch U-Net, overall MAE improves by 20.5% and median error
by 36.0%, but P95 improves by only 2.4%. Object 16 remains the main failure mode:
its MAE improves from 44.44 to 35.02 px, while sequence 000002 still reaches
36.01 px MAE and 124.80 px P95. Transfer learning therefore improves ordinary
accuracy much more than industrial tail risk.

## Selective risk

Raw curve confidence is not a calibrated probability, but it ranks errors usefully
on the full test set (confidence-error correlation -0.493). Validation must determine
any deployed threshold; test results below are descriptive only.

| Accepted coverage | Curve MAE | Median | P95 |
| ---: | ---: | ---: | ---: |
| 100% | 22.54 px | 12.25 px | 83.51 px |
| 75% | 15.81 px | 10.05 px | 60.22 px |
| 50% | 12.08 px | 7.83 px | 41.84 px |
| 25% | 7.22 px | 4.77 px | 14.33 px |
| 10% | 4.92 px | 3.42 px | 10.73 px |

This supports an industrial reject path, but pixel confidence must later be combined
with CAD reprojection support and metric uncertainty.

## Transfer-learning and robustness ablations

An ImageNet-pretrained ResNet34 pilot trained on one fifth of the training
instances first improved full-test MAE from 28.36 to 23.52 px, which justified
the promoted full-data run.

A follow-up pilot that sampled object 16 at 1.5 times its balanced weight slightly
improved stride-5 overall MAE (25.38 to 25.10 px) and P95, but worsened object-16
MAE (19.37 to 20.74 px). The global boost is therefore rejected.

A second full-data ablation resumed the promoted epoch-8 checkpoint with direct
geometry optimization and a detached heteroscedastic uncertainty objective:

| Epoch | Curve MAE | P95 | Worst-object MAE | Decision |
| ---: | ---: | ---: | ---: | --- |
| Promoted source (8) | 23.23 px | 86.58 px | 32.55 px | Keep |
| Decoupled 9 | 23.52 px | 88.62 px | 33.96 px | Reject |
| Decoupled 10 | 23.55 px | 89.69 px | 34.34 px | Reject |
| Decoupled 11 | 23.86 px | 90.31 px | 34.82 px | Reject |

The decoupled objective and object-boost controls remain reproducible switches
but default off. Neither isolated change resolves cross-container tail risk.

## Rejected pilot ablations

The early U-Net pilots used the identical stride-5 validation subset. The unchanged
U-Net checkpoint scores 31.26 px MAE on that subset.

| Ablation | Best MAE | P95 | Contact IoU | Decision |
| --- | ---: | ---: | ---: | --- |
| Pose/object FiLM | 32.23 px | 96.98 px | 0.177 | Reject for curve geometry |
| Per-object spatial expert heads | 32.25 px | 97.91 px | 0.173 | Reject for curve geometry |
| Symmetric curve-mask consistency, weight 0.15 | 32.16 px | 96.58 px | 0.157 | Reject |
| Object-16 sampling boost, 1.5x | 25.10 px | 91.82 px | 0.193 | Reject for object-16 MAE |
| Decoupled uncertainty | 23.52 px | 88.62 px | 0.215 | Reject against full baseline |

All rejected mechanisms remain optional experimental switches and default off.

## Reproducible evaluation

Run the full grouped and selective-risk evaluation with:

```bash
python scripts/evaluate_dtld_contact.py \
  --manifest /root/autodl-tmp/liquid-depth-data/research/manifests/dtld_v1.csv \
  --checkpoint /root/autodl-tmp/liquid-depth-artifacts/training/dtld-crm-resnet34-pretrained-full-v9/best.pth \
  --split test \
  --output /root/autodl-tmp/liquid-depth-artifacts/training/dtld-crm-resnet34-pretrained-full-v9/test_report.json
```

The report includes overall, object, sequence, object-sequence, scenario, and
difficulty-tag groups.

## Sparse reliable contact route

The promoted downstream interface now treats dense curve accuracy as an
intermediate diagnostic, not the final objective. Each predicted sample gets a
point confidence from the independent contact heatmap. The metric stage keeps a
small spatially distributed subset and rejects insufficient point count,
horizontal coverage, CAD reprojection support, ambiguity, or robust-consensus
support. Synthetic cylinder tests cover sparse selection, rejection, robust
metric recovery, and pixel-to-millimetre perturbation propagation.

This changes the research question from "is every liquid-boundary pixel
correct?" to "are enough geometrically observable contact points correct to
support the requested metric risk?" Real millimetre claims still require the
project camera calibration, container CAD/inner-wall scan, poses, and measured
liquid levels.

## Next promoted algorithm work

1. Optimize object 16 with validation-driven hard-example mining over height,
   glare, transparency, occlusion, and edge difficulty rather than a fixed
   object-wide sampling multiplier.
2. Test a tail-aware geometry objective (for example batch CVaR/top-k error) as a
   separate controlled ablation; do not combine it with sampling changes initially.
3. Calibrate perception uncertainty on validation and combine it with robust CAD
   reprojection confidence; report risk at fixed accepted coverage.
4. Obtain project-container CAD or measured inner-wall point clouds. DTLD/TCLD does
   not release the required CAD models, so contact pixels cannot yet be converted
   into honest DTLD millimetre error. Synthetic geometry tests verify the solver,
   not the missing real calibration.
5. After metric geometry is available, add temporal robust Kalman filtering and
   evaluate final relative depth error, absolute error, jitter, coverage, and
   rejection accuracy at the intended industrial ranges.
