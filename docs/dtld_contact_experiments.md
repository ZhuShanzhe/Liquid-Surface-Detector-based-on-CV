# DTLD contact-perception experiments

This note records scene-disjoint DTLD experiments for the contact-line-to-container
geometry route. All promotion decisions use held-out curve error, not training loss.
The v1 manifest has 61,203 train, 12,881 validation, and 16,650 test instances.

## Promoted baseline

The promoted checkpoint is:

`/root/autodl-tmp/liquid-depth-artifacts/training/dtld-crm-bezier-domain-full-v3/best.pth`

It uses CRM, RGB-D input, contact segmentation, four spatial control-point heatmaps,
cubic Bezier sampling, heteroscedastic curve confidence, and the separate robust
contact-to-CAD geometry solver. Training was stopped after epoch 9 because epochs
2-9 did not improve the epoch-1 validation curve MAE.

Validation at the selected epoch:

| Metric | Value |
| --- | ---: |
| Curve MAE | 31.20 px |
| Curve P95 | 93.22 px |
| Contact IoU | 0.145 |
| ALI residual RMSE | 0.124 |
| Confidence-error correlation | -0.458 |

Full scene-disjoint test results:

| Scope | Samples | Curve MAE | Median | P95 | Contact IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 16,650 | 28.36 px | 19.13 px | 85.58 px | 0.165 |
| Object 15 | 4,161 | 30.49 px | 21.64 px | 89.75 px | 0.149 |
| Object 16 | 4,163 | 44.44 px | 37.51 px | 111.61 px | 0.102 |
| Object 17 | 4,163 | 22.08 px | 12.94 px | 67.00 px | 0.223 |
| Object 19 | 4,163 | 16.44 px | 14.62 px | 33.97 px | 0.179 |

Object 16 is the main failure mode. Its 44.44 px learned error is close to the
43.58 px train-object-mean baseline, so the shared model is largely reverting to a
container-specific prior on that domain. Test sequence 000002 is also below the
object-16 training height range, but sequence 000007 remains poor despite being in
range; this is not only an extrapolation problem.

## Selective risk

Raw curve confidence is not a calibrated probability, but it ranks errors usefully
on the full test set (confidence-error correlation -0.599). Validation must determine
any deployed threshold; test results below are descriptive only.

| Accepted coverage | Curve MAE | Median | P95 |
| ---: | ---: | ---: | ---: |
| 100% | 28.36 px | 19.13 px | 85.58 px |
| 75% | 20.68 px | 13.75 px | 63.60 px |
| 50% | 14.79 px | 9.73 px | 40.95 px |
| 25% | 9.44 px | 6.91 px | 26.52 px |
| 10% | 5.60 px | 4.33 px | 13.57 px |

This supports an industrial reject path, but pixel confidence must later be combined
with CAD reprojection support and metric uncertainty.

## Rejected pilot ablations

All pilots warm-started from the promoted checkpoint and used the identical
validation stride-5 subset. The unchanged checkpoint scores 31.26 px MAE on that
subset.

| Ablation | Best MAE | P95 | Contact IoU | Decision |
| --- | ---: | ---: | ---: | --- |
| Pose/object FiLM | 32.23 px | 96.98 px | 0.177 | Reject for curve geometry |
| Per-object spatial expert heads | 32.25 px | 97.91 px | 0.173 | Reject for curve geometry |
| Symmetric curve-mask consistency, weight 0.15 | 32.16 px | 96.58 px | 0.157 | Reject |

The first two ablations improve segmentation while degrading curve position. They
remain optional experimental switches and default off. Curve-mask consistency also
defaults to zero because its pilot did not improve held-out geometry.

## Reproducible evaluation

Run the full grouped and selective-risk evaluation with:

```bash
python scripts/evaluate_dtld_contact.py \
  --manifest /root/autodl-tmp/liquid-depth-data/research/manifests/dtld_v1.csv \
  --checkpoint /root/autodl-tmp/liquid-depth-artifacts/training/dtld-crm-bezier-domain-full-v3/best.pth \
  --split test \
  --output /root/autodl-tmp/liquid-depth-artifacts/training/dtld-crm-bezier-domain-full-v3/test_report.json
```

The report includes overall, object, sequence, object-sequence, scenario, and
difficulty-tag groups.

## Next promoted algorithm work

1. Replace the scratch spatial head with an ImageNet-pretrained, official-style
   BezierLaneNet/ResNet34 or comparable transformer encoder and retain strict
   scene-disjoint evaluation.
2. Optimize object-16 performance explicitly, using balanced hard-example mining
   across its two held-out height regimes instead of changing the global score only.
3. Calibrate perception confidence on validation and combine it with robust CAD
   reprojection confidence; report risk at fixed accepted coverage.
4. Obtain project-container CAD or measured inner-wall point clouds. DTLD/TCLD does
   not release the required CAD models, so contact pixels cannot yet be converted
   into honest DTLD millimetre error. Synthetic geometry tests verify the solver,
   not the missing real calibration.
5. Only after metric geometry is available, add temporal robust Kalman filtering and
   evaluate final centimetre error and jitter at industrial ranges.
