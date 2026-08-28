# Complex-scene evaluation protocol

The specialist router is evaluated with separate scenario buckets rather than one pooled average. A frame may belong to several buckets because glare and depth dropout often occur together.

Run `scripts/build_scene_evaluation_manifest.py` to create a deterministic manifest with per-source caps. The default thresholds are identical to the production automatic router. The generated summary records the thresholds and supervision contract used for every source.

The contracts are intentionally different:

- TransCG, ClearGrasp, and TODD provide metric depth-restoration supervision.
- TRADE real is a container, illumination, and RGB-D stress test; it has no per-pixel metric corrected-depth target.
- LayeredDepth real evaluates relative multi-layer ordering only. It must never be reported as centimeter liquid-height accuracy.

Report each specialist route independently with accepted coverage, false acceptance, MAE/RMSE/P95 when metric targets exist, temporal stability for videos, and warm end-to-end P95 latency. A model is promoted only if it improves its target bucket without an unacceptable regression on the standard route.


## Executable commands

Build the deterministic cross-dataset manifest:

```bash
python scripts/build_scene_evaluation_manifest.py \
  --multitask-manifest /root/autodl-tmp/liquid-depth-data/research/manifests/research_multitask.csv \
  --trade-manifest /root/autodl-tmp/liquid-depth-data/research/manifests/trade_real_v1.csv \
  --layereddepth-root /root/autodl-tmp/liquid-depth-data/research/layereddepth \
  --output /root/autodl-tmp/liquid-depth-artifacts/evaluation/scene_eval_v1.csv \
  --max-per-source-bucket 128
```

Run the no-repair reference and a TorchScript candidate under the same contract:

```bash
python scripts/run_depth_baseline.py \
  --manifest /root/autodl-tmp/liquid-depth-artifacts/evaluation/scene_eval_v1.csv \
  --backend identity --limit 32 \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/evaluation/scene_eval_v1_identity_32

python scripts/run_depth_baseline.py \
  --manifest /root/autodl-tmp/liquid-depth-artifacts/evaluation/scene_eval_v1.csv \
  --backend torchscript \
  --model-path /root/autodl-tmp/liquid-depth-artifacts/models/multitask-object-pretrain-v1.ts \
  --limit 32 \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/evaluation/scene_eval_v1_torchscript_32
```

Non-metric TRADE and LayeredDepth rows are deliberately excluded by this metric
runner and reported as skipped rather than being misrepresented as metric depth.

## 2026-08-28 pilot decision

A 32-record metric pilot compared the identity reference with the existing
object-pretrained TorchScript model. Error is conditional on pixels for which the
method emitted positive depth, so it must always be read together with coverage.

| bucket | identity MAE / coverage | candidate MAE / coverage | decision |
| --- | ---: | ---: | --- |
| glare | 40.3 mm / 0.341 | 26.9 mm / 1.000 | promising; fine-tune and revalidate |
| depth failure | 1963 mm / 0.324 | 1672 mm / 1.000 | fails accuracy gate |
| low light | 0.0 mm / 0.774 | 107.6 mm / 0.774 | regression; keep disabled |
| transparent general | 5.9 mm / 0.778 | 46.6 mm / 1.000 | coverage/error trade; not globally safe |

The zero low-light identity error does not mean perfect restoration: it is measured
only on already-valid raw pixels and the 0.774 coverage exposes the missing area.
The current checkpoint is therefore retained only as an offline glare candidate.
It is not promoted for automatic production routing. Depth-failure, low-light, and
transparent/multi-layer specialists remain disabled until their independent gates
pass. The pilot median model latency was 19.8 ms on the RTX 5090, well below the
500 ms budget; accuracy, not latency, is the blocker.


## Industrial tolerance and automatic promotion

The metric evaluator now reports two complementary acceptance metrics using the
project engineering tolerance

```text
T(z) = max(0.003 m, 0.01 * z)
```

where `z` is metric target depth. `within_tolerance_rate` is conditional on
positive predictions, while `within_tolerance_coverage` is relative to all target
pixels and therefore penalizes both missing and inaccurate predictions. Neither
may be reported without ordinary prediction coverage.

After a candidate is exported and evaluated under the same manifest, apply the
promotion gate:

```bash
python scripts/assess_specialist_promotion.py \
  --baseline /path/to/identity/summary.json \
  --candidate /path/to/candidate/summary.json \
  --scenario glare \
  --max-mae-m 0.010 \
  --min-coverage 0.80 \
  --min-within-tolerance-coverage 0.70 \
  --min-mae-improvement-fraction 0.10 \
  --guard-scenario ordinary \
  --output /path/to/promotion.json
```

The command exits with status 2 when any accuracy, coverage, failure-rate,
latency, or no-regression gate fails. A failed report must not enable the model.

The plane path also checks the convex-hull area of accepted inliers relative to
the liquid mask. This rejects a dense but spatially compact reflection patch that
could otherwise produce a numerically stable false plane. Frames with no usable
depth now produce a normal rejected result with
`insufficient_liquid_depth_support`; they no longer abort a batch.


## 2026-08-28 glare specialist v1 decision

The 12-epoch glare specialist was evaluated on 521 metric RGB-D records after
fixing two evaluation-contract defects: multi-channel EXR depth selection and
independent unit scales for targets and metric predictions. Both the identity
reference and candidate completed with zero failed frames.

| glare route | MAE | RMSE | prediction coverage | within-tolerance coverage | median latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| sensor identity | 21.84 mm | 158.72 mm | 0.434 | 0.163 | 7.69 ms |
| learned replacement | 20.09 mm | 30.24 mm | 1.000 | 0.191 | 22.09 ms |
| preserve-valid fusion | 19.19 mm | 29.04 mm | 1.000 | 0.249 | 23.45 ms |

Preserve-valid fusion also avoided regression on the transparent-general and
low-light guard buckets, but the candidate still failed the 10 mm MAE and 0.70
within-tolerance-coverage gates. It remains disabled. The next experiment adds
a normalized tolerance-exceedance loss and fine-tunes from this checkpoint.
These object-depth benchmarks qualify restoration behavior; a liquid-specific
plane/height holdout is still required before any industrial claim.
