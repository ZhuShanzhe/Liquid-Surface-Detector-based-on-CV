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
  --multitask-manifest /root/autodl-tmp/liquid-depth-data/research/manifests/research_multitask_v1.csv \
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
