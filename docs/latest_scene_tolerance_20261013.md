# 各场景最新版仿真结果：统一液深容差（2026-10-13）

Update 2026-10-14: The ordinary, homogeneous transparent/translucent, and
partial-loss rows below are historical and superseded by
`docs/supported_three_scene_summary_20261014.md`. The low-light, glare,
floating-object, and uneven-surface rows remain research baselines for the
next special-scene phase; they are not field-qualified product claims.

统一通过条件为 `|预测液深−真实液深| ≤ max(20 mm, 真实液深×2%)`。真实液深相对容器底面，不是相机到液面的距离。数值覆盖率指输出有限数值的帧占比；全帧通过率把未输出帧也计入分母。MAE 和 P95 仅统计数值帧。所有数字都来自已有独立仿真记录的重新计分，没有修改预测、归零、裁剪或门控。

| 场景与链路 | 帧数 | 数值覆盖率 | 全帧通过率 | 数值帧 MAE | 数值帧 P95 |
|---|---:|---:|---:|---:|---:|
| 透明 V33（baseline_v32） | 2,672 | 100.00% | 99.96% | 5.02 mm | 14.54 mm |
| 半透明 V33 | 5,344 | 100.00% | 99.93% | 6.41 mm | 19.46 mm |
| 低照 V45 | 396 | 59.85% | 59.34% | 10.08 mm | 31.07 mm |
| 眩光 V45 | 396 | 59.85% | 59.34% | 10.90 mm | 46.09 mm |
| 漂浮物 V45 | 396 | 66.16% | 63.38% | 16.24 mm | 82.50 mm |
| 不平整 V45 | 396 | 84.85% | 59.60% | 76.04 mm | 438.54 mm |
| 深度丢失 V50，0–50% 各档 | 每档 353 | 73.37–76.77% | 70.82–75.07% | 11.65–17.68 mm | 51.01–82.37 mm |
| 普通无候选诊断 V51b，0–50% 各档 | 每档 353 | 100.00% | 92.07–92.35% | 18.20–18.52 mm | 69.96–75.16 mm |
| 普通 V51b 原生完整审计 | 396 | 97.98% | 84.85% | 25.54 mm | 138.91 mm |

透明 V33 的冻结策略是透明使用 baseline_v32、半透明使用 midpoint_dense；上一版汇总误将透明也按 midpoint_dense 统计，此处已纠正。它仅覆盖顶部视角、单一均匀液体、5 帧已知液深先验的独立仿真测试，不能外推到多层透明容器或真实相机。V45 的低照、眩光和漂浮物测试沿用空槽 RGB-D 加人工米制锚点的估计几何，低覆盖主要与该版门控有关；不能仅依据已输出帧的 MAE 判断整体适用性。不平整液面未达到相同单平面指标，需另定义最浅、最深和平均液深等评价量。

深度丢失 V50 仅报告 0–50% 的目标适配区间。普通 V51b 放宽候选是诊断路线，尚非默认工业输出；其 353 帧配对组之外还有 35 帧自然高缺失帧及 8 帧几何初始化失败，因此必须同时看 396 帧完整审计。多层透明 V10 原评估使用相机射线表面深度代理，不是相对底面的真实液深，不能混入这个统一结果表。

可复算脚本：`scripts/rescore_latest_tolerance.py`。逐场景精确数据、源记录路径和清理清单分别位于服务器 `/root/autodl-tmp/liquid-depth-artifacts/evaluation/latest-scene-tolerance-20261013/summary.json`、`REPORT.md`、`cleanup-manifest.json`。先前 5 mm/2% 的分数仅作历史对照，不再作为当前通过率。结果尚未经过真实工业现场验证。
