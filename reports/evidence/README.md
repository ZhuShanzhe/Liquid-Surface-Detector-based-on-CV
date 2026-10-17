# 测试证据索引

当前正式版为 `tof_formal_v1_20261017`，见 [ToF正式版](tof-formal-v1-20261017/README.md)，仅含ToF。以下2026-10-16文件与双相机V2目录均为历史开发依据，不代表当前支持范围。

发布版本 `formal-20261016`，算法冻结版本 `determined_all_scenes_v1_v8_peak_20261016`。

| 文件 | 用途 |
|---|---|
| `current_predictions_20261016.jsonl.gz` | 15,870条精简预测、实际底面相对液深真值、相机/强度/缺失率与来源数据组；用于重算，不替代原始图像 |
| `scene_metrics_20261016.json` | 966个统计单元，液深与最高高度共1,168组指标；键为 `scene\|sensor\|level\|loss\|depth_or_subset` |
| `source_manifest.json` | 原始预测记录、研究文档的SHA256、分组与指标口径 |
| `special_scene_stratified-summary.json` | 新4752帧原始分层汇总，包括其原协议中的置信区间；不可外推至全部容器或真实硬件 |
| `special_scene_capture-eligibility-summary.json` | 采集输入排除数与双相机配对纳入统计，不是按误差筛选 |
| `special_scene_final-quality-audit.json` | 同输入预测一致性、真实来源与分层审核 |
| `special_scene_manifest.json`、`special_scene_protocol.lock.json` | 新采集实验协议、源文件与种子冻结 |
| `runtime-modules.json`、`environment.json` | 原确定版运行模块与服务器版本 |
| `release_validation.json` | 发布回归、入口一致性、独立重算、PDF及备份校验 |
| `portable_cli_smoke_20261016.json`、`pytest_20261016.log` | 八类入口及729项回归的实际日志 |

重算：在仓库根目录运行 `python scripts/recount_release_metrics.py`。它只需NumPy与仓库内精简预测，不需原始图像、模型权重或服务器绝对路径。

完整报告重建：服务器 `/root/autodl-tmp/liquid-depth-artifacts/publication-20261016/publication-inputs.tar.gz` 解包后，运行 `scripts/build_publication_pdfs.py --input-root <解包目录> --output-dir <新输出目录>`；需ReportLab和中文TrueType字体，通过 `PDF_CN_FONT`、`PDF_CN_BOLD_FONT`指定。字体不随仓库分发。

不同数据组的纳入协议、先验预算和相机集合不同，不能将这些预测直接拼成一个对外“总准确率”。只有新低照/眩光/不平整数据完成这次直接液面回波来源审核；透明成绩仍为底纹可见的左右目/IR图像代理，不包含ToF透明液体合格结论。
