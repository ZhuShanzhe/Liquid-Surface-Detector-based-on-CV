# 正式版代码索引与研究边界

发布标识 formal-20261016。相对2026-10-16确定版，本次只增加发布包装、资源位置与报告，不更改测量算法。仓库保留历史模块名以免破坏导入和可复现实验；“存在代码”不等于“已接入默认链路”。

## 确定入口

| 层级 | 文件 | 当前职责 |
|---|---|---|
| 便携发布CLI | `scripts/measure_release.py` | 校验项目小模型哈希，临时解析资源路径；调用原确定版接口 |
| 冻结CLI | `scripts/measure_determined.py` | 原始确定版回放；不为发布改写 |
| 全场景门面 | `src/liquid_depth/determined_scene_meter.py` | 八类显式路由，统一元数据与视觉边界 |
| 空槽初始化 | `src/liquid_depth/empty_scene_routes_v44.py`、`empty_geometry_v41.py` | 空槽RGB-D与独立米制锚点建模容器，不使用当前液深标签 |
| 普通类主液深 | `frozen_symmetric_geometry.py`、`symmetric_visual_meter.py`、`supported_scene_meter.py` | 冻结对称候选主输出，原正式路径回退 |
| 非透明组合器 | `hybrid_scene_meter_v2.py`、`adaptive_rgbd_meter.py` | 纯几何液深与几何最高高度；不是自动启用全部视觉实验 |
| 透明图像先验 | `supported_scene_meter.py`、`range_relative_v32.py`、`range_registration_v33.py` | 一组空槽+四组已知液深左右图；baseline_v32 / midpoint_dense |
| 不平整最高 | `uneven_surface_meter.py`、`uneven_surface_repair.py` | 几何最高包络与V11修复，平均液深仍由冻结对称几何给出 |
| 视觉输出边界 | `floating_peak_only_v4.py` | 即使后端选V8，也只允许漂浮物最高高度被视觉改写 |
| 视觉小模型 | `floating_rank_application_v8.py`、`floating_rank_v8.py`、`float_evidence_v3.py` | V8 selected；ToF/结构光V5分支、主动双目V8分支；模型从便携清单加载 |

## 场景分类

普通、部分失效、低照、眩光、不平整、漂浮物：空槽建模；ToF/结构光均有仿真。透明、轻半透明：底纹可见的左右目/IR图像先验；ToF未适配，专用相位结构光未有正式硬件成绩。强散射、多层、复合与连续运动不得由软件场景名字推定已适配。

## 评价与证据

- 普通/缺失：`scripts/evaluate_determined_depth_grid.py`，既有数据回放，0%-50%数值空洞；不将受控删点当独立安装。
- 原确定版：`scripts/evaluate_determined_v1.py`，透明全记录重计并抽样接口回放，漂浮物V8专项。
- 新低照/眩光/波面：`run_special_scene_strata_v1.py` → 生成 → 标定 → `prepare_special_scene_evidence_v1.py` → 估计 → 分层汇总。
- 输入审核：`special_scene_input_evidence_v1.py` 的普通空间支撑与离线采集来源审核。后者只验证仿真有基本物理证据，不将来源标签传给测量器。
- 发布PDF与统计：`scripts/build_publication_pdfs.py`（Windows中文字体或通过环境变量提供等效字体）与 `reports/evidence/`；报告内容/源记录哈希见发布清单。
- 全量运行依赖：原封存 `runtime-modules.json` 保留在发布证据目录，便携包装与原入口的同输入数值对比另存验证记录。

## 保留但未默认启用

V35/V36量化截断似然、V37静态多帧、V39公共偏置、V40空间去噪、V49-V72早期三场景、照明V2、通用深度多任务网络、SeeGroup/多层/时序及V9：保留为历史研究和候选修复，不把它们的旧成绩移作当前确定版性能。

训练、模拟、诊断、报告和旧交付脚本仍按版本名保留在 `scripts/`；测试依赖这些模块，因此不通过盲目改目录或删旧文件“清理”。今后只围绕远距浅液位修改，其他部分作防回退。

## 部署边界

`product_web.py`、`rail_runtime.py`、`system_runtime.py` 与 `ui/` 是已有面板和产品运行端。新确定版CLI可无USB回放，但本次未声称所有新算法已与该面板及真实硬件完整自动连接；正式上线需独立集成验收。

数据、环境、第三方大权重不进入Git。`assets/models/v8` 仅含四个本项目生成的小模型；加载joblib/PyTorch前需信任来源并核验哈希，不从未知来源下载后直接反序列化。第三方代码和数据仍受各自条款约束，本发布不授予其额外商业许可。
