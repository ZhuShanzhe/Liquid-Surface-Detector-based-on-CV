# RGB-D 液面深度测量系统

当前正式研究版：**formal-20261016**。以相对容器内底面的实际液深为输出，整合普通、均匀透明、轻度均匀半透明、部分深度失效、低照、眩光、不平整和漂浮物八类入口。

这是可复现的服务器研究与离线测量版本，**不是已经完成真实相机、工业面板或计量安全验收的产品**。场景由操作员明确选择；不能把数值输出、仿真通过率或软件著作权准备等同于现场可靠性。

## 两份正式报告

- [正式链路与全场景测试报告（PDF）](reports/2026-10-16_system_and_scenario_evaluation.pdf)：代码架构、应用边界、全场景性能、ToF/结构光、按强度/实际液深/缺失率分层、误差区间占比，以及远距浅液位的修复历史与当前处理。
- [开发过程与重要进展（PDF）](reports/2026-10-16_development_history.pdf)：从工程底座、基线复现、仿真训练、指标纠正、先验折射校正到当前确定版的按序记录，包含未合并和回退决策。

详细证据：[966 个统计单元](reports/evidence/scene_metrics_20261016.json)、[15,870 条当前预测记录（gzip JSONL）](reports/evidence/current_predictions_20261016.jsonl.gz)、[来源与口径清单](reports/evidence/source_manifest.json)。不同数据组不相加成一个“总准确率”。

## 当前路线与相机边界

| 场景 | 当前液深 | 最高高度 / 视觉 | ToF 与结构光 |
|---|---|---|---|
| 普通漫反射 | 空槽建模 + 冻结对称候选几何 | 纯几何，不用视觉 | 两者有参数化仿真；远距浅液位仍弱 |
| 部分失效 | 同一冻结几何 | 不用视觉 | 仅承诺研究范围 0%-50% 液面数值空洞 |
| 低照 / 眩光 | 同一冻结几何，不自动提亮 | 不用视觉 | 两者各按低/中/强重新扩样 |
| 不平整 | 稳健代表液深，按物理面积均值评分 | 既有 V11 几何包络 | 两者有三档波幅仿真，平均与最高单列 |
| 漂浮物 | 同一液深，视觉不得改写 | 显式开关后使用 V8 selected | ToF/结构光沿用选定 V5 分支；主动双目才用 V8 学习排序 |
| 均匀透明 | 1 组空槽 + 4 组已知液深，baseline_v32 | 不启用 V8 | 当前成绩为左右目/IR图像代理；**不是 ToF 折射成绩** |
| 轻度均匀半透明 | 同一五组预算，midpoint_dense 对称配准 | 不启用 V8 | 尚无专用 ToF / 相位结构光正式硬件验证 |

多层透明、底纹消失、强散射强行归为透明、超过50%失效、连续相机运动和极端复合场景不在当前适配承诺内。强散射只有在传感器工作波段有可靠普通表面返回时才可能走普通路线。

## 指标与最新结果

H 是底面相对的实际液深，不是相机轴向距离。普通、透明、半透明、部分失效、低照、眩光、漂浮物液深通过容差为 `max(20 mm, 2% H)`；不平整液深为 `max(50 mm, 5% H)`。最高高度按 `max(50 mm, 5% 真实最高高度)`。

| 场景 | 对应证据数 | MAE mm | P95 mm | 全帧通过率 | 数值覆盖率 |
|---|---:|---:|---:|---:|---:|
| 普通 | 396 | 18.73 | 72.07 | 90.66% | 100% |
| 部分失效 0%-50% | 2,346 | 17.89 | 67.98 | 91.30% | 100% |
| 均匀透明 | 2,672 | 5.01 | 14.54 | 99.96% | 100% |
| 轻度均匀半透明 | 5,344 | 6.41 | 19.46 | 99.93% | 100% |
| 漂浮物液深，有液专项 | 360 | 26.59 | 133.95 | 87.22% | 100% |
| 低照，新有效输入集 | 1,584 | 12.96 | 57.24 | 94.00% | 100% |
| 眩光，新有效输入集 | 1,584 | 13.94 | 61.24 | 94.07% | 100% |
| 不平整，新有效输入集 | 1,584 | 11.67 | 47.19 | 96.91% | 100% |

这些不是同一纳入协议的样本：普通/缺失为既有确定版网格回放；透明为既有冻结图像预测重计（另做288帧接口核验）；漂浮物为既有 V8 有液专项；后三类才是新采集、审核物理来源并同条件补采的4,752帧。旧资料不能伪称已按新来源规则重采。普通等源汇总含主动双目，新特殊场景只含 ToF 和结构光，详见PDF相机分表。

新集共尝试2,566组成对输入，排除190组，合格采样率92.60%；强眩光只有60.27%。来源审核只确认合成输入有基本直接液面回波，不是实地在线可观测性证明；标签不进入测量器，也不按误差大小删帧。100%数值覆盖率是正式有效输入集上的结果，不代表任意原始输入都可测。空槽无相对误差定义。

## 安装与便携测量入口

Python 3.11 是现有服务器验证环境。几何/图像主链路、可选视觉和仿真依赖有所不同；安装完整研究依赖可使用：

```bash
git clone --recurse-submodules https://github.com/ZhuShanzhe/Liquid-Surface-Detector-based-on-CV.git
cd Liquid-Surface-Detector-based-on-CV
python -m pip install -e '.[dev,train,research]'
python -m pip install scikit-learn==1.9.0 joblib==1.5.3
PYTHONPATH=src:scripts python -m pytest -q
```

GPU PyTorch/CUDA 按目标硬件选择；已验证服务器为 RTX5090、Python3.11.15、PyTorch2.11.0+cu128、NumPy2.4.6、OpenCV4.14.0、SciPy1.17.1、scikit-learn1.9.0。不要把历史CUDA编译扩展无条件装到另一机器。ROS2只用于需要它的采集方案，离线推理不要求连接USB相机。

```bash
PYTHONPATH=src:scripts python scripts/measure_release.py \
  --scene ordinary_diffuse \
  --initialization /absolute/path/initialization.json \
  --depth /absolute/path/current_depth_m.npy
```

普通类初始化JSON：`{"kind":"empty_geometry","bundle":"container_geometry.json","dry_depth":"empty_depth_m.npy"}`。相对路径以该JSON所在目录解析；需要真实相机内参、对齐深度以及空槽/操作员少量独立米制标定，不能用当前测试真值替代。见[初始化说明](docs/empty_geometry_initialization_v41.md)。

```bash
# 只有漂浮物最高高度可启用视觉；液深保持相同算法
PYTHONPATH=src:scripts python scripts/measure_release.py \
  --scene floating_objects --initialization /absolute/path/initialization.json \
  --depth /absolute/path/current_depth_m.npy --rgb /absolute/path/current_rgb.png \
  --enable-floating-peak-vision
```

便携入口把[小模型资源清单](configs/release_assets.json)转换为临时绝对路径并验证SHA256；模型文件位于 `assets/models/v8/`。这些仅为项目生成的浮物视觉辅助模型，不包含DFNet/SwinDRNet/ClearGrasp等第三方大权重。只加载可信来源的PyTorch/joblib文件；哈希校验不代替来源信任或第三方许可核查。不开视觉时不加载这些权重。

透明类初始化JSON采用 `kind=five_priors`、`calibration`、左右掩码NPZ `masks`、五组左右图路径 `prior_images`、五个已知实际液深 `known_heights_m`。第一组为空槽0，其他四个互异为正，必须在实际容器容量内。测量时传 `--scene homogeneous_transparent`（或 `homogeneous_translucent`）和 `--left` / `--right`；它们必须是已整流、同机位、底纹可见的左右图像。

## 代码分层与复现

```text
src/liquid_depth/           测量、几何、图像先验、可选视觉；模型与面板分开
configs/                   确定路由、评分口径与可移植小模型清单
scripts/measure_release.py  当前便携单帧入口，只封装路径，不改变测量逻辑
scripts/measure_determined.py 原确定版回放入口，保持冻结
tests/                     回归、应用边界与资源校验
reports/                   两份正式PDF与可重算的精简预测/统计证据
docs/                      版本化研究记录；历史不代表当前默认算法
assets/models/v8/          四个项目生成的小型可选最高高度模型
ros2_ws/、third_party/      采集与上游驱动；第三方内容保留各自许可
```

完整模块关系、启用/未启用状态和研究分组见[代码索引](docs/formal_release_code_map.md)。原README在[历史备份](docs/archive/README_before_formal_20261016.md)。

服务器原始数据与模型大文件仍位于 `/root/autodl-tmp/liquid-depth-data` 和 `/root/autodl-tmp/liquid-depth-artifacts`，不会把数百GB数据或环境上传Git。最新特殊场景的生成/标定/输入审核/测量/统计入口为 `scripts/run_special_scene_strata_v1.py`，参数 `--root <数据目录> --out <新的结果目录> --workers 6`；渲染器需服务器已有Blender4.2路径或对应安装调整。已有证据不能冒充新盲测。

发布校验：**729 项测试通过、14 条依赖警告**；八类场景的便携CLI均与归档原入口同输入数值一致，视觉开关不改变液深。参见[验证清单](reports/evidence/release_validation.json)。精简预测可用 `python scripts/recount_release_metrics.py` 独立重算全部966个统计单元；完整PDF重建需要服务器归档的 `publication-inputs.tar.gz`，解包后传 `--input-root` / `--output-dir`，并提供合适的中文字体 `PDF_CN_FONT`、`PDF_CN_BOLD_FONT`。

## 下一阶段只优化远距浅液位

当前远浅定义为 `B>=6m && 0<H<=1m`（仅离线分组，推理不读H）。主要问题是小视差/量化、共同偏置、底面标定误差及底面/液面候选重叠。历史 V37多帧、V39公共偏置、V40空间去噪、V72量化似然并未全部默认接入当前冻结对称几何。透明则采用五组图像先验与相对响应，不依赖失效液面点云拟平面。

后续在独立安装级验证/测试集上做配对消融，并保留全量程防回退；不强制归零、不裁剪液深、不用测试误差反选模型。其他缺陷只记录，V9、多层透明、95%-100%极端失效不合并或继续扩展。
