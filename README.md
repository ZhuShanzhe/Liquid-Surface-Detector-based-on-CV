# ToF RGB-D 液深测量系统

当前正式版：`tof_formal_v1_20261017`。开发、公开测量入口及正式性能报告**只支持 ToF**；统一输出相对容器内底面的米制液深。这是项目维护范围决策，不代表所有真实设备类别的普遍排名。

本项目是固定机位、限定物理假设的仿真与软件接口版本，**不是任意真实相机、USB驱动、工业GUI或工业计量验收均已完成的产品**。

## 模块化统一入口与仿真展示（实验版，2026-10-19）

新增 `UnifiedToFMeter` 与统一 `ToFObservation` / `ToFInstallation`，整合既有数值算法；
RGB-D 多标签分类头提出场景建议，独立物理先验决定透明链路是否可用。
液深只使用一个核心；可选视觉仍仅影响漂浮物最高高度。
这不是自动选路已达到原正式版精度的声明，原人工场景正式入口保留不变。

```bash
LIQUID_PYTHON=/root/autodl-tmp/envs/liquid-depth/bin/python bash scripts/launch_scene_console.sh
# 本机另开终端，建立回环隧道后访问 http://127.0.0.1:18765
ssh -N -L 127.0.0.1:18765:127.0.0.1:8765 -p 26663 root@connect.bjb2.seetacloud.com
```

展示可旋转三维液面/容器、查看测量点云、调节液位、光照、眩光、漂浮物、起伏和缺失比例，
对照真值与估计值。空槽与五帧先验在仿真初始化中自动建立，当前真值不传给测量模块。
需要已配置的 Blender 与项目研究环境。没有连接 USB 相机，不控制真实泵阀。

- [使用流程、模块边界与组合限制](docs/UNIFIED_TOF_CONSOLE.rst)
- [分类与组合实验记录](reports/evidence/unified-console-v1-20261019/summary.json)

独立 288 图分类试验不等于液深计量认证；材质混淆、分类误报以及组合工况的高度长尾尚未解决。
透明+漂浮物/起伏、多层、强散射和大于50%液面缺失不纳入展示支持范围。
原正式性能 PDF 不由这些初步自动选路结果替换。

## 正式报告与历史依据

- [ToF 正式性能报告](reports/2026-10-17_tof_performance.pdf)：5744帧，场景总表、各场景独立表、失效/低照/眩光程度分层、绝对及相对误差区间、延时。不按实际液深分层。
- [ToF 正式计算链路](reports/2026-10-17_tof_formal_pipeline.pdf)：封面、目录、八类场景结构示意图、模块注释、标定与输入契约。
- [历史双相机性能报告](reports/2026-10-17_dual_camera_performance.pdf)：**原文件原样保留**，记录比较后选择ToF的开发依据；不是当前支持范围。原实现见Git提交 `15bfe99`。
- [更早开发进展](reports/2026-10-16_development_history.pdf) 与 [历史全场景报告](reports/2026-10-16_system_and_scenario_evaluation.pdf) 是历史记录，不是当前验收报告。

远距浅液位专题报告已从当前目录删除，其原始帧仍全部进入ToF统计，不通过删除困难帧提高成绩。

## 场景及应用边界

| 场景 | 输入与标定 | 当前计算 |
|---|---|---|
| 普通漫反射 | 空槽RGB-D、内参、轮廓、独立米制锚点 | 冻结对称候选几何 |
| 部分深度失效 | 同上，液面数值空洞0%-50% | 同一液深几何；不加置信拒绝门槛 |
| 低照 / 眩光 | 同上 | 不自动以RGB增强替代液深 |
| 不平整液面 | 同上 | 代表液深与V11最高包络 |
| 漂浮物 | 同上，可选RGB | 液深不变，视觉仅最高高度；V8 selected的ToF策略沿用V5 |
| 均匀透明 / 轻半透明 | 1空槽+4已知液深原始相位组，独立几何标定 | AMCW iToF光程与混合回波逆解 |

透明链路需要**三频四相原始相关图**，本轮20/30/50MHz；只导出深度图的SDK不能直接启用。脉冲dToF不等于已经支持该相位路线。底部调制须可见、液体性质和机位须稳定；不承诺多层、强散射、持续移动或超过50%深度失效。

仿真消光不是850/940nm实测水光谱，不证明近红外能穿透9m水层；真实部署须核查吸收、散射、信号预算、多径和温漂。

## 性能口径

从冻结的11390帧比较测试中保留**全部5744帧ToF**，预测与真值不变。未重新训练、未重新挑样，也未声称新的独立测试提升。保留原来源与安装聚类；每个报告小类不少于100帧。

液深容差 `max(20mm,2%真实H)`；不平整代表高度和所有最高高度为 `max(50mm,5%真实高度)`。空槽不算相对误差，仍计绝对误差及通过率；数值覆盖率不是准确率。

注意：漂浮物最高高度保留约3.43m孤立异常；液深稳定不等于最高高度安全可靠。算法延时不含完整采集，最高高度尾部可能超过0.5s；报告未隐藏这些限制。

## 运行

```bash
git clone --recurse-submodules https://github.com/ZhuShanzhe/Liquid-Surface-Detector-based-on-CV.git
cd Liquid-Surface-Detector-based-on-CV
python -m pip install -e '.[dev,train,research]'
python -m pip install scikit-learn==1.9.0 joblib==1.5.3
PYTHONPATH=src:scripts python -m pytest -q

PYTHONPATH=src:scripts python scripts/measure_tof.py \
  --scene ordinary_diffuse --initialization /path/tof_init.json --depth /path/current.npy

PYTHONPATH=src:scripts python scripts/measure_tof.py \
  --scene homogeneous_transparent --initialization /path/tof_phase_init.json \
  --phase-capture /path/current.npz

python scripts/recount_tof.py
```

初始化JSON的 `sensor_family` 必须为 `tof`，与bundle一致。普通 `kind=empty_geometry`，含 `bundle`、`dry_depth`；透明 `kind=five_phase_priors`，还含 `scene`、`profile`、`rays_roi`、五个 `prior_captures`、五个 `known_heights_m`（0及四个容量内递增非零高度）。NPZ含 `correlations[3,4,H,W]`。

漂浮物最高高度可加 `--rgb` 与 `--enable-floating-peak-vision`，其他场景不调用该视觉模块。长期运行常驻一个 `ToFSceneMeter`，不要每帧重新标定。服务器验证环境：Python3.11、PyTorch2.11.0+cu128、NumPy2.4.6、SciPy1.17.1、OpenCV4.14.0。

## 代码与复现

- `configs/application_route.json` / `tof_scene_routes.json`：唯一当前入口及场景契约。
- `src/liquid_depth/tof_scene_meter.py` / `tof_phase_refraction.py`：ToF门面及专用相位逆解，不含投影相位逆解。
- `scripts/measure_tof.py`：非ToF初始化直接报错，不隐式回退。
- `generate_tof_far.py` / `evaluate_tof_far.py`：ToF几何采样及输入审核。
- `tof_phase_simulator.py` / `evaluate_tof_phase.py`：ToF原始相位仿真，前向24层、逆向4层散射积分。
- `publish_tof_evidence.py` / `recount_tof.py` / `verify_tof_release.py`：证据导出、独立复算与观察回放。
- `build_tof_reports.py`：两份正式PDF，需ReportLab及中文字体；`PDF_CN_FONT`、`PDF_CN_BOLD_FONT`可指定字体。
- `reports/evidence/tof-formal-v1-20261017/`：当前预测、指标、验证、哈希。
- `reports/evidence/dual-camera-v2-20261017/`：仅为历史比较报告审计，不是当前相机入口；源码锁对应提交 `15bfe99`。

共享冻结几何源码与更早研究模块保留历史兼容实现以免破坏ToF依赖，**研究脚本不是支持的生产入口**。当前双相机门面、专用投影相位实现、配置及采样入口已撤销，后续不维护另一类相机链路。

服务器原观察位于 `/root/autodl-tmp/liquid-depth-data/synthetic/dual-camera-v2-20261017/`；原目录名保留用于来源核对，不伪造重新采集。切换备份及验证：`/root/autodl-tmp/liquid-depth-artifacts/tof-release-20261017/`。大数据、环境、密码不上传Git；可选小模型在 `assets/models/v8`，仅加载可信文件。
