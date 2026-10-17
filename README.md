# RGB-D 液深测量系统

当前版本：`dual_camera_v2_20261017`。正式入口只接受 **ToF / 结构光**，相对容器内底面输出米制液深；不以主动双目或旧左右目图像代理作为当前性能基准。

本项目目前是固定机位的可复现仿真研究与软件接口版本，**不是已经完成任意真实相机、USB驱动、工业GUI或计量验收的成品**。场景由操作员明确选择。“已接通路线”与“所有工况精度达标”不是一回事。

## 本轮三份报告

- [性能报告](reports/2026-10-17_dual_camera_performance.pdf)：封面、目录、场景总表；每场景一张ToF/结构光表，失效/低照/眩光按程度分行；绝对和相对误差区间、延时与输入审核。不按实际液深分层。
- [正式计算链路](reports/2026-10-17_dual_camera_formal_pipeline.pdf)：封面、目录，按八类场景给结构示意图、模块注释、标定与硬件数据契约。
- [远距浅液位不足](reports/2026-10-17_far_shallow_limitations.pdf)：封面、目录，全部场景两类相机的远浅总表、分路线原因与具体诊断案例。

报告统计11,390帧：保留未变更普通类链路的6,813帧ToF/结构光结果，新增远浅3,377帧，新增相位观测1,200帧。所有报告小类至少150帧，每场景每相机远浅至少100帧。旧透明左右目代理不混入；初次相位验证发现定义错误后，旧初测转为开发证据，修正后换独立种子重测。新旧不同采样构成不能直接作为算法提升百分比。

## 当前应用边界

| 场景 | 输入与标定 | 当前算法 |
|---|---|---|
| 普通漫反射 | 空槽RGB-D + 内参/轮廓/独立米制锚点 | 冻结对称候选几何 |
| 部分深度失效 | 同上，限定液面数值空洞0%-50% | 同一几何；不强制拒绝数值输出 |
| 低照 / 眩光 | 同上，程度固定仿真参数 | 不自动用视觉或提亮替代液深 |
| 不平整 | 同上 | 代表液深 + V11几何最高包络 |
| 漂浮物 | 同上，可按需输入RGB | 液深不变；视觉仅最高高度；两类相机沿用V5选定分支 |
| 均匀透明 / 轻半透明 | 1空槽 + 4已知液深的原始相位组 | ToF光程/混合回波逆解；结构光Snell投影相位逆解 |

透明路线需要原始AMCW iToF三频四相相关图，或可控投影器两周期四相条纹图及标定。**普通只导出深度图的ToF/封闭结构光SDK不能直接宣称可启用透明路线**；dToF、多层、强散射、持续移动和超过50%失效不在本版承诺内。当前模拟的35mm短基线结构光在远浅仍明显不足，不能据此承诺商业相机10m工业精度。

透明相位成绩以底部调制仍可见为条件：当前低消光常数模型未标定为850/940nm水的实测光谱，也未验证近红外穿透9m水层。ToF有效群折射率与结构光相折射率作常数近似；现场还必须核查发射波长下的吸收/散射及信号预算。

## 指标

液深均为内底面相对实际H。普通/透明/半透明/部分失效/低照/眩光/漂浮物液深用 `max(20mm,2%H)`；不平整均值用 `max(50mm,5%H)`；最高高度用 `max(50mm,5%真实最高高度)`。

数值覆盖率不等于达标率。空槽不计算相对误差，仍计入绝对误差。新远浅采集按空间支持与离线回波来源审核，不按测量误差选帧；来源标签不进入推理。相位残差和灵敏度只作诊断，不用它们删掉高误差帧。

## 运行

Python3.11；已有服务器验证环境为RTX5090、PyTorch2.11.0+cu128、NumPy2.4.6、SciPy1.17.1、OpenCV4.14.0、scikit-learn1.9.0。

```bash
git clone --recurse-submodules https://github.com/ZhuShanzhe/Liquid-Surface-Detector-based-on-CV.git
cd Liquid-Surface-Detector-based-on-CV
python -m pip install -e '.[dev,train,research]'
python -m pip install scikit-learn==1.9.0 joblib==1.5.3
PYTHONPATH=src:scripts python -m pytest -q

# 普通类：initialization含sensor_family、kind=empty_geometry、bundle、dry_depth
PYTHONPATH=src:scripts python scripts/measure_dual_camera.py \
  --scene ordinary_diffuse --initialization /path/init.json --depth /path/current.npy

# 透明类：five_phase_priors初始化，当前NPZ含correlations[F,4,H,W]
PYTHONPATH=src:scripts python scripts/measure_dual_camera.py \
  --scene homogeneous_transparent --initialization /path/tof_initialization.json \
  --phase-capture /path/tof_test.npz
```

普通初始化的 `sensor_family` 必须为 `tof` 或 `structured_light` 且与bundle一致。相位初始化还包含 `scene`、`profile`、`rays_roi`、五个 `prior_captures` 及五个 `known_heights_m`；第一组H=0，其余四组递增且在容器容量内。模型对象应常驻，避免每帧重新标定。漂浮物最高视觉可加 `--rgb` 与 `--enable-floating-peak-vision`，其他场景不执行此模块。

## 代码与复现

- `src/liquid_depth/dual_camera_scene_meter_v2.py`：当前正式双相机门面。
- `src/liquid_depth/phase_refraction_v2.py`：相位解码、五先验拟合与液深逆解。
- `scripts/generate_dual_far_v2.py` / `evaluate_dual_far_v2.py`：新容器、RGB、物理底面/浮物与双机制测试。
- `scripts/phase_capture_simulator_v2.py` / `evaluate_phase_refraction_v2.py`：原始相关/条纹观测、独立正向求解及相位测试。
- `scripts/summarize_dual_camera_v2.py`：无实际液深分层的统计、聚类区间及配额校验。
- `scripts/build_dual_camera_reports.py`：三份带封面目录的PDF，需ReportLab及中文字体。
- `reports/evidence/dual-camera-v2-20261017/`：预测、汇总、审核与代码/输入哈希，原始数据外置。

服务器原始数据：`/root/autodl-tmp/liquid-depth-data/synthetic/dual-camera-v2-20261017/`；完整结果与开发证据：`/root/autodl-tmp/liquid-depth-artifacts/dual-camera-v2-20261017/`。不将大数据、环境、凭据或第三方大权重上传Git。可选最高高度小模型位于 `assets/models/v8`，只加载可信来源的PyTorch/joblib文件。

旧2026-10-16入口、代码和报告保留为明确版本化的复现依据，不是当前默认应用；旧README见 `docs/archive/README_formal20261016.md`。以后主要处理远浅共性问题，其他场景以本版为防回退基线。
