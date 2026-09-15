# V20.1 实验使用说明（未替换V19）：固定相机的双层透明液体折射估计

## 输入边界

沿用固定顶部相机、一次空容器标定。不输入液体种类、真实折射率、层厚、层数、真实液体掩码或相机至液面距离。当前只实现静止、近水平、两种透明液体的条件性估计；不能用于任意层数的液体。

一次空容器标定需要同一相机位置的 RGB、深度图及 SDK 内参和深度单位。运行时采集同位置的 RGB 和连续深度帧。测试使用 320×180 RGB、64 帧深度；不需要新增视角或分辨率。实机采集应锁定曝光、白平衡并维持相机位置，RGB 和深度须对齐。64 帧的等待时间取决于设备帧率，不计入算法运行时间。

当前传感器前向模型仅支持声明为 `nondispersive_coaxial_tof_proxy` 的无色散、共轴、单路径 ToF 仿真代理。不能把实际设备的类型改名来绕过检查；实机必须先完成其深度生成模型的标定和验证。

## 推理

在服务器仓库目录执行：

```bash
export PYTHONPATH="$PWD/src"
/root/autodl-tmp/envs/liquid-depth/bin/python scripts/infer_refraction_consensus_v20.py \
  --rgb <当前RGB.png> \
  --raw <连续深度.npz> --raw-key raw \
  --empty-rgb <空容器RGB.png> \
  --empty-depth <空容器深度.npz> --empty-depth-key empty_raw \
  --camera <camera.json> \
  --output <result.json>
```

深度支持 NPY 或 NPZ。当前深度形状为 T×H×W；空容器深度为 H×W。RGB 文件由工具读取到 [0,1]。推理至少要求16帧，64帧是此次验证条件。动态回波、缺失数据和明显整体漂移会被拒绝。

`camera.json` 示例仅展示字段；数值必须来自采集和 SDK：

```json
{
  "width": 320,
  "height": 180,
  "fx": 277.128,
  "fy": 277.128,
  "cx": 160,
  "cy": 90,
  "depth_scale_to_m": 1,
  "depth_alignment": "color",
  "sensor_model": "nondispersive_coaxial_tof_proxy"
}
```

## 输出含义

新增 `selection` 表示 `v20_multiscale`（三种平滑尺度一致）、`v19_fallback`（原 V19 完整折射检查回退）或 `rejected`。`consensus` 保留各尺度结果、候选范围并集及旧分支是否接受；两个分支厚度相差超过3 mm时拒绝。回退分支不是 SeeGroup。


- `accepted`：在当前前向模型与证据门槛下，是否允许输出内部界面位置。
- `thickness_m`：接受时的上层液体厚度；拒绝时为 null。
- `internal_depth_m`：接受时的内部界面相机轴向深度。
- `total_depth_m`：由观测顶面与空容器底面标定推得的总液深；下层厚度可由总液深减上层厚度得到。
- `candidate_thickness_m`：诊断候选，拒绝时不能当作可靠测量使用。
- `profile_range_m`：符合当前代价阈值的一组参数对应的厚度范围，**不是经过概率校准的 95% 置信区间**。
- `liquid_identity_verified` 始终为 false。通过检查不能证明任意稳定回波一定来自真实液体界面。
- `temporal`：独立空间像素数、残余噪声和帧间漂移。新分支重复帧不增加独立空间支持点数。若 selection 为 v19_fallback，internal_points 沿用原 V19 的跨帧观测计数，不能当作独立空间点数。

V20.1 默认入口始终执行多尺度检查与完整回退规则。旧 `infer_refraction_v20.py` 保留为初版复现实验，其结果不能替代 V20.1 的默认流程。

## 复现与冻结

- 新方法：`src/liquid_depth/refraction_*_v20.py`。
- 独立测试编排：`scripts/validate_refraction_consensus_v20.py`。
- 新数据生成：`scripts/generate_stratified_refraction_v20.py`，修复四组场景随机参数消耗不一致的问题。
- 旧 V19 文件和 V17 单液体模型未修改。
- 输出目录必须全新；已有实验不会被覆盖。
- 详细配置与源文件哈希见交付包 `frozen-consensus-before-heldout.json`。完整仿真深度序列保留在服务器，交付包含哈希清单和代表样本。
