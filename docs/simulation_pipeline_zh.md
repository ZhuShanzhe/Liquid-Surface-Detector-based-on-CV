# 项目专用液体仿真与训练流水线

## 目标

仿真数据用于扩大透明、半透明、眩光、暗光、深度大面积失效、多层透明、液面波动和漂浮物的覆盖。最终工业精度仍必须由独立真实场景验证，仿真测试不能替代真实验收。

每帧同时生成：

- Blender光学渲染RGB；
- 液面米制真值深度和相机坐标法向；
- 液面掩码；
- 前后容器壁与液面的有序多层深度；
- 主动双目代理模型生成的原始坏深度；
- 深度空洞、高光、逐像素不确定度；
- 相机内外参、液体/容器/照明参数和场景标签。

当前 `active_stereo_proxy_v1` 是快速的结构化噪声代理，不宣称精确复现任意具体相机。生产阶段应使用少量实拍的空洞率、视差噪声和错误回波分布校准代理参数，并把DREDS式红外投影+双目匹配作为高保真子集。

## 数据生成

冒烟集：

```bash
/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender \
  --background --python scripts/generate_synthetic_liquid.py -- \
  --output-root /root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v1 \
  --count 40 --width 640 --height 360 --engine eevee
```

数据可以分片追加；`--start-index` 指定起始编号，已完成样本默认跳过。正式生成使用 `--engine cycles --render-samples 32`，先完成2,000帧试验再决定是否扩展到50,000帧。

重建和验证清单：

```bash
python scripts/build_synthetic_manifest.py \
  --root /root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v1

python scripts/validate_synthetic_dataset.py \
  --manifest /root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v1/manifest.csv \
  --min-samples 40
```

## 0.1-10m通用模型

v3模型对输入和输出深度使用对数量程参数化。同一个检查点覆盖0.1-10m，分量程只用于采样和诊断，不是多个量程模型。

```bash
python scripts/train_universal_multitask.py \
  --manifest /root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v1/manifest.csv \
  --output-dir /root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v3-smoke \
  --epochs 3 --batch-size 16 --image-size 320,180 \
  --min-depth-m 0.1 --max-depth-m 10.0
```

测试按复杂场景与距离段分别汇报MAE、RMSE、AbsRel、1%/3mm容差命中率和推理延时：

```bash
python scripts/evaluate_universal_checkpoint.py \
  --checkpoint /root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v3-smoke/best.pth \
  --manifest /root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v1/manifest.csv \
  --split test \
  --output /root/autodl-tmp/liquid-depth-artifacts/evaluation/universal-liquid-v3-smoke.json
```

## 数据扩展顺序

1. 40帧冒烟集：检查几何、单位、标签和Blender兼容性。
2. 2,000帧试验集：验证训练能收敛，并和v2及SwinDRNet作公开真实集回归测试。
3. 根据误差分层补采仿真场景，而不是盲目平均扩容。
4. 加入少量真实工业帧校准主动深度噪声和光照分布。
5. 只有仿真、公开真实集和项目真实留出集同时通过，才允许替换生产模型。

完整流体求解只用于少量动态高保真样本。大多数样本使用可控的倾斜、波纹和弯月面高度场，避免把算力消耗在对当前深度恢复任务没有额外监督价值的CFD帧上。
