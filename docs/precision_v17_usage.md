# V17 仿真精度优化：推理与复现说明

## 输入约束

本轮沿用 V16 的现场输入链路：RGB、原始深度、来自 SDK/出厂标定的单位与注册信息。没有增加液深、底高、容器尺寸、真值液面掩码、材料类别或真实尺度倍率。

训练损失使用真值监督；误差分组使用真值做统计。这些操作仅出现在训练和评估脚本中，独立推理入口不读取标签或仿真元数据。

“正常输入”指完整场景传感器近似生成的原始观测，仍含噪声、缺失和错误回波，不是无噪声真值深度。

## 新增处理

- 液面定位模块固定为 V16 权重。
- 比较预测区域外扩 4 像素、无外扩两种原始深度筛选方式，均保留至少 32 点的观测要求。
- 继续微调 DFNet 与连续修正分支；训练损失为逐帧绝对误差（米）加 0.1 倍相对误差，以减少大误差被压低权重的情况。
- 对预测液面区域做可选的倒深度平面稳健拟合：1/z = a*u + b*v + c。系数完全来自当前预测点，不需要已知相机到液面的距离、相机位姿或真实平面。
- 拟合只能抑制部分形状噪声，不能自动辨认正确回波层或修正整体尺度。静态近平面仿真液面的结果不能推广到明显波动的液面。本轮仿真相机没有镜头畸变；真实接入时，畸变校正需要先通过相机 SDK/标定进入观测预处理，不能直接宣称同等拟合精度。

权重和是否启用拟合均在开发集选择并固定，不能在测试时根据真值选择分支。材料类别不参与推理分支选择。

## 独立推理

```bash
PYTHONPATH=src python scripts/predict_precision_v17.py \
  --rgb /path/rgb.png \
  --raw-depth /path/raw_depth.npy \
  --camera-json /path/camera.json \
  --weights /root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916 \
  --transcg-source /root/autodl-tmp/liquid-depth-data/research/sources/TransCG \
  --output /path/new-output-directory
```

默认读取 runtime-profile.json 中在测试前确定的 selected 分支。--variant 可用于显式复现其他实验分支，不应按测试真值选择。

camera.json 沿用 V16：已对齐输入需要明确 depth_scale_to_m 和 depth_alignment="color"；未对齐时需要实际分辨率的出厂内参、畸变系数，以及设备内深度相机到彩色相机的旋转和平移。缺失参数不能默认当作正确标定。具体获取流程见 V16 输入说明。

输出 depth_m.npy、masked_depth_m.npy、mask_probability.npy、status.json。区域概率不是深度可信度；尚无最终液深输出，生产默认配置不变。

## 实验复现

1. 保留 V16 开发数据、定位与深度权重、官方 TransCG 源码及环境依赖。
2. scripts/train_precision_v17.py --stage diagnostic 运行输入范围与拟合诊断。
3. scripts/run_precision_v17.sh 训练两种输入范围。
4. scripts/run_precision_v17_evaluation.sh 在另一组验证序列选择处理方式，审计剩余验证序列，再生成和测试新场景。
5. scripts/export_precision_v17.py 收集冻结的定位与参考权重，准备独立推理目录。
6. scripts/summarize_precision_v17.py 汇总误差、恢复率、覆盖率、分距离结果与成对 bootstrap 区间。

训练、新场景测试和已冻结的选择结果拒绝覆盖；汇总脚本会更新统计文件。脚本为本轮使用固定服务器路径，复现时需先调整实验输出目录及新测试数据目录，不应覆盖本轮数据和权重。数据路径及种子记录在 protocol.json 与审计文件中。

## 验证边界

本轮目标是降低仿真中的轴向相机深度误差。容差仍为 max(5 mm, 2%)。深度置信度校准、真实相机泛化、底面参考和最终液深精度尚未验收。

## 距离组合对照（V17 第二轮）

第一轮新场景暴露近距离回退后，新增三组距离过渡范围：0.5–1.0 m、1.0–1.5 m、1.0–2.0 m。近距离采用 V16 加稳健拟合，远距离采用第一轮选定的 V17，过渡区线性组合。工作距离来自预测液面区域内的预测深度中位数，推理不接收真实距离或材料标签。

开发验证集没有 0.1–0.3 m 透明场景，三组方案均无法通过完整的近距离保护验证。因此只将开发集总体 MAE 最低的 0.5–1.0 m 组锁定为实验对照，并在另一批新场景上测试；没有升级为默认推理方案。第一轮测试集在第二轮中属于诊断来源，第二轮使用独立种子 20260917081。

需要复现实验对照时，在推理命令添加：

```bash
--variant distance_blend_diagnostic
```

该组合需要两次模型推理。status.json 额外输出 estimated_distance_m、far_weight 和 diagnostic_only。没有可用近分支深度的位置仍然不输出深度。

复现新增步骤：scripts/run_precision_v17b.sh → scripts/summarize_precision_v17b.py → scripts/export_precision_v17.py。可用 scripts/verify_precision_v17.py 检查独立推理与评估实现的一致性，并测量预热后的单帧耗时。该脚本默认使用当前服务器 GPU；不包含读取文件和 SDK 注册耗时。

第一轮实验默认仍是 mae_r4_plane；第二轮的保守选择、未获验证的距离组合及所有负面结果分别保存在 phase-b/ 中。生产配置未修改。
