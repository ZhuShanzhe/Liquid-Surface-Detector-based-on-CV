# V19 输入与运行说明

该入口用于已验证传感器模型下的研究验证。当前仅支持声明为 nondispersive_coaxial_tof_proxy 的模拟数据，不能把真实双目相机或尚未验证的 ToF 相机改名后当作已支持设备。

## 一次标定

固定相机位置、内参、分辨率和曝光设置，采集一次空容器的已对齐 RGB、深度。当前算法要求开口容器位于视野中央、底面及液面水平、相机正视。标定区域由空容器实测深度处理获得，不提供液体真值掩码。

## 每次测量

采集当前 RGB 和连续深度帧。深度可为 H×W 或 T×H×W，使用相机导出的 depth_scale_to_m 转为米。运行时保持相机和容器静止；64 帧是本轮实验配置。

必需相机字段：fx、fy、cx、cy、depth_scale_to_m、depth_alignment=color、sensor_model。图像必须已去畸变并与深度对齐。

## 服务器运行示例

在项目目录，使用现有 liquid-depth Python 环境，并令 PYTHONPATH 指向 src：

```bash
export PYTHONPATH="$PWD/src"
/root/autodl-tmp/envs/liquid-depth/bin/python scripts/infer_refraction_v19.py \
  --rgb CURRENT_RGB.png \
  --raw CURRENT_DEPTH_SEQUENCE.npz \
  --empty-rgb EMPTY_REFERENCE.png \
  --empty-depth EMPTY_DEPTH.npz \
  --camera CAMERA.json \
  --output RESULT.json
```

当前深度 npz 键默认 raw，空容器深度 npz 键默认 empty_raw，也支持 npy。模型不会读取 scene.json 或任何标签文件。

默认入口执行折射证据检查。accepted=false 时，正式 thickness_m / internal_depth_m 为空；candidate_thickness_m 和 profile_range_m 仅用于诊断。accepted=true 仍是模型条件下的证据支持，不是绝对液体身份认证；liquid_identity_verified 始终为 false。

--geometry-only 仅用于对照诊断，会绕过单层介质竞争解释，不能把它的通过率当成最终测量可靠性。
--raw-only 用于关闭图像证据的消融。

## 复现实验

依次使用：

1. generate_stratified_refraction_v19.py：Blender Cycles 生成当前与空容器参考。
2. generate_refraction_burst_v19.py：生成明确声明为代理的固定/随机回波序列。
3. evaluate_refraction_v19.py：不输入真值的几何估计，之后才评分。
4. evaluate_refraction_baselines_v19.py：冻结神经网络对照。
5. evaluate_refraction_evidence_v19.py：应用开发集确定的色度与竞争解释检查。
6. stress_refraction_v19.py：传感器模型违反条件下的反例。
7. summarize_refraction_v19.py：汇总表格与图。

完整的种子、冻结参数和源文件哈希见交付包中的 frozen-before-heldout.json、evidence-frozen.json、delivery.json。


## 参考曝光质检

默认入口检查空容器可用底面区域的饱和比例：任一颜色通道达到 0.995 的像素比例超过 1% 时，返回 reference_requires_exposure_adjustment，正式测量为空。标定阶段调整曝光并获得合格参考后，应锁定同一曝光采集液体。

仿真采集脚本支持 --auto-exposure --exposure -2：仅根据空容器预览逐档调整，再以同一设置采集参考和液体。该流程没有使用液体标签；真实相机 SDK 的曝光控制仍需设备联调。
