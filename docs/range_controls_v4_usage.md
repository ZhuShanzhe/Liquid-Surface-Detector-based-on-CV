# v4 距离与独立 RGB 控制：接口及复现

本功能仍是可选实验路线，不自动替换操作面板默认路径。算法与 UI 分离；目前无 USB 相机测试只证明服务器处理代码可用。结果和限制见 [实验报告](range_controls_v4.md)。

## 启用条件

1. 主 RGB-D 已配准，深度单位为米；相机到容器世界坐标变换、重力及底面已标定。
2. 选择与相机类型匹配的传感器族：active_stereo、structured_light 或 tof。本仓库配置只来自仿真，真实相机须重新采样拟合，不能直接作为设备标定证书。
3. 独立 RGB 使用可见液面轮廓和已知椭圆容器几何。首帧由操作员标注液面 ROI 和已知液深；不支持未经验证的任意透明、无色、遮挡容器。
4. 若使用高分辨率独立图像，使用真正的原生 RGB 采集，不插值假装增加信息；提供相应内参和相机位姿，并保证与当前 RGB-D 同步。
5. calibration_error_m 设置为人工参考液深的真实误差预算。默认 0.001 m 仅是本轮仿真设定，不应在现场无依据沿用。

## Python 集成

以下变量由既有相机/标定模块提供；不需要改变主模型权重。

```python
from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.surface_video_runtime import UniversalSurfaceVideoSystem

witness = RGBContourWitness(calibration_error_m=0.001)
witness.calibrate(
    initialization_rgb_highres,
    operator_liquid_roi_highres,
    known_liquid_level_m,
    rgb_highres_intrinsics,
    rgb_camera_to_world,
    bottom_world_m,
    vessel_radius_x_m,
    vessel_radius_y_m,
)
# witness.to_dict() 可保存；RGBContourWitness.from_dict(payload) 可恢复。

system = UniversalSurfaceVideoSystem(
    checkpoint_path,
    range_profile="configs/range_calibration_v4.json",
    sensor_family="tof",
    rgb_witness=witness,
    strict_rgb=True,
)
result = system.process(
    rgb_registered,
    raw_depth_m,
    depth_intrinsics,
    depth_camera_to_world,
    bottom_world_m,
    pose_valid=True,
    witness_frame={
        "rgb_bgr": current_rgb_highres,
        "intrinsics": rgb_highres_intrinsics,
        "camera_to_world_cv": rgb_camera_to_world,
        "synchronized": True,
    },
)
if result["accepted"]:
    display_liquid_depth(result["level_m"])
else:
    display_rejection(result["reasons"])
```

不要把 rejected 的 candidate_level_m 当作正式液深；也不要在严格模式拒绝后无提示切换到宽松输出。state=confirming 表示尚未完成连续 5 帧参考确认。相机移动后必须有经过确认的新位姿，或重新标定并调用 reset_reference()。

range_profile 绑定网络权重哈希。更换模型后需要重新拟合标定；模型本身仍是统一量程模型，分箱只用于选点可靠性和诊断。若只做噪声门控消融，可在 memory_options 中传入 RangeNoiseCalibration(payload, sensor_family, calibrate_confidence=False)，不要与 range_profile 同时指定。

## 服务器复现

项目目录：

`/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV`

运行环境：

`/root/autodl-tmp/envs/liquid-depth/bin/python`

对应脚本及参数：

- fit_range_calibration.py：--data 指向 range_reacquisition_v3_data，--checkpoint 指向 v8 best.pth，--output 指向标定 JSON。仅拟合种子 10607。
- evaluate_range_controls.py：使用同一 --data、--checkpoint、--profile，并指定 --output。评估种子 10709 的低分辨率五路径消融。
- Blender 运行 generate_range_sequences.py：--seeds 10709 --rgb-scale 4 --cache-static-rgb，--output 使用独立 RGB 数据目录。该缓存不能当作原始深度视频训练集。
- evaluate_highres_rgb_control.py：--report 指向前一步低分辨率结果，--rgb-data 指向高分辨率目录，传入同一 --profile、--checkpoint 和新的 --output。
- summarize_range_controls.py：--report 使用高分辨率追加结果，--output 为 Markdown 报告；可附加 --runtime-report 记录未缓存的真实封装延时。

本轮服务器成果均在 `/root/autodl-tmp/liquid-depth-artifacts/evaluation`：

- range_reacquisition_v3_data：原始 RGB-D 序列。
- range_rgb_1280_v4：真正高分辨率 RGB 参考渲染。
- range_controls_v4.json：低分辨率逐帧消融。
- range_controls_highres_v4.json：加入高分辨率独立校验的逐帧结果。
- range_v4_runtime_smoke.json：20 次未缓存封装调用的处理延时。

日志在同一 artifacts 目录的 logs 子目录。所有数据留在服务器，仓库只保存代码、标定参数和紧凑报告。
