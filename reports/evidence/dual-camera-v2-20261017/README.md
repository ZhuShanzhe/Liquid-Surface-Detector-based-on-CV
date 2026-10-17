# 历史双相机V2统计与复现证据（非当前入口）

本目录保留，用于开发过程比较与选择ToF的依据。当前正式版见 `../tof-formal-v1-20261017/`。旧源码锁、验证及PDF清单对应提交 `15bfe99`，不是当前文件列表。旧链路与远浅专题已从当前发布目录撤下；性能PDF原样保留。

当前数据仅含 `tof` / `structured_light`。`active_stereo` 与旧 `stereo_IR_image_proxy` 没有重新贴标签混入。

- `current-dual-predictions.jsonl.gz`：11,390条预测、真实底面相对液深、场景/相机/程度、安装聚类和来源，部分包含在线诊断。
- `metrics.json`：87个场景统计单元、液深与最高高度共99组指标。键为 `scene|sensor|level|subset`，不按实际液深分层；`far_shallow`为专项离线范围。
- `evaluation-audit.json`：报告配额、小类型最低150帧、输入排除、300次相位标定、旧/新数据量及源哈希。
- `final-holdout-source-lock.json`：最终相位测试开始前冻结的代码；不包含后续文档排版或统计展示脚本。
- `verification.json`：八场景×两相机16个实际CLI回放与统计复核。
- `far-failure-cases.json`：远浅每场景/相机最大的几个误差案例，仅用于事后诊断，不用于挑帧、调参或门控。
- `pytest-final.log`：754项通过，14条依赖警告。
- `pdf-qa.json`：三份PDF页数、目录页码/链接及页面安全边界检查；另有人眼PNG检查。
- `release_manifest.json`：代码/证据/PDF哈希与发布边界。

在新机器只需NumPy即可独立重算数据：

```bash
git show 15bfe99:scripts/recount_dual_camera_v2.py > /tmp/recount_historical_dual.py
python /tmp/recount_historical_dual.py --evidence reports/evidence/dual-camera-v2-20261017
```

完整源观测位于服务器 `/root/autodl-tmp/liquid-depth-data/synthetic/dual-camera-v2-20261017/`：`far/` 为新25组安装的Blender RGB/底面/浮物几何和双机制深度代理；`phase-final/` 为最终独立相位观测。`dev-phase/` 与 `phase-holdout/` 是开发数据/初测，不属于最终1200帧。初测发现结构光散射相位定义错误后保留原记录，修正并换独立种子再测，未将初测好帧拼入最终结果。

`far/` 新有效采集为每安装/条件首次满足双相机输入支持的4组噪声；候选上限64次，来源审核不看数值误差。由普通帧受控删点生成的部分失效保留同一安装聚类，不能当成新增独立容器。

保留的6,813旧帧沿用原纳入协议；新增3,377远浅和1,200相位帧使用本轮协议。报告明确采样构成、依赖条件和误差，不把合并数值等同实地通用设备精度。相位实验给定光学几何，未加入其系统误差；低消光不是850/940nm水的实测光谱。全部报告均为研究结果，不是工业计量认证。
