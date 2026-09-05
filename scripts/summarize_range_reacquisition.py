#!/usr/bin/env python3
"""Audit tables for range, independent echo verification and reacquisition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_range_reacquisition import metrics


def fmt(v):
    return "-" if v is None else f"{v:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.report.read_text())
    held = [r for r in p["frames"] if r["seed"] == 10709]
    normal = [r for r in held if r["variant"] == "normal" and r["phase"] == "evaluation"]
    cases = sorted({(r["standoff_m"], r["truth_m"]) for r in normal})
    lines = [
        "# 全量程、缓慢错误回波独立校验与失效后重新确认",
        "",
        "日期：2026-09-05。版本：v3 实验接口。未修改网络权重，增加独立 RGB 校验和恢复状态管理。",
        "",
        "## 已固定的任务边界",
        "",
        "95%–100% 严重深度失效不再作为恢复优化目标。新验证路径在预测液面区域有效深度占比不超过 5% 时拒绝输出。这里的分母是预测液面区域，并非真值掩码。长时失效序列只验证数据恢复之后的重新确认；失效期间不进行伪深度补全。",
        "",
        "## 实验设计与适用前提",
        "",
        f"生成 {p['rendered_frames']} 张 RGB-D 图像，评估 {p['evaluated_frames']} 帧条件组合；10607 为开发场景种子，10709 为独立检查种子。三个传感器族使用项目已有噪声和错误回波仿真参数，不能等同于真实设备认证。",
        "相机到液面距离与实际液深分别控制：覆盖 0.1、0.2、0.5、1、2、3、4、6、8、10 m 距离节点，并加入远相机/浅液深和近相机/深容器组合。所有节点仍由同一个 v8 通用 RGB-D 模型推理，没有按量程切换网络。",
        "每个静止组合 16 帧，其中前 5 帧作初始化确认，后 11 帧评分。该数量是初步范围审计，尚未满足多次独立工业测量的统计验证要求。图像分辨率 320×180，RGB 使用 Eevee，液体为有颜色的可分辨表面。",
        "相机内参、重力、固定容器底面、椭圆截面半径及当前相机位姿已知。RGB 校验额外使用第一张图像的人工液面 ROI 与已知液深作一次标定；仿真以第一帧标注模拟该操作。后续帧的真值只用于评分和构造传感器故障，不进入推理。",
        "",
        "## 普通场景量程审计",
        "",
        "各单元为原有时序几何路径的输出覆盖率 / 已输出液深 MAE(mm)。零覆盖率以缺失误差表示，不能解释为零误差。",
        "",
        "| 相机距离/m | 液深/m | 主动双目 | 结构光 | ToF |",
        "|---:|---:|---|---|---|",
    ]
    compact = {}
    for distance, level in cases:
        values = []
        for sensor in ("active_stereo", "structured_light", "tof"):
            subset = [
                r
                for r in normal
                if r["standoff_m"] == distance and r["truth_m"] == level and r["sensor"] == sensor
            ]
            m = metrics(subset, "baseline")
            compact[f"d{distance:g}_h{level:g}/{sensor}"] = {
                method: metrics(subset, method) for method in ("network", "baseline", "verified")
            }
            values.append(f"{m['coverage'] * 100:.1f}% / {fmt(m['mae_mm'])}")
        lines.append(f"| {distance:g} | {level:g} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "这轮数据不能支持“整个量程均已达标”。尤其主动双目/结构光在数米距离下出现大量拒绝，10 m 端点三类仿真都未稳定输出。ToF 部分节点的亚毫米均值来自已知完美标定、平面大样本平均和仿真中较弱的独立零均值噪声，不能据此宣传真实 8 m 测量达到亚毫米。",
            "",
            "原因检查：在独立场景第 5 帧，1 m 结构光的平面中位残差约 13.50 mm，2 m 主动双目约 24.33 mm，均超过现有固定 12 mm 拒绝阈值。现有双目/结构光模拟的深度噪声随距离显著放大；固定残差门限未按噪声尺度归一化。",
            "0.1 m 主动双目检查帧拟合倾角约 53.67°、残差约 23.04 mm，显示近端错误回波/有效支持出现异常，并非能够归结为正常高斯测量噪声。10 m ToF 检查帧模型液面内中位置信度约 0.279，低于 0.3 初筛阈值，出现空间支撑不足。这两端还需分别审查仿真近距离退化与模型边界行为。",
            "全量程下一阶段可以优化噪声归一化的平面门控、近远端置信度校准与训练采样；门限放宽后必须同时检查误接受率，不能仅用提高覆盖率作为成功标准。",
            "",
            "## RGB 独立校验",
            "",
            "RGBContourWitness 使用初始化图像建立液体颜色模型，再用当前 RGB 分割轮廓，与已标定椭圆容器在不同液位下的投影比较，得到独立米制液位。接口没有深度输入，未使用受污染 RGB-D 模型的掩码来生成这一路校验量。它依赖已知容器几何、当前位姿和可见颜色/轮廓；尚未验证透明、无色或强遮挡场景。",
            "校验量与深度几何结果不一致时拒绝，且失败候选不会写回可靠历史。当前 uncertainty_proxy_m 是轮廓匹配曲线的分辨率指标，并非经覆盖率校准的统计置信区间。",
            "",
            "| ToF 相机距离/液深(m) | 原路径越界占已输出比例 | RGB 校验后输出覆盖率 | 校验后越界占已输出比例 | 首次独立冲突拒绝延时/s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    drift = {}
    for distance, level in cases:
        subset = [
            r
            for r in held
            if r["variant"] == "slow_echo"
            and r["phase"] == "evaluation"
            and r["sensor"] == "tof"
            and r["standoff_m"] == distance
            and r["truth_m"] == level
        ]
        before = metrics(subset, "baseline")
        after = metrics(subset, "verified")
        rejected = [r["index"] for r in subset if "depth_rgb_metric_disagreement" in r["verified"]["reasons"]]
        delay = (min(rejected) - 5) / 10 if rejected else None
        drift[f"d{distance:g}_h{level:g}"] = {"baseline": before, "verified": after, "detection_s": delay}
        lines.append(
            f"| {distance:g}/{level:g} | {fmt(None if before['false_accept_given_output'] is None else before['false_accept_given_output'] * 100)}% | {after['coverage'] * 100:.1f}% | {fmt(None if after['false_accept_given_output'] is None else after['false_accept_given_output'] * 100)}% | {fmt(delay)} |"
        )
    lines.extend(
        [
            "",
            "漂移从第 5 帧开始，每帧增加 3 mm×max(1,相机距离/m) 的深度偏差，目的是构造相关性错误回波反例，不代表设备实际漂移率。低覆盖率主要表示拒绝了受到污染的帧，不应当被称为恢复了正确液深。",
            "仍有漏洞：1 m 相机/0.3 m 液深等条件在漂移初期仍会输出少量越界值；远相机/浅液深时，轮廓分辨率门限可能大于液深容差，独立校验并不能证明毫米级可靠性。下一步应校准独立校验的分辨率上限，或增加已知刻度/接触线、固定目标、第二视角等独立信息。",
            "",
            "## 长时失效后的重新确认",
            "",
            "VerifiedSurfaceTracker 将状态分成获取参考、正常跟踪、失去观测、连续确认。数据恢复后必须有新鲜米制锚点、有效位姿及 RGB 独立一致性，连续 5 帧稳定后建立新参考；此前的历史点不继续支配液位。该机制允许失效期间液位改变，不再依赖旧液位的固定跳变门限。",
            "下表只统计第 68 帧恢复深度之后的 12 帧。按 10 Hz 采样，连续 5 帧确认意味着最早 0.4 s 才恢复输出。这里重新确认的是液位参考；实际相机 6DoF 位姿仍需已有标记/CAD 跟踪或重新标定提供。",
            "",
            "| 情况 | 原路径覆盖率 | 新路径覆盖率 | 新路径首次输出延时/s | 新路径 MAE/mm |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    recovery = {}
    for motion in ("recovery_static", "recovery_changed"):
        for variant in ("recovery", "recovery_bad_echo"):
            subset = [
                r
                for r in held
                if r["motion"] == motion and r["variant"] == variant and r["phase"] == "reacquisition"
            ]
            before = metrics(subset, "baseline")
            after = metrics(subset, "verified")
            accepted = [r for r in subset if r["verified"]["accepted"]]
            delay = (accepted[0]["index"] - 68) / 10 if accepted else None
            recovery[f"{motion}/{variant}"] = {"baseline": before, "verified": after, "delay_s": delay}
            lines.append(
                f"| {motion}/{variant} | {before['coverage'] * 100:.1f}% | {after['coverage'] * 100:.1f}% | {fmt(delay)} | {fmt(after['mae_mm'])} |"
            )
    evaluated = [r for r in held if r["phase"] not in ("calibration", "outage")]
    overall = {method: metrics(evaluated, method) for method in ("network", "baseline", "verified")}
    lines.extend(
        [
            "",
            "## 独立检查集汇总与指标定义",
            "",
            "以下合并普通、75%/90% 空洞、缓慢错误回波和恢复后条件，排除初始化与失效期间；不是单独缓慢回波指标。",
            "容差为 max(5 mm, 真实液深的 2%)；越界比例以已输出帧为分母，覆盖率以全部待评估帧为分母。AbsRel 为已输出液深绝对误差除以真实液深后取均值。拒绝帧没有计作零误差。",
            f"共 {len(evaluated)} 帧。原路径覆盖率 {overall['baseline']['coverage'] * 100:.2f}%，越界占已输出比例 {overall['baseline']['false_accept_given_output'] * 100:.2f}%；新路径覆盖率 {overall['verified']['coverage'] * 100:.2f}%，越界占已输出比例 {overall['verified']['false_accept_given_output'] * 100:.2f}%。",
            "改善主要来自拒绝不可靠结果，不能据此宣称全量程高覆盖已达标。",
            "长时失效测试目前仅覆盖 6 秒（60 帧）中断，以及中断期间 4 cm 液位变化；不证明分钟/小时级遮挡、外观变化或相机重新定位已通过验证。",
        ]
    )
    lines.extend(
        [
            "",
            f"新路径 GPU 模型＋独立 RGB 校验＋时序几何的处理 P95 约 {overall['verified']['latency_p95_ms']:.2f} ms；不包含相机采集、磁盘输入、真实位姿求解和显示。模型权重没有重新训练。",
            "",
            "## 接口与后续安排",
            "",
            "可将已标定 RGBContourWitness 传给 UniversalSurfaceVideoSystem 的 rgb_witness 参数。标定可用 to_dict()/from_dict() 保存、恢复；算法与界面保持分离。未启用独立参考的原路径保持可用，不能把它标成经过 RGB 独立校验。",
            "优先级：①审查 0.1 m/10 m 端点和数米双目噪声门控；②对独立 RGB 校验做分辨率、误接受率及外观变化验证；③扩展参考重新确认到真实相机位姿重定位。95%–100% 严重失效期间的恢复优化已从计划移除。",
            "复现：generate_range_sequences.py 生成连续图像；evaluate_range_reacquisition.py 执行同模型推理与三路径对照；summarize_range_reacquisition.py 输出本报告。服务器保存所有图像、逐帧结果与日志，仓库保存代码及紧凑指标。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": p["checkpoint_sha256"],
                "rendered_frames": p["rendered_frames"],
                "evaluated_frames": p["evaluated_frames"],
                "range": compact,
                "drift": drift,
                "recovery": recovery,
                "overall": overall,
                "rejection_counts": dict(
                    Counter(reason for r in evaluated for reason in r["verified"]["reasons"])
                ),
            },
            indent=2,
        )
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
