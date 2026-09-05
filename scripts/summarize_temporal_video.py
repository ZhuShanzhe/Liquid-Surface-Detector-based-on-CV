#!/usr/bin/env python3
"""Produce auditable scenario tables from temporal video evaluation output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_temporal_video import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--drift-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = report["frames"]
    held = [r for r in rows if r["seed"] == 9323]

    def fmt(value, scale=1):
        return "-" if value is None else f"{value * scale:.2f}"

    lines = [
        "# 大面积深度失效：连续仿真视频验证与时序优化",
        "",
        "日期：2026-09-05。状态：实验分支，可通过独立接口调用，尚未提升为默认生产路线。",
        "",
        "## 实验范围",
        "",
        f"生成 {report['data_frames']} 张连续渲染 RGB-D 图像，回放 {report['evaluated_frames']} 帧条件组合。场景覆盖圆柱与矩形容器；9107、9212 用于开发及消融，9323 在冻结方案后生成，作为最终独立场景检查。大量相邻帧高度相关，不能当成相同数量的独立实验。",
        "相机位于容器上方，约 1.1 m 安装高度；液深主要为 0.30–0.36 m。普通序列 36 帧、长序列 120 帧，按 10 Hz 采样；常规空洞持续 2 s，长序列 95% 空洞持续 11.2 s。",
        "使用 v8 通用 RGB-D 模型真实推理，输入与训练一致的 RGB 归一化、对数米制深度和有效性。RGB 由 Blender Eevee 渲染，传感器深度使用已有仿真噪声，再施加连续区域失效。没有使用真值掩码筛选模型输出；真值掩码只用于构造传感器故障及计算实际失效率。",
        "内参、重力、容器底面及相机位姿由仿真标定提供；另注入 25 mm 位姿偏差作为压力测试。本轮不等价于视觉位姿估计已经验证，也不覆盖全部 0.1–10 m 量程或所有透明材质。",
        "",
        "## 本轮实现",
        "",
        "新增米制液面记忆路径：模型液面区域 → 腐蚀边缘 → 当前有效深度锚点 → 重力坐标下鲁棒平面 → 历史点按当前液位增量更新 → 位姿重投影、光流与 RGB 外观检查 → 融合与可观测性拒绝 → 相对已标定底面的液深。",
        "历史点最多保留 45 帧，保存 4 组有足够空间覆盖的直接观测；每个空间单元只计一次，历史占比不超过 75%。当前至少需 24 个可靠点；有合格历史时允许较小的当前空间覆盖，但至少需要 64 个有效历史点。完全没有当前米制证据时拒绝输出。",
        "液位按当前可靠点估计的高度变化更新历史，避免缓慢加液时拖向旧平面。相邻已接受结果相差超过 15 mm 时锁定拒绝，必须由可靠的新参考重新初始化。这是可配置的运行约束，不是能够自动区分真实液位跃升和错误回波的识别器。",
        "",
        "## 独立场景：失效窗口内结果",
        "",
        "开发集消融显示无条件记忆会轻微降低静止场景精度，最终改成按需启用：当前可靠点的空间支撑足够时直接使用当前估计；只有支撑不足时才启用历史恢复。",
        "",
        "",
        "容差为 max(5 mm, 真值液深×2%)。MAE、AbsRel、通过率只针对被接受输出，覆盖率分母是窗口内全部帧。无输出记为缺失，不记作零误差。",
        "",
        "| 运动 | 注入失效条件 | 实际有效深度占比 | 输出覆盖率 | 液深 MAE/mm | AbsRel/% | 容差通过率 | 困难场景工程门槛 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    compact = {}
    for motion in ("static", "rising", "moving", "step", "long_static"):
        for variant in (
            "normal",
            "drop75",
            "drop90",
            "drop98",
            "total",
            "long95",
            "wrong_echo",
            "pose_error",
            "cold_total",
        ):
            subset = [
                r
                for r in held
                if r["motion"] == motion and r["variant"] == variant and r["phase"] == "failure"
            ]
            if not subset:
                continue
            metrics = summarize(subset)
            compact[f"{motion}/{variant}"] = metrics
            m = metrics["surface_memory"]
            qualified = (
                m["abs_rel"] is not None
                and m["abs_rel"] <= 0.03
                and m["tolerance_pass_rate"] >= 0.75
                and m["coverage"] >= 0.8
            )
            lines.append(
                f"| {motion} | {variant} | {np.mean([r['raw_surface_valid_ratio'] for r in subset]) * 100:.2f}% | {m['coverage'] * 100:.1f}% | {fmt(m['mae_mm'])} | {fmt(m['abs_rel'], 100)} | {fmt(m['tolerance_pass_rate'], 100)}% | {'满足本窗口数值门槛' if qualified else '未满足/应拒绝'} |"
            )
    lines.extend(
        [
            "",
            "## 消融：记忆的独立贡献",
            "",
            "以下均为独立场景、90% 空洞的失效窗口。guarded_fresh 与 surface_memory 使用相同跳变门控，仅后者使用历史点；因此可以分离门控和记忆的效果。",
            "",
            "| 运动 | 方法 | 覆盖率 | MAE/mm | P95/mm |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for motion in ("static", "rising", "moving"):
        for method, m in compact[f"{motion}/drop90"].items():
            lines.append(
                f"| {motion} | {method} | {m['coverage'] * 100:.1f}% | {fmt(m['mae_mm'])} | {fmt(m['p95_mm'])} |"
            )
    rec = [
        r for r in report["recovery"] if r["sequence"].startswith("9323_") and r["method"] == "surface_memory"
    ]
    lines.extend(
        [
            "",
            "## 恢复与延时",
            "",
            "恢复时间定义为故障结束后首次连续 3 帧被接受且在容差内的确认时间；未在观察窗口恢复的记录保留为空，不赋予 0。",
            "",
            "| 序列 | 故障 | 恢复确认时间/s |",
            "|---|---|---:|",
        ]
    )
    for r in rec:
        if r["variant"] in ("drop90", "total", "wrong_echo") and "long_static" not in r["sequence"]:
            lines.append(f"| {r['sequence']} | {r['variant']} | {fmt(r['recovery_confirmation_s'])} |")
    overall = summarize(held)
    lines.extend(
        [
            "",
            f"独立场景模型推理加记忆/几何处理 P95：{overall['surface_memory']['latency_p95_ms']:.2f} ms（320×180、模型常驻 GPU；不包括渲染、磁盘读取、相机采集、位姿求解和界面显示）。不能把它直接视为现场相机到显示的端到端延时。",
            "",
            "## 仍未解决的问题",
            "",
            "1. 全失效、初始化就失效、98% 失效及长期高度局部化的残余点：主要靠拒绝保护，没有证明能够恢复有效液深。",
            "2. 真实 60 mm 液位突变：会触发与错误回波相同的锁定，覆盖率下降。需要来自液位接触线、泵/阀流量、独立姿态或其他传感信息支持重新确认，不能通过放宽门限直接解决。",
            "3. 稳定但错误的回波层、缓慢系统漂移、错误标定：几何平整和时序一致都可能成立。只有独立绝对证据才能排除；本轮跳变门控只处理突变。",
            "4. 本轮条件较窄，尚需扩展真实传感器误差形态、低照度、多层透明、波动和漂浮物。结果不能推出全量程毫米级工业精度。",
            "5. 模型置信度用于初筛，尚未在此新序列任务中作概率校准；拟合残差不能代替测量不确定度。可评估输出率在仿真中高，是因为真值齐全，不代表现场可观测性。",
            "",
            "## 调用与复现",
            "",
            "模块与界面分离：使用 liquid_depth.surface_video_runtime.UniversalSurfaceVideoSystem，逐帧调用 process(rgb_bgr, raw_depth_m, intrinsics, camera_to_world_cv, bottom_world_m)。深度单位必须为米，位姿采用 OpenCV 相机坐标到固定重力世界坐标的 4×4 变换。相机移动时必须输入当前已验证位姿。",
            "超过配置跳变限制后，只有新参考得到确认才能调用 reset_reference()。缺失值必须由 accepted/reasons 呈现，不能把拒绝帧显示成最新有效测量。",
            "复现脚本：scripts/generate_temporal_sequences.py（Blender）、scripts/evaluate_temporal_video.py（推理评估）、scripts/summarize_temporal_video.py（本报告）。报告旁的指标 JSON 保留 checkpoint SHA256、分场景消融和恢复结果；服务器保留逐帧 JSON 与 RGB-D 数据。",
        ]
    )
    if args.drift_report:
        drift = json.loads(args.drift_report.read_text())
        subset = [
            r
            for r in drift["frames"]
            if r["seed"] == 9323
            and r["motion"] == "static"
            and r["phase"] == "failure"
            and r["variant"] == "slow_echo"
        ]
        m = summarize(subset)["surface_memory"]
        lines.extend(
            [
                "",
                "## 附加反例：缓慢错误回波",
                "",
                f"在深度上每帧增加 3 mm 偏差，独立静止场景中覆盖率 {m['coverage'] * 100:.1f}%，MAE {fmt(m['mae_mm'])} mm，已输出结果的越界比例 {fmt(m['false_accept_given_output'], 100)}%。这直接说明跳变门控无法覆盖缓慢漂移，不能宣布大面积失效问题已整体解决。",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines.extend(
        [
            "",
            f"记忆实际触发 {overall['surface_memory']['memory_activated_frames']} 帧；触发子集 P95 为 {fmt(overall['surface_memory']['memory_active_p95_ms'])} ms，全部帧最大处理时间 {fmt(overall['surface_memory']['latency_max_ms'])} ms。",
            "",
            "![独立场景时序曲线；红色区域表示拒绝输出](assets/temporal_depth_failure_v2.png)",
            "",
        ]
    )
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint": report["checkpoint"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "data_frames": report["data_frames"],
                "evaluated_frames": report["evaluated_frames"],
                "scope": report["scope"],
                "heldout": overall,
                "scenarios": compact,
                "recovery": rec,
                "drift_stress": m if args.drift_report else None,
            },
            indent=2,
        )
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
