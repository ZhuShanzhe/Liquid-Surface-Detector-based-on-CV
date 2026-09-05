#!/usr/bin/env python3
"""Write reproducible range-control ablation and calibration reports."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_range_controls import METHODS
from evaluate_range_reacquisition import metrics


def pct(x):
    return "-" if x is None else f"{100 * x:.2f}%"


def number(x):
    return "-" if x is None else f"{x:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--runtime-report", type=Path)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    scored = [r for r in report["frames"] if r["phase"] not in ("calibration", "outage")]
    comparable = [r for r in scored if not r["variant"].startswith("rgb_")]
    totals = {m: metrics(comparable, m) for m in METHODS}
    variants = {
        v: {m: metrics([r for r in scored if r["variant"] == v], m) for m in METHODS}
        for v in sorted({r["variant"] for r in scored})
    }
    lines = [
        "# 距离噪声门控、近远端点可靠性标定与 RGB 误接受控制",
        "",
        "日期：2026-09-05；v4 实验版本。沿用 v8 通用网络权重，本轮是后处理、标定和拒绝机制优化，不是重新训练网络。",
        "",
        "## 范围与实现",
        "",
        "- 95%–100% 失效保持拒绝，不开展失效期间恢复优化。",
        "- 距离噪声：仅使用开发种子 10607 的 576 个普通场景帧，按传感器族拟合 sigma(z)=a+b*z²；按相对误差加权，避免远端噪声支配近端截距。z 是预测液面有效原始深度的中位数（光轴距离），不是传入真实液深。",
        "- 平面残差门限改为 min(250 mm, max(4 mm, 1.5*sigma(z)))，保留 12° 倾角、新鲜点数量与空间支撑约束，增加至少 60% 拟合内点要求。只对启用标定的路径生效。",
        "- 近远端标定：距离分箱×模型分数分箱估计原始深度点为液面内点的可靠性；原始深度与标注误差不超过 3*sigma+1 mm 且属于真实液面才为正例。使用保守下界 0.90、样本量至少 128 选点，空分箱和范围外拒绝，不再统一用原始分数 0.3。",
        "- RGB 分辨率：在原有轮廓匹配之外，对轮廓腐蚀/膨胀 1 像素，估计液位敏感度；取它与匹配平台宽度代理的较大值，再加标定液深误差预算（默认 1 mm，可配置）。",
        "- 严格校验：独立 RGB 误差预算必须小于液位容差，深度/RGB 差值只能使用剩余预算。RGB 越不确定，不再越容易通过。",
        "",
        "标定概率仅表示仿真域中的原始深度点支持可靠性，不等于液深达到精度目标的概率。分箱像素存在相关性，Wilson 形式的保守下界不能当作独立采样得到的统计保证；RGB 误差预算也不是经覆盖率校准的置信区间。",
        "",
        "## 验证设计",
        "",
        f"冻结开发集标定后评估种子 10709，共 {report['evaluated_frames']} 帧条件。它未参与参数拟合，但此前已经用于 v3 审计，因此属于回归检查集，不是全新盲测。",
        "沿用 320×180 有色可见液面、已知椭圆容器几何、底面、内参和位姿。第一帧人工 ROI/已知液深用于 RGB 初始化；后续真值只用于评价、故障注入和离线可靠性评分。",
        "普通、75%/90% 空洞、缓慢回波与恢复测试和 v3 对齐。另对 ToF 组增加亮度降至 35%、中心矩形遮挡、RGB 水平偏移 3 像素的反例。仍未覆盖真实透明液体及任意容器。",
        "容差为 max(5 mm, 真值液深×2%)。误接受率=越界帧/已输出帧；覆盖率=已输出帧/全部待评估帧。初始化与失效期间不计入下表，拒绝不计零误差。",
        "",
        "## 逐项消融：同一组可比较帧",
        "",
        "| 路径 | 覆盖率 | 已输出越界率 | MAE/mm | P95误差/mm | 已输出AbsRel | 处理P95/ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "v3": "v3 独立校验",
        "noise": "+距离噪声门控",
        "calibrated": "+近远端点可靠性标定",
        "strict": "+严格RGB分辨率控制",
        "geometry": "标定几何路径（无RGB校验）",
    }
    for m, v in totals.items():
        lines.append(
            f"| {labels[m]} | {pct(v['coverage'])} | {pct(v['false_accept_given_output'])} | {number(v['mae_mm'])} | {number(v['p95_mm'])} | {pct(v['abs_rel'])} | {number(v['latency_p95_ms'])} |"
        )
    lines += [
        "",
        "处理延时包含同帧模型推理及对应校验/几何计算，不包含采集、磁盘读取、界面和真实位姿求解。无 RGB 路径仅作消融，不可宣称能独立排除错误回波。",
        "",
        "## 分场景严格路径结果",
        "",
        "| 场景 | v3覆盖率 | 标定后宽松覆盖率 | 严格覆盖率 | 严格越界率 | 严格已输出MAE/mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v, items in variants.items():
        lines.append(
            f"| {v} | {pct(items['v3']['coverage'])} | {pct(items['calibrated']['coverage'])} | {pct(items['strict']['coverage'])} | {pct(items['strict']['false_accept_given_output'])} | {number(items['strict']['mae_mm'])} |"
        )
    lines += [
        "",
        "## 普通场景各距离覆盖率",
        "",
        "| 相机距离/液深(m) | 传感器 | v3 | 噪声门控 | 点标定后 | 严格RGB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    normal = [r for r in scored if r["variant"] == "normal"]
    for d, h, sensor in sorted({(r["standoff_m"], r["truth_m"], r["sensor"]) for r in normal}):
        part = [r for r in normal if (r["standoff_m"], r["truth_m"], r["sensor"]) == (d, h, sensor)]
        values = [pct(metrics(part, m)["coverage"]) for m in METHODS[:4]]
        lines.append(f"| {d:g}/{h:g} | {sensor} | " + " | ".join(values) + " |")
    lines += [
        "",
        "## 点可靠性标定检查",
        "",
        "以下 Brier 与 ECE 针对“原始深度点为内点”这一事件，不是液深误差。原模型分数并非为同一事件专门标定，比较仅衡量作为选点分数使用时的匹配程度。分距离列出可避免总体均值掩盖端点退化。",
        "",
        "| 传感器/相机距离 | 原分数Brier | 标定Brier | 原分数ECE | 标定ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, v in sorted(report["confidence_metrics"].items()):
        lines.append(
            f"| {key} | {v['original_score']['brier']:.4f} | {v['range_calibrated']['brier']:.4f} | {v['original_score']['ece']:.4f} | {v['range_calibrated']['ece']:.4f} |"
        )
    rejection = Counter(reason for r in comparable for reason in r["strict"]["reasons"])
    lines += ["", "## 拒绝原因与恢复", ""]
    for reason, count in rejection.most_common():
        lines.append(f"- {reason}: {count} 帧。")
    lines += [
        "",
        "恢复试验为 6 秒深度中断，另有 4 cm 液位变化；下表只统计恢复后 12 帧。0 输出表示拒绝，不得解释为恢复成功。",
        "",
        "| 序列/情况 | v3恢复覆盖率 | 严格恢复覆盖率 | 严格首次恢复延时/s |",
        "|---|---:|---:|---:|",
    ]
    for motion in ("recovery_static", "recovery_changed"):
        for variant in ("recovery", "recovery_bad_echo"):
            rows = [
                r
                for r in scored
                if r["motion"] == motion and r["variant"] == variant and r["phase"] == "reacquisition"
            ]
            accepted = [r for r in rows if r["strict"]["accepted"]]
            delay = (accepted[0]["index"] - 68) / 10 if accepted else None
            lines.append(
                f"| {motion}/{variant} | {pct(metrics(rows, 'v3')['coverage'])} | {pct(metrics(rows, 'strict')['coverage'])} | {number(delay)} |"
            )
    lines += [
        "",
        "## 使用与边界",
        "",
        "实验接口支持 range_profile（标定 JSON 路径）、sensor_family 和 strict_rgb=True；严格模式必须提供已标定 RGBContourWitness。标定文件绑定网络权重 SHA256，不匹配会报错。修改网络后必须重新拟合点可靠性。",
        "RGBContourWitness(calibration_error_m=...) 可传入人工参考液深的误差预算，保存/恢复时保留；不能在现场不满足时沿用默认 1 mm。",
        "这些控制保持可选，未自动替换工业操作面板默认路线。严格模式因分辨率不足拒绝时，不应自动回退宽松路线并继续标为可信。可以提示操作员改善视角、增加液面像素或重新标定。",
        "0.1–10 m 的仿真传感器噪声不能视为真实设备覆盖该范围的证明。数米端点的均值误差仍受理想标定和零均值噪声平均影响；必须同时检查覆盖率与误接受。",
        "复现顺序：fit_range_calibration.py → evaluate_range_controls.py → summarize_range_controls.py。原始序列、逐帧指标和日志保留在服务器。",
        "",
    ]
    if "highres_rgb" in report:
        high = metrics(comparable, "strict_highres")
        totals["strict_highres"] = high
        lines += [
            "",
            "## 真正高分辨率 RGB 独立校验追加对照",
            "",
            "为避免把插值当作信息增益，重新渲染 1280×720 RGB。主模型输入、原始深度、噪声标定与全部门限不变；只替换独立 RGB 校验的图像和对应内参。高分辨率首帧重新执行同一人工标注初始化。",
            "静态序列复用同一静态 RGB 渲染，液位变化序列渲染变化前后状态。RGB 缓存中的重复高分辨率深度文件没有进入推理，原始带逐帧噪声的 320×180 深度仍取自 v3 序列。评估器只缓存完全相同的 RGB/位姿校验结果，并计入实测未缓存校验耗时；真实封装不使用此缓存。",
            f"可比帧整体：覆盖率 {pct(high['coverage'])}，已输出越界率 {pct(high['false_accept_given_output'])}，MAE {number(high['mae_mm'])} mm，AbsRel {pct(high['abs_rel'])}，处理耗时合成 P95 {number(high['latency_p95_ms'])} ms。",
            "",
            "| 场景 | 320严格覆盖率 | 1280严格覆盖率 | 1280已输出越界率 | 1280 MAE/mm |",
            "|---|---:|---:|---:|---:|",
        ]
        for variant, items in variants.items():
            subset = [r for r in scored if r["variant"] == variant]
            hm = metrics(subset, "strict_highres")
            items["strict_highres"] = hm
            lines.append(
                f"| {variant} | {pct(items['strict']['coverage'])} | {pct(hm['coverage'])} | {pct(hm['false_accept_given_output'])} | {number(hm['mae_mm'])} |"
            )
        lines += [
            "",
            "| 相机距离/液深(m) | 传感器 | 1280普通覆盖率 | 已输出MAE/mm | 越界率 |",
            "|---|---|---:|---:|---:|",
        ]
        for d, h, sensor in sorted({(r["standoff_m"], r["truth_m"], r["sensor"]) for r in normal}):
            subset = [r for r in normal if (r["standoff_m"], r["truth_m"], r["sensor"]) == (d, h, sensor)]
            hm = metrics(subset, "strict_highres")
            lines.append(
                f"| {d:g}/{h:g} | {sensor} | {pct(hm['coverage'])} | {number(hm['mae_mm'])} | {pct(hm['false_accept_given_output'])} |"
            )
        lines += ["", "| 恢复情况 | 1280恢复覆盖率 | 首次恢复延时/s | 已输出MAE/mm |", "|---|---:|---:|---:|"]
        for motion in ("recovery_static", "recovery_changed"):
            for variant in ("recovery", "recovery_bad_echo"):
                rows = [
                    r
                    for r in scored
                    if r["motion"] == motion and r["variant"] == variant and r["phase"] == "reacquisition"
                ]
                hm = metrics(rows, "strict_highres")
                accepted = [r for r in rows if r["strict_highres"]["accepted"]]
                delay = (accepted[0]["index"] - 68) / 10 if accepted else None
                lines.append(
                    f"| {motion}/{variant} | {pct(hm['coverage'])} | {number(delay)} | {number(hm['mae_mm'])} |"
                )
        lines += [
            "",
            "接口：process(..., witness_frame={rgb_bgr: 高分辨率图像, intrinsics: 对应内参, camera_to_world_cv: 对应位姿, synchronized: True})；这里的键在 Python 中须使用字符串。调用方必须实际保证同步和标定，synchronized 标志不是自动同步算法。",
            "RGB 图像仅位移的反例仍保留正确深度，因此通过帧不等于检测到了位姿偏差；本测试仅检查最终液深误接受。暗光/遮挡下拒绝也不是恢复能力。",
            "零观测越界不等于真实误接受概率为零；帧之间相关，仍需独立场景、未知容器和真实系统偏差验证。",
        ]
    lines += [
        "",
        "## 本轮结论与保留问题",
        "",
        "距离噪声门控解决了部分数米观测被固定 12 mm 门限误拒绝的问题，但单独放宽会增加误接受，不能作为完整方案直接替换。点可靠性标定使 10 m ToF 仿真从空间支撑不足变为可输出，也使 0.2 m 主动双目从可输出退化为拒绝；这是现有分箱校准/样本分布的不足，不能只宣传远端收益。",
        "0.1 m 主动双目与结构光的标定 Brier 反而变差，几何约束仍在拒绝这些帧。这说明近端错误回波造成的距离估计/分箱归属和类别比例变化尚未解决；该端点不应宣称可靠标定完成。",
        "真正高分辨率 RGB 恢复了大量因低分辨率预算不足而拒绝的正常输出，同时保留错误回波拒绝。当前更适合作为可选高可信路径，而不是承诺全量程无条件输出的默认路线。",
        "仍需优化：①近端传感器有效距离识别及分箱分布漂移检测；②数米双目/结构光的稳健几何支持和校准样本多样性；③远相机/浅液深的独立参考分辨率（当前仍拒绝）；④透明、无色、暗光、遮挡下可用的替代独立观测。95%–100% 失效期间不做恢复优化。",
        "绝对 MAE 可能因新增远端输出而上升，应同时比较液深 AbsRel、容差越界率与覆盖率。零越界只描述本次有限、相关的仿真输出，不是工业误接受率为零的保证。",
        "完整启用和复现步骤见 [接口使用说明](range_controls_v4_usage.md)。",
    ]
    runtime = None
    if args.runtime_report:
        runtime = json.loads(args.runtime_report.read_text())
        lines += [
            "",
            "## 未缓存的封装实测",
            "",
            f"同一已标定仿真帧连续调用真实封装 {runtime['frames']} 次，逐次执行高分辨率 RGB 校验，没有缓存；第 {runtime['first_accepted_index'] + 1} 帧开始输出，共输出 {runtime['accepted']} 帧。处理 P95 {runtime['p95_ms']:.2f} ms、最大 {runtime['max_ms']:.2f} ms，末帧液深 {runtime['last_level_m']:.6f} m。",
            "这是服务器模型+算法处理时延，不包含真实摄像头采集、传输、界面和位姿求解；此实测在评估任务结束后执行。",
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    compact = {k: v for k, v in report.items() if k != "frames"}
    compact["runtime_smoke"] = runtime
    compact.update(comparable_totals=totals, variant_totals=variants, strict_rejection_counts=dict(rejection))
    args.output.with_suffix(".json").write_text(json.dumps(compact, indent=2, allow_nan=False))
    print(json.dumps({"comparable": totals, "variants": variants}, indent=2))


if __name__ == "__main__":
    main()
