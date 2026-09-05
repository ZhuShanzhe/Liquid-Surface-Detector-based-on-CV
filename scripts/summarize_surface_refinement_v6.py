#!/usr/bin/env python3
"""Generate the v6 audit report from completed result artifacts (never partial logs)."""

import argparse
import hashlib
import json
from pathlib import Path


def number(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--long-file", type=Path)
    args = p.parse_args()
    names = [
        "surface_refinement_v6_final",
        "sensor_components_v6_final",
        "rgb_continuous_v6_regression",
        "rgb_continuous_v6_r03",
        "rgb_continuous_v6_r06",
        "surface_refinement_v6_latency",
    ]
    inputs = {name: args.root / (name + ".json") for name in names}
    if args.long_file:
        inputs["surface_refinement_v6_final"] = args.long_file
    data = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    for row in data["sensor_components_v6_final"]["rows"]:
        if row["sensor_family"] not in ("active_stereo", "structured_light", "tof"):
            raise ValueError("Sensor metadata was overwritten by a method result")
    long = data["surface_refinement_v6_final"]
    for row in long["frames"]:
        if row.get("sensor_family") not in ("active_stereo", "structured_light", "tof"):
            raise ValueError("Invalid long-sequence sensor-family metadata")
    if long["frames_count"] != 36000:
        raise ValueError("Expected complete 36000-frame paired regression")
    payload = {
        "schema_version": 1,
        "sources": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in inputs.items()
        },
        "results": {
            name: {k: v for k, v in item.items() if k not in ("frames", "rows")}
            for name, item in data.items()
        },
        "limitations": [
            "unverified research candidates",
            "known geometry/pose",
            "simulation-only noise parameters",
            "wave intervals are conditional envelopes, not calibrated confidence intervals",
            "legacy v5 data are regression data, not a blind test",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "surface_refinement_v6.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False)
    )
    lines = [
        "# 静态偏置、连续 RGB 校验与波动分区修正（v6）",
        "",
        "实现验证：本轮完整测试 172 项通过（13 条既有警告）；静态检查通过。测试通过不等于测量精度达标。",
        "",
        "日期：2026-09-05。主网络权重保持不变，本轮为后处理、可观测性表达与仿真诊断修正。所有实验在服务器完成。",
        "",
        "## 1. 实现与上线边界",
        "",
        "- 静态：新增图像网格均衡高度估计；主动双目/结构光可选含量程截断归一化的视差域稳健似然，ToF 不使用视差公式。",
        "- 数值求解：稳健混合似然存在平坦区。先在包含稳健初值的粗网格定位，再做局部连续求解，避免近距离窄最优区被跳过。",
        "- RGB：新增连续投影轮廓+双线性有符号距离场+鲁棒损失，保留一像素扰动预算。标定文件采用不同版本，禁止混用旧拟合器偏置。",
        "- 波动：用当前局部网格锚点和局部距离加权插值替代全局二次曲面外推；返回最深/最浅/均值/P05/P95，以及有条件的范围。",
        "- 仿真：增加深度噪声、视差噪声、量化、量程截断的独立开关；默认开关与旧行为一致。",
        "- 生产路径不自动替换。新增 process_refined_surface 返回 accepted=false；candidate_available 只表示有研究候选。95%–100% 失效继续拒绝。",
        "",
        "## 2. 测试设计",
        "",
        "同帧回归：沿用 v5 的两个种子、96 条序列，每组合 100 个评估帧，共 36000 帧条件。使用冻结模型预测掩码；初始化不计入指标。它是回归集，不是新盲测集。",
        "",
        "独立传感器实验：种子 11647/11749；距离 1/3/6/10 m、固定半径 0.3/0.6 m、液深 0.1/0.3 m，三类传感器完全交叉。每组合 100 帧，共 9600 个基准帧、5 种噪声条件即 48000 帧条件。使用真实液面掩码来隔离传感器误差，因此不能冒充端到端性能。",
        "",
        "独立 RGB 实验：种子 11441/11543，固定半径 0.3/0.6 m，距离/初始液深为 1/0.3、3/0.1、6/0.3 m。共 84 张原生 HR 图；每组首张标定，另外 6 个真实液深变化，比较低/高分辨率、两种拟合器、干净/模糊JPEG/水平偏移3像素/35%亮度。RGB 液深并未与距离完全交叉，这是本轮剩余限制。",
        "",
        "容差沿用 max(5 mm, 液深×2%)；拒绝不计零误差。通过率只针对有值帧。新增候选与原方法的可用子集可能不同，覆盖率必须一起看。",
        "",
        "## 3. 静态同帧结果",
        "",
        "| 条件 | 路径 | 候选覆盖率 | 液深 MAE/mm | 容差通过率 |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {"gravity": "原重力中位数", "balanced": "网格均衡", "sensor": "传感器域修正"}
    for variant in ("normal", "drop75", "drop90", "slow_echo"):
        for method, label in labels.items():
            s = long["summaries"]["static/" + variant][method]
            m = s["mean_depth_m"]
            lines.append(
                f"| {variant} | {label} | {100 * s['coverage']:.2f}% | {number(m['mae_mm'])} | {100 * (1 - m['outside_tolerance']):.2f}% |"
            )
    lines += [
        "",
        "分距离结果（90% 额外空洞）：",
        "",
        "| 距离/液深(m) | 原重力 MAE/mm | 传感器修正 MAE/mm | 原通过率 | 修正通过率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for d, h in ((0.2, 0.1), (0.5, 0.3), (1, 0.3), (3, 0.1), (6, 0.3), (8, 8), (10, 10)):
        s = long["summaries"][f"static/drop90/d{d:g}/h{h:g}"]
        a, b = s["gravity"]["mean_depth_m"], s["sensor"]["mean_depth_m"]
        lines.append(
            f"| {d:g}/{h:g} | {number(a['mae_mm'])} | {number(b['mae_mm'])} | {100 * (1 - a['outside_tolerance']):.2f}% | {100 * (1 - b['outside_tolerance']):.2f}% |"
        )
    lines += [
        "",
        "改善汇总 MAE 不等于所有距离和通过率都改善。慢错误回波仍可整体偏移平面；本模块没有独立信息可证明它是错回波，不能凭低残差输出。",
        "",
        "## 4. 独立传感器消融与偏置来源",
        "",
        "下表距离为 10 m，固定容器、浅液体、真实掩码；有符号偏差用于区分系统偏移与随机噪声。",
        "",
        "| 传感器 | 条件 | 均衡估计偏差/mm | 均衡估计 MAE/mm |",
        "|---|---|---:|---:|",
    ]
    sensor = data["sensor_components_v6_final"]["summaries"]
    for family in ("active_stereo", "structured_light", "tof"):
        for variant in (
            "all",
            "without_depth_noise",
            "without_disparity_noise",
            "without_quantization",
            "without_range_cutoff",
        ):
            s = sensor[f"d10/{family}/{variant}"]["balanced"]
            lines.append(f"| {family} | {variant} | {number(s['bias_mm'])} | {number(s['mae_mm'])} |")
    lines += [
        "",
        "新种子、固定容器的修正效果（10 m相机距离，液深仅0.1/0.3 m）：",
        "",
        "| 传感器 | 均衡 MAE/mm | 修正 MAE/mm | 修正偏差/mm | 均衡有值帧 | 修正有值帧 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("active_stereo", "structured_light", "tof"):
        a, b = sensor[f"d10/{family}/all"]["balanced"], sensor[f"d10/{family}/all"]["sensor"]
        lines.append(
            f"| {family} | {number(a['mae_mm'])} | {number(b['mae_mm'])} | {number(b['bias_mm'])} | {a['available']} | {b['available']} |"
        )
    lines += [
        "",
        "远距离小容器的有效支持较少，修正虽降低MAE，但仍有显著剩余偏差，且不能因有值子集缩小而夸大改善。这里没有证明浅液体达到目标。ToF这一组很低的误差仅是当前噪声代理的结果，不代表真实ToF对透明液体具有同等表现。",
        "",
        "关闭视差噪声或量程截断显著改变远端偏差；单独关闭量化或原始深度噪声的作用较小。这支持优先处理视差噪声与截断的耦合，但逐项关闭不是严格可加的误差分解。",
    ]
    lines += [
        "",
        "真实设备可能有不同基线、量化、有效量程与噪声。StereoNoiseModel.simulation_proxy 明确仅供仿真；部署必须传入独立标定参数，不能复制这些常数当行业标准。",
        "",
        "## 5. 波动表面与条件区间",
        "",
        "| 条件 | 二次曲面均值 MAE/mm | 分区均值 MAE/mm | 分区最浅 MAE/mm | 分区最深 MAE/mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("normal", "drop75", "drop90"):
        s = long["summaries"]["waves/" + variant]
        a, b = s["wave"], s["partition"]
        lines.append(
            f"| {variant} | {number(a['mean_depth_m']['mae_mm'])} | {number(b['mean_depth_m']['mae_mm'])} | {number(b['min_depth_m']['mae_mm'])} | {number(b['max_depth_m']['mae_mm'])} |"
        )
    lines += [
        "",
        "| 条件 | 有区间帧比例 | 均值区间含真值率 | 均值区间平均宽度/mm | 锚点网格覆盖占比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("normal", "drop75", "drop90"):
        s = long["summaries"]["waves/" + variant]["conditional_intervals"]
        lines.append(
            f"| {variant} | {100 * s['available'] / s['frames']:.2f}% | {100 * s['mean_depth_m']['contains_truth']:.2f}% | {number(s['mean_depth_m']['mean_width_mm'])} | {100 * s['observed_area_mean']:.2f}% |"
        )
    lines += [
        "",
        "区间以给定最大坡度和锚点误差预算为前提：每个锚点给出随水平距离扩大的允许高度范围，再取交集。冲突时不输出区间；没有显式坡度先验时，也不输出全局区间。实验坡度上限为 2 m/m，噪声预算沿用冻结 v4 标定。",
        "",
        "这些范围不是经过概率校准的置信区间；预算没有证明覆盖所有系统误差、位姿和锚点横向位置误差。区间很宽意味着无法精确判断，不是测量成功。均值改善也不代表波峰/波谷恢复成功，完整最深/最浅指标仍保留。",
        "",
        "observed_area_fraction 是当前锚点覆盖的水平网格占比，不是真实逐像素可观测面积；未观测网格由局部插值推断。波动分支不复用旧形状，避免过期波峰；既有 early 法向记忆仍在独立 v5 接口，不能称本轮实现了新的动态波面时序网络。",
        "",
        "## 6. 连续 RGB 的收益与限制",
        "",
        "以下仅比较两方法都有值的同一图像，均为干净输入；可用帧数量变化仍见配套 JSON。",
        "",
        "| 数据 | 距离(m) | 宽度(px) | 同帧数 | 原搜索 MAE/mm | 连续搜索 MAE/mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("rgb_continuous_v6_regression", "rgb_continuous_v6_r03", "rgb_continuous_v6_r06"):
        for key, s in data[name]["common_frame_mae_mm"].items():
            if key.endswith("/clean"):
                d, w, _ = key.split("/")
                lines.append(
                    f"| {name.replace('rgb_continuous_v6_', '')} | {d[1:]} | {w} | {s['frames']} | {number(s['grid'])} | {number(s['continuous'])} |"
                )
    lines += [
        "",
        "连续求解在部分近距离 HR 图上改善明显，但固定半径的新场景存在退化或无法标定，不作为默认替代。对旧数据种子11231的 +20 mm 变化，3/6 m 的真值掩码分别改变2544/1283像素，而颜色分割掩码只改变286/303像素：说明后续还需解决实际液面边界识别，不是单纯增加搜索精度或 SR 就能解决。",
        "",
        "暗光、偏移等反例的可用率、误差预算越界、人工注入 +20 mm 深度错误的准入结果均在 JSON。该准入探针不是完整时序系统误接受率；零准入也不能作为测量成功。超分仍默认关闭，原始像素预算继续传递。",
        "",
        "## 7. 热身后延时",
        "",
        "输入/输出 IO、相机采集、界面与真实位姿求解不计入。batch=1、同帧热身5次后20次；RGB单独计时，不能把独立模块P95相加当完整系统P95。",
        "",
        "| 路径 | 中位数/ms | P95/ms |",
        "|---|---:|---:|",
    ]
    latency = data["surface_refinement_v6_latency"]
    for key in ("balanced", "sensor", "partition", "rgb_grid", "rgb_continuous"):
        s = latency[key]
        lines.append(f"| {key} | {number(s['median_ms'])} | {number(s['p95_ms'])} |")
    lines += [
        "",
        "## 8. 使用和复现",
        "",
        "默认生产 process 不变。显式调用 process_refined_surface(..., mode='balanced'/'sensor'/'partition', area_xy=均匀占地网格, radii=已知椭圆半径)。sensor 模式传入 StereoNoiseModel；不提供时仅做均衡高度估计。partition 的 max_surface_slope 默认 None，表示不能声称全局范围已知。世界坐标、底面和相机位姿必须一致。",
        "",
        "连续 RGB 使用 RGBContinuousWitness，并重新 calibrate；其 schema_version=2，旧 RGBContourWitness 文件不能互换。没有新增工业面板的默认开关，也没有发布新主网络权重。",
        "",
        "复现入口：scripts/evaluate_surface_refinement_v6.py、scripts/evaluate_sensor_components_v6.py、scripts/evaluate_rgb_continuous_v6.py、scripts/benchmark_surface_refinement_v6.py。新固定容器图像由 scripts/generate_sr_level_sequences.py --fixed-radius 0.3/0.6 --seeds 11441 11543 生成，使用各自新的输出目录。",
        "",
        "完整结果位于 /root/autodl-tmp/liquid-depth-artifacts/evaluation/。配套 docs/surface_refinement_v6.json 保留各结果摘要、原始文件路径与 SHA256。本报告只采用 *_final 的静态结果；早期数值搜索和审计字段问题已修正，其旧中间结果不作为最终结论。",
        "",
        "## 9. 后续仍需处理",
        "",
        "1. 用新设备标定与新种子验证噪声模型失配；特别是远距离浅液体，不能因为10 m深液体改善就宣称10 cm浅液体同样达标。",
        "2. RGB 先改能随真实液位变化的边界证据，再考虑局部超分；连续拟合保持按场景可选。",
        "3. 波动极值采用更可靠锚点、局部形状模型与专门的不确定度校准；本轮范围仍可能过宽。",
        "4. 用新的非周期波形、漂浮物、容器形状和真实位姿误差扩展验证；不扩展95%–100%失效恢复优化。",
        "",
    ]
    (args.output_dir / "surface_refinement_v6.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
