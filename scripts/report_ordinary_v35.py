"""Summarize every depth bin, tails, missing outputs and paired case bootstrap."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np

def paired(rows):
    q=[r for r in rows if all(r['methods'][m]['numeric'] for m in ('baseline','core_0'))]
    clusters=sorted(set(r['cluster'] for r in q));rng=np.random.default_rng(35177)
    sums=np.array([[sum(r['methods']['baseline']['absolute_error_m']-r['methods']['core_0']['absolute_error_m'] for r in q if r['cluster']==c),
        sum(r['cluster']==c for r in q)] for c in clusters])
    draws=rng.integers(0,len(clusters),(2000,len(clusters)));sample=sums[draws].sum(1)
    ci=np.percentile(sample[:,0]/sample[:,1]*1000,[2.5,97.5]).tolist()
    return dict(common_numeric_frames=len(q),mean_error_reduction_mm=float(sums[:,0].sum()/sums[:,1].sum()*1000),
        cluster_bootstrap_95_interval_mm=ci,improved_fraction=float(np.mean([r['methods']['core_0']['absolute_error_m']<r['methods']['baseline']['absolute_error_m'] for r in q])))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.output/'raw.json').read_text())['records'];s=json.loads((a.output/'summary.json').read_text())
    v=json.loads((a.output/'verification.json').read_text());protocol=json.loads((a.output/'protocol.json').read_text())
    audit=dict(paired=paired(rows),far_shallow_paired=paired([r for r in rows if r['camera_to_surface_m']>=3 and r['true_height_m']<1]),
        selected_route_flags=dict(Counter(f for r in rows for f in r['methods']['core_0']['reasons'])),
        negative_outputs={m:sum(r['methods'][m]['numeric'] and r['methods'][m]['statistics']['mean_m']<0 for r in rows) for m in ('baseline','core_0')},
        optimized_compute_ms_percentiles=np.percentile([r['methods']['core_0']['compute_ms'] for r in rows],[50,95,99]).tolist())
    (a.output/'paired-audit.json').write_text(json.dumps(audit,indent=2))
    f=lambda x:'—' if x is None else f'{x:.2f}'
    lines=['# 普通场景：干槽底面修复、容器选点约束与独立分量程测试','',
        '日期：2026-10-12。状态：仿真研究链路；透明、半透明链路保持不变。','',
        '## 结论与边界','',
        '本轮比较的是同一批修正后的独立仿真输入上，原 V8 掩码＋传感器域估计与新增容器几何选点方法。不是把旧仿真结果与新仿真结果直接相减。没有训练或更换神经网络权重，没有改变产品默认路由。',
        '', '## 仿真修复','',
        '1. 为容器添加闭合、可渲染的不透明内底面；空槽不创建液体网格。使用 Blender 实体底面射线与解析深度核对，而不是只给空槽一个零标签。',
        '2. 每个重复帧重新独立生成液面噪声、背景噪声和壁面回波，不再使用第一次噪声的有效区域决定后续采样。',
        '3. 深度有效量程与似然模型统一：双目/结构光 0.12–10.5 m，ToF 0.008–10.5 m；这属于仿真传感器假设，不是对实际相机的规格承诺。',
        f'4. 新测试集干槽实体检查共 {v["dry_floor_cases"]} 组，最大射线偏差 {v["floor_max_error_m"]*1000:.6f} mm。相机到底面最大距离 {v["camera_bottom_max_m"]:.3f} m，最大真实液深 {v["height_max_m"]:.3f} m。','',
        '## 方法与开发过程','',
        '基于已知内参、位姿、容器内底面、槽口高度和固定竖直凸容器轮廓，在射线空间选择容器内部区域。与按噪声三维点坐标筛选不同，它不因同一点的深度噪声而改变入选概率；不读取真实液深、真实液面掩码或真值深度。',
        '开发集 99 个几何配置、396 帧。比较 65% 与 85% 半径核心区：65% 核心区更有利于远距离浅液位，但深窄容器中会丢失支持。最终冻结自适应方案：优先 65% 核心区；若仅按安装几何计算的像素数小于 256，使用 65% 槽口核心区并标记约束减弱。没有用测试液深切换算法。',
        '后备方案不保证消除槽壁回波，必须保留其标记。依然使用原传感器域似然/ToF 鲁棒统计，不做液深归零、上下界裁剪、强制置信度拒绝。结构上无解时仍明确报告未输出。',
        '开发集最初容器材料元数据写作 metal，但实际渲染为玻璃槽壁；开发后更正为 glass。两者在本轮普通场景噪声函数中都不启用 coated_glass 特殊项，开发输入未因该标签修改而重新生成。','',
        '## 独立测试协议','',
        f'新测试种子 {protocol["seed"]}；198 个几何配置，每个配置 6 次深度噪声重复，总计 {len(rows)} 帧。11 个液深区间 × 3 类传感器 × 6 组配置；每个液深区间 108 帧。先冻结代码、权重和选点方案，再生成新测试集，完成后核对全部哈希。',
        '空槽单列，其他真实液深分为 0–0.1、0.1–1、1–2、…、8–9 m。远距离浅液位子集定义为相机到水平液面垂直距离 ≥3 m 且实际液深 <1 m，包含空槽。',
        '测量量为相对容器内底面的液深。通过容差 max(5 mm, 2%×实际液深)。绝对误差分箱以全部帧为分母；相对误差分箱仅使用非空且有输出帧。空槽相对误差未定义。',
        '重复帧共享静态 RGB，不能视为独立容器；置信区间按几何配置整组重采样。高液深因相机到底面限制而具有更短的相机—液面距离，不能把其优秀结果解释为 9 m 相机测距精度。','',
        '## 同输入总体对比','',
        '| 子集 | 方法 | 帧数 | 未输出 | MAE mm | P95 mm | 偏差 mm | 全帧通过率 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for key in ('all','far_shallow','sensor/active_stereo','sensor/structured_light','sensor/tof'):
        for m in ('baseline','raw_grid','core_0'):
            q=s[key][m];lines.append('| '+' | '.join([key,m,str(q['frames']),str(q['missing_frames']),f(q['mae_mm']),f(q['p95_mm']),f(q['signed_bias_mm']),f(q['all_frame_pass_percent'])])+' |')
    lines+=['','baseline：V8 掩码＋原传感器域估计；raw_grid：V8 掩码＋原容器范围网格法，作为辅助参照；core_0：本轮自适应容器选点。',
        '',f'共同有值帧配对 MAE 改善：{f(audit["paired"]["mean_error_reduction_mm"])} mm；按几何配置 bootstrap 的 95% 区间 {audit["paired"]["cluster_bootstrap_95_interval_mm"]} mm。这不是跨真实场景泛化的置信区间。','',
        '## 各液深区间误差与未输出','',
        '| 液深 m | 方法 | 未输出/总数 | MAE mm | 中位数 mm | P95 mm | 最大 mm | AbsRel % | 全帧通过率 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    bands=['empty','0-0.1','0.1-1']+[f'{i}-{i+1}' for i in range(1,9)]
    for b in bands:
        for m in ('baseline','core_0'):
            q=s[b][m];lines.append('| '+' | '.join([b,m,f'{q["missing_frames"]}/{q["frames"]}']+[f(q[k]) for k in ('mae_mm','median_mm','p95_mm','max_mm','abs_rel_percent','all_frame_pass_percent')])+' |')
    for relative in (False,True):
        lines+=['', '## '+('相对误差分布（非空且有值帧）' if relative else '绝对误差分布（全部帧分母）'),'',
            '| 液深 m | 方法 | '+(' <0.5% | 0.5–1% | 1–2% | 2–3% | 3–5% | 5–10% | ≥10% |' if relative else '<5 mm | 5–10 | 10–20 | 20–50 | 50–100 | 100–200 | ≥200 | 未输出 % |'),
            '|---|---|'+('---:|'*(7 if relative else 8))]
        for b in bands[1:] if relative else bands:
            for m in ('baseline','core_0'):
                q=s[b][m];values=q['relative_bins_percent_numeric' if relative else 'absolute_bins_percent_all']
                if not relative:values=values+[100*(1-q['numeric_coverage'])]
                lines.append('| '+' | '.join([b,m]+[f(x) for x in values])+' |')
    lines+=['','## 未输出、负值与限制','',
        f'原方法未输出原因：{s["all"]["baseline"]["no_solution_reasons"]}。优化方法未输出原因：{s["all"]["core_0"]["no_solution_reasons"]}。',
        f'保留的负液深输出数：{audit["negative_outputs"]}，未通过强制归零提高空槽通过率。',
        '本轮“有值率”是放宽门控的诊断输出比例，不是工业可接受输出率。所有新方法输出 qualified=false，尚未经过独立硬件校准和资格验证。',
        f'线程并行评估下新估计器自身耗时 P50/P95/P99：{audit["optimized_compute_ms_percentiles"]} ms，不能作为完整相机采集、网络、GUI 的端到端延迟。',
        '当前容器为固定、竖直、凸轮廓；底面与位姿为理想安装标定。不会证明标定误差、斜底面、移动相机、强波动、透明折射或严重遮挡下的效果。',
        'Eevee RGB＋解析几何＋参数化深度噪声仍不是完整光学级 RGB-D 传感器仿真；不同噪声重复共享 RGB，数据量不能冒充多台真实传感器验证。',
        '', '## 复现与代码','',
        f'测试产物目录：`{a.output}`。逐帧结果 raw.json，完整分箱 summary.json，配对检验 paired-audit.json，冻结清单 frozen.json，核验 verification.json。',
        '`src/liquid_depth/ordinary_surface_v35.py` 为独立可调用的普通平面液面估计模块；显式传入 fraction=0 使用本轮自适应策略。产品默认链路未替换；后续接入时应由普通场景路由显式启用并提供所需安装标定。',
        '`scripts/generate_ordinary_v35.py`、`evaluate_ordinary_v35.py`、`run_ordinary_v35.py`、`report_ordinary_v35.py` 保留完整生成、评估、冻结和汇总流程。','']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
