"""Produce the auditable all-bin ordinary-scene comparison report."""
import argparse,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np

def paired(q):
    both=[r for r in q if all(r['methods'][m]['numeric'] for m in ('v35','v36'))]
    cluster=defaultdict(list)
    for r in both:cluster[r['cluster']].append((r['methods']['v35']['absolute_error_m']-r['methods']['v36']['absolute_error_m'])*1000)
    sums=np.array([[sum(v),len(v)] for v in cluster.values()]);rng=np.random.default_rng(36267)
    sample=sums[rng.integers(0,len(sums),(2000,len(sums)))].sum(1)
    return dict(common=len(both),clusters=len(cluster),improvement_mm=float(sums[:,0].sum()/sums[:,1].sum()),
        bootstrap95_mm=np.percentile(sample[:,0]/sample[:,1],[2.5,97.5]).tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.output/'raw.json').read_text())['records'];s=json.loads((a.output/'summary.json').read_text());v=json.loads((a.output/'verification.json').read_text())
    groups={'all':rows,'far_sl':[r for r in rows if r['sensor']=='structured_light' and r['camera_to_surface_m']>=3 and r['true_height_m']<1],
        '7-8':[r for r in rows if 7<=r['true_height_m']<8]}
    audit={k:paired(q) for k,q in groups.items()}
    audit['negative_outputs']={m:sum(r['methods'][m]['numeric'] and r['methods'][m]['statistics']['mean_m']<0 for r in rows) for m in ('v35','v36')}
    audit['compute_ms']={m:np.percentile([r['methods'][m]['compute_ms'] for r in rows],[50,95,99]).tolist() for m in ('v35','v36')}
    audit['effective_far_sl']=dict(before=float(np.mean([r['methods']['v36'].get('effective_points_before_uniform',0) for r in groups['far_sl']])),
        after=float(np.mean([r['methods']['v36'].get('effective_points',0) for r in groups['far_sl']])))
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2))
    f=lambda x:'—' if x is None else f'{x:.2f}'
    lines=['# 普通场景远距离浅液位结构光偏差修正与独立测试','',
        '日期：2026-10-12。测量量为相对容器内底面的实际液深。透明、半透明及 ToF 分支未更改。','',
        '## 方法与适用范围','',
        '本轮在同一批未降低噪声的新仿真输入上比较 V35 与 V36。V35 是上一轮自适应容器选点与连续视差似然；V36 在其基础上修正量化及截断概率、加入底面回波项，并调整普通平面液面的点权重和近端选点范围。没有重新训练网络，也没有替换产品默认链路。',
        '新似然计算观测深度量化格的概率，而非把量化值当连续观测。有效量程由量化视差格中心决定，归一化边界扩展到首尾格的半格边界；同时考虑最终 1 mm 深度舍入。已知容器底面只用于解释错误回波，不充当液深标签，不把接近底面的测量值归零。',
        '原图像分块等权会放大边缘小块中的少量噪声点。本轮对普通平面液面使用点等权并提高最多使用点数，避免少数点主导。该策略不能直接替代波动液面的面积加权评价。',
        '根据初步观测距离小于 3 m 时扩大容器内部范围到 90%，增加近端有效点；远端仍采用 65% 核心范围。选择不读取真实液深区间。没有为 7–8 m 液深写专门的真值路由。',
        '传感器噪声参数和视差/深度量化步长可显式传入。默认是现有仿真代理：320×180、约 314 px 焦距、基线 0.055/0.035 m、视差标准差 0.18/0.26 px、视差步长 1/16 px。它们不是工业相机的实测规格。缩放图像时，焦距、视差噪声和视差步长必须同时换算。','',
        '## 原因验证与开发数据','',
        '上一轮独立测试集现在用于本轮开发和消融，不能再算本轮独立证据。先比较量化修正、底面回波项、近端扩大范围与点等权，确定方案后才生成新测试集。',
        '受控分布实验中，结构光连续噪声用旧连续似然估计的 MAE 为 47.47 mm；加入量化后升至 120.95 mm，换成量化似然后降至 45.33 mm。再加入 5% 底面回波时，旧方法 MAE 为 256.53 mm，新方法约 44.12 mm。这支持量化截断边界和回波分布不匹配是重要偏差来源，不是单纯调低噪声后的改善。',
        '在这组受控实验中，额外底面回波项相对于单独量化修正的收益很小，不夸大其贡献。完整开发集的消融保存在 ordinary-v36-dev 与 ordinary-v36-dev-weights；受控实验结果为 noise-probe.json。','',
        '## 独立测试协议','',
        f'新的 297 个几何配置，每组 6 次噪声重复，总计 {len(rows)} 帧；11 个液深区间 × 3 种传感器 × 9 组配置，每区间 162 帧。空槽单列，其余为 0–0.1、0.1–1、1–2 至 8–9 m。相机到底面最大距离 {v["camera_bottom_max_m"]:.3f} m，最大真实液深 {v["height_max_m"]:.3f} m。',
        f'冻结源码后生成，完成后全部哈希一致。干槽实体底面射线检查 {v["dry_floor_cases"]} 组，最大几何偏差 {v["floor_max_error_m"]*1000:.6f} mm。仿真生成器与噪声参数沿用上轮修复版，没有降低噪声难度。',
        '固定顶部相机、理想底面/位姿/内参和固定竖直凸容器。相机到液面距离 ≥3 m 且液深 <1 m 定义为远距离浅液位，包含空槽。重复帧共享静态 RGB，不是独立容器或真实动态视频。',
        '放宽门控，保留负液深和越界输出，不做强制归零或容量裁剪；仅结构无解时记录未输出。容差为 max(5 mm, 2%×实际液深)。MAE、P95 对有输出帧计算；通过率与绝对误差分箱以全部帧为分母。相对误差只统计非空且有值帧。','',
        '## 总体与传感器子集','',
        '| 子集 | 方法 | 总帧 | 未输出 | MAE mm | P95 mm | 偏差 mm | 全帧通过率 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for key in ('all','far_shallow','far_shallow/structured_light','far_shallow/active_stereo','near','far','sensor/tof'):
        for m in ('v35','v36'):
            q=s[key][m];lines.append('| '+' | '.join([key,m,str(q['frames']),str(q['missing_frames'])]+[f(q[k]) for k in ('mae_mm','p95_mm','signed_bias_mm','all_frame_pass_percent')])+' |')
    bands=['empty','0-0.1','0.1-1']+[f'{i}-{i+1}' for i in range(1,9)]
    lines+=['','## 各液深区间误差与未输出','',
        '| 液深 m | 方法 | 未输出/总帧 | MAE mm | 中位数 mm | P95 mm | 最大 mm | AbsRel % | 全帧通过率 % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for b in bands:
        for m in ('v35','v36'):
            q=s[b][m];lines.append('| '+' | '.join([b,m,f'{q["missing_frames"]}/{q["frames"]}']+[f(q[k]) for k in ('mae_mm','median_mm','p95_mm','max_mm','abs_rel_percent','all_frame_pass_percent')])+' |')
    for rel in (False,True):
        lines+=['','## '+('相对误差分布' if rel else '绝对误差分布'),'',
            '| 液深 m | 方法 | '+('<0.5% | 0.5–1% | 1–2% | 2–3% | 3–5% | 5–10% | ≥10% |' if rel else '<5 mm | 5–10 | 10–20 | 20–50 | 50–100 | 100–200 | ≥200 | 未输出 % |'),
            '|---|---|'+('---:|'*(7 if rel else 8))]
        for b in bands[1:] if rel else bands:
            for m in ('v35','v36'):
                q=s[b][m];vals=q['relative_bins_percent_numeric' if rel else 'absolute_bins_percent_all']
                if not rel:vals=vals+[100*(1-q['numeric_coverage'])]
                lines.append('| '+' | '.join([b,m]+[f(x) for x in vals])+' |')
    lines+=['','## 配对核验与使用限制','']
    for key in ('all','far_sl','7-8'):
        z=audit[key];lines.append(f'{key}：共同有值 {z["common"]} 帧、{z["clusters"]} 个配置，MAE 改善 {z["improvement_mm"]:.3f} mm；按配置成组重采样的 95% 区间为 {z["bootstrap95_mm"]} mm。')
    lines+=['',f'负液深输出数：{audit["negative_outputs"]}。未输出原因：V35 {s["all"]["v35"]["no_solution_reasons"]}；V36 {s["all"]["v36"]["no_solution_reasons"]}。',
        f'结构光远距离浅液位平均有效样本数（按权重平方和定义）从原分块权重的 {audit["effective_far_sl"]["before"]:.1f} 增至点等权的 {audit["effective_far_sl"]["after"]:.1f}。这不代表真实传感器像素噪声完全独立，硬件空间相关性仍需实测。',
        f'并行评估中的估计器耗时 P50/P95/P99，单位 ms：{audit["compute_ms"]}。不是采集、网络和 GUI 的端到端延迟。',
        '所有输出仍为 qualified=false。有值率不等于工业可接受输出率。新模型依赖原始传感器量化与噪声标定；经过空间滤波、超分或神经深度修复的深度图未必服从此分布，不能直接套用。',
        '高液深区间受相机到底面 ≤10 m 约束，相机往往更靠近液面；不能把其低误差解释为十米远距离测距精度。当前 ToF 普通噪声代理较简单，其结果不能作为真实传感器选型结论。','',
        '## 复现路径','',f'服务器结果目录：`{a.output}`。raw.json 保存逐帧值与原因；summary.json 保存全部分箱；audit.json 保存配对统计；protocol.json、frozen.json、verification.json 保存协议和源码核验。',
        '实现模块为 `src/liquid_depth/ordinary_surface_v36.py`。本轮选定配置为 `estimate_v36(..., uniform=True)`；显式提供安装几何与传感器标定。未自动替换产品默认路由。','']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
