"""Paired temporal results, relative histograms and depth-bin regression audit."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def paired(rows,method):
    q=[r for r in rows if r['methods']['single']['numeric'] and r['methods'][method]['numeric']]
    if not q:return None
    groups=defaultdict(list)
    for r in q:groups[r['cluster']].append((r['methods']['single']['absolute_error_m']-r['methods'][method]['absolute_error_m'])*1000)
    z=np.array([[sum(v),len(v)] for v in groups.values()]);rng=np.random.default_rng(37721)
    b=z[rng.integers(0,len(z),(2000,len(z)))].sum(1)
    return dict(common_frames=len(q),clusters=len(z),mae_improvement_mm=float(z[:,0].sum()/z[:,1].sum()),
        cluster_bootstrap95_mm=np.percentile(b[:,0]/b[:,1],[2.5,97.5]).tolist())

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.output/'raw.json').read_text())['records'];s=json.loads((a.output/'summary.json').read_text())
    protocol=json.loads((a.output/'protocol.json').read_text());verify=json.loads((a.output/'verification.json').read_text())
    modes=('iid','spatial','persistent');names={'iid':'独立噪声','spatial':'空间相关 每帧刷新','persistent':'空间相关 跨帧持续'}
    methods=('single','joint2','joint4','guarded4');audit={};bands=['empty','0-0.1','0.1-1']+[f'{i}-{i+1}' for i in range(1,9)]
    for mode in modes:
        base=[r for r in rows if r['noise']==mode]
        for key,subset in [('all',base),('far_sl',[r for r in base if r['sensor']=='structured_light' and r['camera_to_surface_m']>=3 and r['true_height_m']<1]),
                           ('7-8',[r for r in base if 7<=r['true_height_m']<8])]:
            audit[mode+'/'+key]={m:paired(subset,m) for m in methods[1:]}
    audit['regression_bins']={mode:{m:[b for b in bands if s[mode+'/'+b][m]['mae_mm']>s[mode+'/'+b]['single']['mae_mm']]
        for m in methods[1:]} for mode in modes}
    audit['negative_counts']={mode:{m:sum(r['noise']==mode and r['methods'][m]['numeric'] and r['methods'][m]['statistics']['mean_m']<0 for r in rows) for m in methods} for mode in modes}
    (a.output/'paired-audit.json').write_text(json.dumps(audit,indent=2))
    f=lambda x:'—' if x is None else f'{x:.2f}'
    lines=['# 普通场景多帧联合液深估计与空间相关噪声测试','',
        '日期：2026-10-12。测量量为相对容器内底面的实际液深。仅适用于固定相机、普通平面液面；透明和半透明链路不改动。','',
        '## 比较方法和边界','',
        '单帧基线为上一轮 V36 的量化及底面回波似然、容器范围选点与点等权。2 帧和 4 帧方法把各帧原始观测的对数似然合并，共同估计一个液位；每帧的总权重相同，防止有效点多的一帧主导。另保留独立估计后算术平均 mean4 作为消融参照。',
        '不使用真实液深、真值掩码或噪声模式标签选择算法。保留负液深和超过容器容量的测量，不归零、不裁剪、不按置信度筛掉测试帧。当前帧结构无解时单列未输出，不能把仅有历史数据的结果冒充当前观测。',
        '模块可接收时间戳、位姿有效性与外部运动提示；历史跨度超过 0.5 s、相机非固定或外部运动标记为真时，退回当前帧，不额外拒绝当前输出。这里没有实现或验证自动运动检测器。',
        '另增加 guarded4 保守策略：根据当前测量得到的相机到液面距离小于 3 m 时保留单帧；远端从原始观测估计 8×8 块残差的跨帧相关性，中位相关系数大于 0.5 或无法估计时保留当前帧，否则用 4 帧联合。没有读取真实液深或噪声类型标签；该策略只是选择测量方法，不因相关性而拒绝输出。',
        '相关性诊断对残差作稳健限幅，但不裁剪原始输入用于液位拟合的测量，更不裁剪液位输出。相关性阈值和距离阈值在独立测试前固定。',
        '本轮 ToF 路由保持单帧不变，因此其 single/joint2/joint4 结果相同；不是对 ToF 多帧收益的测试。对双目和结构光使用联合估计。',
        '所有结果 qualified=false，输出可用比例不是工业合格覆盖率。普通液面平面假设下的点权重不能直接替代不平整液面的面积统计。','',
        '## 三种噪声条件','',
        '| 条件 | 单点总噪声方差 | 空间关系 | 时间关系 |',
        '|---|---|---|---|',
        '| iid | 保持 V36 参数 | 独立 | 独立 |',
        '| spatial | 保持 V36 参数 | 50% 方差来自 8×8 像素块公共高斯场 | 每帧重新采样 |',
        '| persistent | 保持 V36 参数 | 同上 | 块公共场 AR(1) 系数 0.9 |','',
        '相关性在深度噪声和视差噪声生成阶段引入，再执行量化、量程截断和背景合成。失效与错误回波结构在三种条件间配对；没有先丢弃独立噪声失效点再追加相关噪声。',
        '高斯场只保证跨样本的理论单位方差；不会逐帧减均值或强制归一化，这避免人为去掉区域共同偏差。相关参数是敏感性测试假设，不是已测定的某款相机参数。','',
        '这是传感器深度／视差域的参数化噪声仿真，并非真实结构光投影条纹、曝光、解码及多径的完整物理重建。RGB 不参与本轮普通液位估计；不能将结果当成真实工业相机精度。','',
        '| 条件 | 场平均值 | 实测方差 | 水平相邻相关系数 | 同像素相邻帧相关系数 |',
        '|---|---:|---:|---:|---:|']
    for mode in modes:
        q=verify['noise_audit'][mode];lines.append('| '+mode+' | '+' | '.join(f(q[k]) for k in ('mean','variance','horizontal_adjacent_correlation','temporal_correlation'))+' |')
    geometries=verify['geometries'];nframes=protocol['frames'];endpoints=protocol['frame_endpoint_indices'];permode=len(rows)//3
    lines+=['','## 独立测试协议','',
        f'新测试种子 {protocol["seed"]}，{geometries} 个容器及相机配置，每配置 {nframes} 帧、3 种噪声，共 {geometries*nframes*3} 帧 RGB-D 条件。RGB 为每配置一张静态渲染重复使用，深度观测随时间变化，不是真实动态视频。',
        f'只在每个不重叠 4 帧窗口的共同结束帧 {endpoints} 比较，单帧取相同结束时刻；每种噪声 {permode} 个可比较输出条件。相对误差只对非空槽且有值的共同评估时刻统计；不把空槽计为零相对误差。',
        '新测试生成前冻结算法和参数，完成后检查源码哈希。开发集与测试集种子不同；开发阶段不调整噪声幅度来改善结果。旧 V36 测试集不是本轮独立证据。',
        '所有相对误差分箱均左闭右开；最后一格为 ≥10%。0–0.1 m 的非空采样从约 0.01 m 起，并未覆盖任意接近零的液深。容器形状与距离档位沿用既有分层生成器，两者不是完全交叉，因此形状效应与距离效应尚不能彻底分离。',
        f'最大相机到底面距离 {verify["camera_bottom_max_m"]:.3f} m，最大真实液深 {verify["height_max_m"]:.3f} m。理想内参、外参、底面和竖直凸容器；远距离浅液位指相机到液面 ≥3 m、液深 <1 m。',
        '静态窗口以 30 fps 定义，2 帧历史跨度约 33 ms，4 帧约 100 ms。完整窗口测试不包含首次启动的暖机输出率；实际模块允许先使用当前帧。','',
        '## 总体与远距离浅液位','',
        '| 噪声 | 子集 | 方法 | 未输出/总数 | MAE mm | P95 mm | AbsRel % | <2% 占比 % | ≥10% 占比 % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for mode in modes:
        for key in ('all','far_shallow','far_shallow/structured_light','far_shallow/active_stereo'):
            for method in methods+('mean4',):
                q=s[mode+'/'+key][method];rel=q['relative_bins_percent_numeric']
                lines.append('| '+' | '.join([mode,key,method,f'{q["missing_frames"]}/{q["frames"]}',f(q['mae_mm']),f(q['p95_mm']),f(q['abs_rel_percent']),f(sum(rel[:3])),f(rel[-1])])+' |')
    for mode in modes:
        for key in ('all','far_shallow/structured_light'):
            lines+=['',f'## {names[mode]} {key} 相对误差分布','',
                '| 方法 | 非空有值帧数 | <0.5% | 0.5–1% | 1–2% | 2–3% | 3–5% | 5–10% | ≥10% |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for method in methods:
                q=s[mode+'/'+key][method];lines.append('| '+' | '.join([method,str(q['relative_evaluable_frames'])]+[f(x) for x in q['relative_bins_percent_numeric']])+' |')
        lines+=['',f'## {names[mode]} 各液深区间退化检查','',
            '| 液深 m | 单帧 MAE mm | 2帧 MAE mm | 4帧 MAE mm | guarded4 MAE mm | 单帧 P95 mm | guarded4 P95 mm | guarded4未输出 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for b in bands:
            q=s[mode+'/'+b];lines.append('| '+' | '.join([b]+[f(q[m]['mae_mm']) for m in methods]+[f(q['single']['p95_mm']),f(q['guarded4']['p95_mm']),str(q['guarded4']['missing_frames'])])+' |')
        for b in bands[1:]:
            lines+=['',f'### {b} m 相对误差分布','',
                '| 方法 | <0.5% | 0.5–1% | 1–2% | 2–3% | 3–5% | 5–10% | ≥10% |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
            for method in methods:
                lines.append('| '+' | '.join([method]+[f(x) for x in s[mode+'/'+b][method]['relative_bins_percent_numeric']])+' |')
    lines+=['','## 配对不确定性与限制','']
    for mode in modes:
        q=audit[mode+'/far_sl']['joint4'];lines.append(f'{mode} 结构光远距离浅液位：4 帧相对单帧 MAE 改善 {q["mae_improvement_mm"]:.2f} mm，按容器配置成组重采样 95% 区间 {q["cluster_bootstrap95_mm"]} mm；{q["clusters"]} 个配置、{q["common_frames"]} 个共同有值时刻。')
    post=a.output/'post-test-audit.json'
    if post.exists():
        runtime=json.loads(post.read_text())
        lines+=['','## 方法选择与串行耗时','',
            '下表统计保守策略最终用了几帧。退回单帧不等于未输出，也不是按真实误差选择结果。','',
            '| 噪声 | 单帧输出数 | 四帧输出数 |','|---|---:|---:|']
        for mode in modes:
            q=runtime['routing'][mode]['frames_used'];lines.append(f'| {mode} | {q.get("1",0)} | {q.get("4",0)} |')
        lines+=['',f'在评估结束后单线程串行计时，共 {runtime["latency_windows"]} 个预定义窗口，排除磁盘读取；重算测量与评估结果一致。','',
            '| 方法 | P50 ms | P95 ms | 最大 ms |','|---|---:|---:|---:|']
        for method in methods:
            q=runtime['latency_ms'][method];lines.append('| '+' | '.join([method]+[f(q[k]) for k in ('p50','p95','max')])+' |')
        lines+=['','4 帧在 30 fps 下的首尾跨度为 100 ms；串行耗时加采集跨度只能估计启动延迟，不能替代包含真实采集、驱动、网络及界面的端到端延迟。计时为 320×180 深度图，不代表高分辨率相机。']
    lines+=['',f'各噪声模式的负液深输出数：{audit["negative_counts"]}。',
        f'按 MAE 点估计检测出的退化区间：{audit["regression_bins"]}。点估计退化不等于统计显著退化，需结合 P95 与配置级置信区间。',
        '联合似然目前使用边缘噪声模型，未把相关观测当作真实独立证据来给出已校准置信度。固定或缓慢漂移的共同误差无法保证通过 4 帧消除。真实硬件的空间/时间相关性与安装误差仍需实测。',
        '本轮为静态普通液面，不证明快速液位变化、相机移动、波动、透明折射或强散射条件下的联合恢复性能。总体相对误差会被较深液位的较大分母改善，必须单看浅液位子集。','',
        '## 服务器复现路径','',f'产物目录：`{a.output}`。逐时刻结果 raw.json，分箱 summary.json，配对统计 paired-audit.json，噪声与源码核验 verification.json。',
        '实现：`src/liquid_depth/ordinary_temporal_v37.py`；相关噪声生成：`src/liquid_depth/correlated_noise_v37.py`；生成、评估、调度与报告脚本均为 scripts 下 ordinary_v37 对应文件。产品默认链路未替换。','']
    (a.output/'report.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps({k:v for k,v in audit.items() if k in ('regression_bins','iid/far_sl','spatial/far_sl','persistent/far_sl')},indent=2))
if __name__=='__main__':main()
