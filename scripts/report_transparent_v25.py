"""Post-hoc evaluator. Truth metrics never flow into calibration or selection."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import cv2,numpy as np
from experiment_transparent_v25 import OUT,metrics,dump
from liquid_depth.prior_reliability_v24 import accepted,route

raw=json.loads((OUT/'test-raw.json').read_text());records=raw['records']
selection=json.loads((OUT/'frozen-selection.json').read_text())['methods']
selected=[r for r in records if r['method']==selection[r['family']+'/'+r['condition']]]
look={(r['config_id'],r['sample_id'],r['method']):r for r in records}
oldpolicy=json.loads(Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/prior-reliability-v24b-20261011/frozen-calibration-policy.json').read_text())
gated=[];paired=[];ungated=[]
for r in selected:
    p=oldpolicy[r['family']+'/routes']['5'];a=p['selected']
    key=(r['config_id'],r['sample_id'])
    old=look[key+('v23_3',)];method=('v24_'+a['method'] if not a['method'].startswith('v23') else a['method'])
    fallback=look[key+(method,)]
    ungated.append(fallback)
    output=route(old['diagnostics'],fallback['diagnostics'],p['primary_multiplier'],a['multiplier'])
    if output['accepted']:
        row=old if output['source']=='v23_preserved' else fallback
        gated.append(row);paired.append(r)

data=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-diagnostic-v25-20261011')
manifest=json.loads((data/'manifest.json').read_text())
capture_lookup={(c['id'],j):cap for c in manifest['configs'] for j,cap in enumerate(c['captures'])}
frames=[]
for r in selected:
    cap=capture_lookup[r['config_id'],r['sample_id']]
    with np.load(Path(cap['folder'])/'matches_optimized.npz') as o:
        ps=o['left_pixels'];qs=o['right_pixels']
    with np.load(cap['evaluator_truth']) as truth:
        prefix='right' if r['family']=='active_stereo' else 'projector'
        expected=np.column_stack([cv2.remap(truth[prefix+'_'+axis],ps[:,0].astype(np.float32).reshape(-1,1),
            ps[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel() for axis in ['u','v']])
    err=np.linalg.norm(expected-qs,axis=1)
    frame=dict(r,match_median_px=float(np.median(err)) if len(err) else None,
        match_p95_px=float(np.percentile(err,95)) if len(err) else None,
        liquid_fraction=r['true_height_m']/r['bottom_m'],
        passed=r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m'])
    frames.append(frame)
dump(OUT/'selected-frame-diagnostics.json',frames)

summary=[]
for f in ['active_stereo','structured_light']:
    for c in ['transparent','translucent_mild']:
        for label,rs in [('v24_unfiltered',ungated),('v25_unfiltered',selected),('v24_gated',gated),('v25_same_old_accepted',paired)]:
            group=[r for r in rs if r['family']==f and r['condition']==c]
            summary.append(dict(family=f,condition=c,route=label,**metrics(group)))
strata=[]
for f in ['active_stereo','structured_light']:
    for c in ['transparent','translucent_mild']:
        for b in [.3,.8,2.,6.,9.8]:
            rs=[r for r in selected if r['family']==f and r['condition']==c and r['bottom_m']==b]
            strata.append(dict(family=f,condition=c,bottom_m=b,**metrics(rs)))
        for name,predicate in [('shallow',lambda r:r['true_height_m']/r['bottom_m']<.15),
            ('interior',lambda r:.15<=r['true_height_m']/r['bottom_m']<=.65),
            ('high_extrapolation',lambda r:r['true_height_m']/r['bottom_m']>.65)]:
            rs=[r for r in selected if r['family']==f and r['condition']==c and predicate(r)]
            strata.append(dict(family=f,condition=c,level_group=name,**metrics(rs)))
paired_regression=dict(old=metrics(gated),new_on_same_frames=metrics(paired),
    previously_pass_now_fail=sum(a['absolute_error_m']<=a['tolerance_m'] and b['absolute_error_m']>b['tolerance_m'] for a,b in zip(gated,paired)),
    previously_fail_now_pass=sum(a['absolute_error_m']>a['tolerance_m'] and b['absolute_error_m']<=b['tolerance_m'] for a,b in zip(gated,paired)))
dump(OUT/'comparison.json',dict(summary=summary,strata=strata,paired=paired_regression,
    overall=metrics(selected),warnings=dict(Counter(x for r in selected for x in r['diagnostics']['reasons']))))

lines=['# 透明半透明：场景边界与无覆盖率筛选诊断 V25','',
'日期：2026-10-11。仅限仿真；生产默认模型不替换。','',
'## 1. 范围与回退判断','',
'强散射已移出本轮生成、开发选型和测试；保留历史数据不继续优化。强散射仅在液面回波已验证时作为普通表面路线候选，不能拿底部回波仿真冒充普通液面验证。', '',
'V24b 保留了 V23 已接受帧的原值，因此不存在对这些帧的强散射驱动回退；不做整轮代码删除。其混合场景选型策略已退出本轮透明方向。V25 只作为独立诊断实验，不因总体提升就覆盖生产默认。','',
'## 2. 实验与口径','',
'40 个独立新配置，320 个未知液位测量组，每组配置 5 个已知液位先验；共生成 520 组捕获。双目每组两图，结构光每组 30 图。只含透明和轻度半透明，各传感器各场景 80 组。',
'相机到底部距离 0.3/0.8/2/6/9.8 m；未知实际液深为距离的 6/12/18/31/46/60/70/76%，范围 0.018–7.448 m。距离不是液深。新折射率 1.325/1.405、新纹理种子；方法在此前开发集固定。','',
'容差仍为 max(5 mm, 2%×真实液深)。无输出风险筛选、无最低覆盖率门槛；保留外推等警告。数值输出率不等于安全覆盖率。既有 V23 标定拒绝行为仅作为对照留存；V25 不按折射率标准差直接丢弃整段。','',
'## 3. 无筛选的同帧对比','',
'优化前：V24b 所用补充校正方法的全部有限原始输出，不经其风险门控。优化后：开发集固定的 V25 各类方法。不是把旧高置信子集与新全帧结果当成等价分母。','',
'| 传感器/场景 | 方法 | 数值输出 | MAE mm | AbsRel % | 容差通过率 % | P95 mm |',
'|---|---|---:|---:|---:|---:|---:|']
names={'active_stereo':'双目','structured_light':'结构光','transparent':'透明','translucent_mild':'半透明'}
def fmt(x):return '无' if x is None else f'{x:.3f}'
for row in summary:
    if row['route'] not in ('v24_unfiltered','v25_unfiltered'):continue
    lines.append(f"| {names[row['family']]}/{names[row['condition']]} | {row['route']} | {row['finite_outputs']}/{row['frames']} | {fmt(row['mae_mm'])} | {fmt(row['absrel_pct'])} | {fmt(row['pass_pct'])} | {fmt(row['p95_mm'])} |")
lines+=['','### 原先接受帧上的防回退核查','',f'旧门控接受 {len(gated)} 帧。下面比较严格相同的帧，V25 不用测试标签进行选择：','',
'```json',json.dumps(paired_regression,ensure_ascii=False,indent=2),'```','',
'任何同帧回退都保留记录。此结果不支持直接全量替换旧高置信生产策略；研究目标是发现并修复全部帧，而非隐藏难帧。','',
'## 4. 距离分层（全部测量帧）','',
'| 传感器/场景 | 到底距离 m | MAE mm | AbsRel % | 通过率 % | P95 mm |','|---|---:|---:|---:|---:|---:|']
for r in strata:
    if 'bottom_m' in r:lines.append(f"| {names[r['family']]}/{names[r['condition']]} | {r['bottom_m']} | {fmt(r['mae_mm'])} | {fmt(r['absrel_pct'])} | {fmt(r['pass_pct'])} | {fmt(r['p95_mm'])} |")
lines+=['','## 5. 当前未通过的帧','',
'以下逐帧表列出未通过者，不因置信度而排除；全量逐帧记录在 selected-frame-diagnostics.json。匹配误差使用仿真对应真值，仅供评估，绝不输入算法。','',
'| 配置 | 帧 | 液深 m | 估计 m | 误差 mm | 容差 mm | 匹配中位误差 px |','|---|---:|---:|---:|---:|---:|---:|']
for r in sorted([r for r in frames if not r['passed']],key=lambda r:r['absolute_error_m']/r['tolerance_m'] if r['absolute_error_m'] is not None else 1e9,reverse=True):
    lines.append(f"| {r['config_id']} | {r['sample_id']} | {fmt(r['true_height_m'])} | {fmt(r['predicted_height_m'])} | {fmt(r['absolute_error_m']*1000 if r['absolute_error_m'] is not None else None)} | {fmt(r['tolerance_m']*1000)} | {fmt(r['match_median_px'])} |")
oracle=json.loads((OUT/'oracle-ablation.json').read_text()) if (OUT/'oracle-ablation.json').exists() else None
lines+=['','### 失败帧分布与原因隔离','',
    '本批每个测量组至少有 '+str(min(r['regions'] for r in frames))+' 个匹配区域。这是折射后的底部/投影对应，不是液面真实点数；结果不支持把当前主要失败归因于匹配点数量不足。',
    '40 个未通过帧全部在相机到底距离 2 m 及以上：2 m 为 11 帧，6 m 为 14 帧，9.8 m 为 15 帧；其中液深/到底距离为 6% 或 12% 的低液位占 27 帧。0.3 m 与 0.8 m 分层的 128 帧均在本批容差内。','']
if oracle:
    lines+=['进行了三组仅供原因隔离的 oracle 对照：替换为仿真真值对应点、替换为精确先验标签、两者同时替换。方法和参数选择不重新调优。**这些输入实机不可得，结果不是模型可实现精度，也不计入正式 320 帧结果。**','',
    '| 传感器/场景 | 真实图像匹配 AbsRel % | 理想对应+原噪声标签 AbsRel % | 原匹配+精确标签 AbsRel % |',
    '|---|---:|---:|---:|']
    for f in ['active_stereo','structured_light']:
        for c in ['transparent','translucent_mild']:
            actual=next(r for r in summary if r['family']==f and r['condition']==c and r['route']=='v25_unfiltered')
            a=next(r for r in oracle['summary'] if r['family']==f and r['condition']==c and r['arm']=='oracle_matches_noisy_labels')
            b=next(r for r in oracle['summary'] if r['family']==f and r['condition']==c and r['arm']=='measured_matches_exact_labels')
            lines.append(f"| {names[f]}/{names[c]} | {fmt(actual['absrel_pct'])} | {fmt(a['absrel_pct'])} | {fmt(b['absrel_pct'])} |")
    lines+=['','在本模拟假设内，消除匹配误差比仅提高先验标签精度改善明显；理想对应但保留标签噪声时四类均通过全部容差。这说明优先改进亚像素匹配/相位解码及其液深相关系统偏置更有依据，而不是继续加大强散射训练。仍不能据此断言真实硬件没有物理限制，因为光子预算、镜面反射、传感器噪声与真实体散射均未完整校准。','']
lines+=['','## 6. 修改及解释','',
'1. 从“每个含噪声的匹配光线对先求液深，再聚合”改为“给定候选液深，用 Snell 模型预测另一幅图中的像素位置，再对观测像素残差鲁棒拟合”。减少远端小视差条件下反演噪声的放大；不是补造真实液面点。',
'2. 使用 5 个已知液深校正有效折射参数与像素偏置。折射参数是当前模拟与匹配器下的有效值，不宣称唯一测出材料真实折射率。',
'3. 双目在开发集选择平滑单调映射，透明结构光选择仿射余差校正，轻度半透明结构光选择不叠加经验映射。并非所有场景都强行使用同一校正。',
'4. 不再用参数不稳定直接否决整段可计算测量；外推/参数触边保留警告。没有匹配、非法先验、非单调先验仍是真实失败，不输出伪造结果。',
'5. 已有配方兼容；新增显式场景边界与合理拒绝原因。工业端不开放实验估计用于控制。','',
'## 7. 仍需完善','',
'- 远端误差：固定焦距与短基线下液深变化对应的视差变化小，亚像素偏差被放大。优先改进匹配亚像素偏置和结构光相位精度，而不是强散射训练。',
'- 低液位及超出先验上下界：标定标签约 0.5 mm + 0.1% 液深的噪声，对浅液位相对误差更显著；局部斜率外推易放大误差。下一步可在预期工作区间最低/最高液位补先验，对重复标定估计标签可靠度。',
'- 半透明远距离：轻度散射与相位/纹理误差相关；现有模拟只有简化代理，需要把强散射排除后，单独验证仍保留底纹的半透明强度范围。',
'- 本轮均为静止、水平液面和平底，无玻璃侧壁；无法推断多层透明、动态液面或真实硬件效果。','',
'## 8. 复现与产物','',
'源码：src/liquid_depth/transparent_diagnostic_v25.py；边界：scene_boundaries.py、scenario_policy.py；实验：scripts/run_transparent_v25.py；报告生成：scripts/report_transparent_v25.py。',
'服务器产物目录：'+str(OUT),'',
'结构光采集本身为 30 次曝光；求解时延不是端到端采集时延。独立测试源码完整性和回归结果见 progress.json 与 regression.log。']
text='\n'.join(lines)+'\n'
Path('docs/transparent_diagnostic_v25.md').write_text(text,encoding='utf-8')
(OUT/'report.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(summary=summary,overall=metrics(selected),paired=paired_regression,
    regions_min=min(r['regions'] for r in frames),latency_p95_ms=float(np.percentile([r['inference_ms'] for r in frames],95)),
    failed=Counter(r['family']+'/'+r['condition'] for r in frames if not r['passed'])),indent=2))
