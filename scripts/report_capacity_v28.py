"""Capacity-limited all-frame report, including evaluator-only causal diagnostics."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from capacity_priors_v28 import OUT,DATA,LAYOUTS,summary,dump

FN={'active_stereo':'双目','structured_light':'结构光'}
CN={'transparent':'透明','translucent_mild':'半透明'}
LN={'capacity_uniform':'容量内均匀基线','low_dense':'浅端加密','endpoints':'端点兼顾','wide_uniform':'宽跨度均匀'}
MN={'prior0_current0_labels0':'正常观测/带噪先验标签',
    'prior0_current0_labels1':'仅先验液深标签理想化',
    'prior1_current0_labels0':'仅先验匹配理想化',
    'prior0_current1_labels0':'仅当前帧匹配理想化',
    'prior1_current1_labels0':'先验和当前匹配均理想化',
    'prior1_current0_labels1':'先验匹配和标签理想化',
    'prior0_current1_labels1':'当前匹配和先验标签理想化',
    'prior1_current1_labels1':'全部匹配和标签理想化'}
def fmt(x):return '—' if x is None else f'{x:.3f}'
def passed(r):return r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m']
def rowkey(r):return r['config_id'],r['sample_id']

def comparison(before,after):
    old={rowkey(r):r for r in before};pairs=[(old[rowkey(r)],r) for r in after]
    clusters={}
    for a,b in pairs:clusters.setdefault(a['config_id'],[]).append((a,b))
    values=list(clusters.values());rng=np.random.default_rng(71231);mae=[];ps=[]
    for _ in range(2000):
        sample=[p for i in rng.integers(0,len(values),len(values)) for p in values[i]]
        finite=[(a,b) for a,b in sample if a['absolute_error_m'] is not None and b['absolute_error_m'] is not None]
        if finite:mae.append(float(np.mean([(a['absolute_error_m']-b['absolute_error_m'])*1000 for a,b in finite])))
        ps.append(float(np.mean([int(passed(b))-int(passed(a)) for a,b in sample])*100))
    return dict(fixed=sum(not passed(a) and passed(b) for a,b in pairs),
        new_failures=sum(passed(a) and not passed(b) for a,b in pairs),both_failed=sum(not passed(a) and not passed(b) for a,b in pairs),
        error_reduced=sum(a['absolute_error_m'] is not None and b['absolute_error_m'] is not None and b['absolute_error_m']<a['absolute_error_m'] for a,b in pairs),
        mae_reduction_ci95_mm=np.percentile(mae,[2.5,97.5]).tolist(),pass_increase_ci95_pp=np.percentile(ps,[2.5,97.5]).tolist())

def main():
    raw=json.loads((OUT/'test-raw.json').read_text());rs=raw['records'];policy=json.loads((OUT/'frozen-policy.json').read_text())
    base=[r for r in rs if r['stage']=='capacity_uniform'];new=[r for r in rs if r['stage']=='selected']
    overall=summary(new);far=summary([r for r in new if r['bottom_m']>=6]);transition=comparison(base,new)
    failures=[r for r in new if not passed(r)];bymatch={rowkey(r):r for r in base}
    newfail=[r for r in failures if passed(bymatch[rowkey(r)])]
    selected_fits={f['config_id']:f['prior'] for f in raw['fits'] if f['layout']==policy['selection'][next(r['family']+'/'+r['condition'] for r in new if r['config_id']==f['config_id'])]}
    oracle=json.loads((OUT/'oracle-raw.json').read_text())['records'];om={}
    for r in oracle:om.setdefault(r['stage'],{})[rowkey(r)]=r
    oracles=[]
    for mode,lookup in om.items():
        values=list(lookup.values())
        oracles.append(dict(mode=mode,overall=summary(values),far=summary([r for r in values if r['bottom_m']>=6]),
            fixes_on_normal_failures=sum(passed(lookup[rowkey(r)]) for r in failures)))
    replay_equal=all(new_r['predicted_height_m']==om['prior0_current0_labels0'][rowkey(new_r)]['predicted_height_m'] for new_r in new)
    if not replay_equal:raise RuntimeError('audit baseline does not exactly replay frozen test')
    sens=json.loads((OUT/'sensitivity.json').read_text());sm={rowkey(r):r for r in sens}
    rows=[]
    for r in failures:
        p=selected_fits[r['config_id']];s=sm[rowkey(r)]
        rows.append(dict(r,analysis=s,prior=p,regression_from_baseline=passed(bymatch[rowkey(r)]),
            counterfactual_pass={mode:passed(lookup[rowkey(r)]) for mode,lookup in om.items()}))
    dump(OUT/'failed-frames.json',rows)
    range_groups=[]
    for b in (2.2,6.5,9.8):
        for cap in (.15,.35,.75):
            for stage in ('capacity_uniform','selected'):
                values=[r for r in rs if r['stage']==stage and r['bottom_m']==b and r['container_max_height_m']==cap]
                range_groups.append(dict(bottom_m=b,capacity_m=cap,stage=stage,**summary(values)))
    reason_counts=Counter(reason for r in failures for reason in r['diagnostics'].get('reasons',[]))
    bound_configs=[k for k,p in selected_fits.items() if 'index' in p and (p['index']<=1.205 or p['index']>=1.595)]
    invalid_configs=[k for k,p in selected_fits.items() if 'error' in p]
    failure_evidence=dict(far_failures=sum(r['bottom_m']>=6 for r in failures),
        without_outside_prior_span=sum('outside_prior_span' not in r['diagnostics'].get('reasons',[]) for r in failures),
        region_min=min(r['diagnostics']['regions'] for r in failures),
        region_max=max(r['diagnostics']['regions'] for r in failures),
        failed_frames_in_bound_configs=sum(r['config_id'] in bound_configs for r in failures))
    target_range=[]
    for stage in ('capacity_uniform','selected'):
        for far_only in (False,True):
            values=[r for r in rs if r['stage']==stage and r['true_height_m']>=.1 and (not far_only or r['bottom_m']>=6)]
            target_range.append(dict(stage=stage,far_only=far_only,**summary(values)))
    layout_fit_errors={layout:dict(Counter(str(f['prior']['error']) for f in raw['fits'] if f['layout']==layout and 'error' in f['prior'])) for layout in LAYOUTS}
    groups=[]
    for f in FN:
        for c in CN:
            for stage in ('capacity_uniform','selected'):
                values=[r for r in rs if r['stage']==stage and r['family']==f and r['condition']==c]
                groups.append(dict(family=f,condition=c,stage=stage,overall=summary(values),far=summary([r for r in values if r['bottom_m']>=6])))
    result=dict(overall=overall,far=far,baseline=summary(base),transition=transition,groups=groups,ranges=range_groups,
        failure_count=len(failures),new_failure_count=len(newfail),failure_reasons=dict(reason_counts),oracle_counterfactuals=oracles,
        audit_replays_baseline=True,effective_index_bound_configs=bound_configs,invalid_prior_configs=invalid_configs,
        failure_evidence=failure_evidence,target_range=target_range,layout_fit_errors=layout_fit_errors)
    dump(OUT/'comparison.json',result)
    dump(OUT/'promotion-decision.json',dict(promote=False,reason='No stable independent-test improvement; paired confidence intervals cross zero',production_changed=False))
    lines=['# 透明半透明：实际容器容量约束下的5帧先验验证 V28','',
        '日期：2026-10-11。测量目标为容器底面到液面的实际液深H，容器最大液深为C，相机到底面的距离为B。三者严格区分。','',
        '## 结论概览','',
        f"新独立测试{overall['frames']}组全部计入分母。容量内均匀基线MAE {fmt(summary(base)['mae_mm'])} mm、通过{summary(base)['passed']}/{len(base)}；开发集选定布置MAE {fmt(overall['mae_mm'])} mm、通过{overall['passed']}/{overall['frames']}，数值输出{overall['outputs']}/{overall['frames']}。",
        f"相机远距离子集(B≥6 m)共{far['frames']}组，选定布置MAE {fmt(far['mae_mm'])} mm、AbsRel {fmt(far['absrel_pct'])}%、通过{far['passed']}/{far['frames']}。最终仍有{len(failures)}组超标或无法求解，较基线新增超标{len(newfail)}组。",
        '本轮布置调整未显示稳定收益：全帧通过率由81.53%降至81.25%，MAE仅降低约0.057 mm，两项配对置信区间均跨零。选定组合仅作为验证记录，不推荐为已验证改进、不替换默认算法；也不根据测试结果改选另一布置。',
        '容差保持max(5 mm, 2%×真实H)。本轮无输出置信度或覆盖率筛选；超出容量的有限预测照常评分、记录警告，不静默截断到[0,C]。无法求解也计入通过率分母；MAE/AbsRel仅对有限预测计算。','',
        '## 设计与独立性','',
        '每种布置严格使用5组已知液深先验，所有真液深标定点均在(0,C)内。同一配置相同液深的图像及标签在各布置间复用，所有方案使用完全相同的未知帧。生成多个候选池用于消融，不表示每次拟合用了超过5组。',
        '开发集：B=3/8.5 m，C=20/50 cm，折射率1.325/1.405，32配置×10未知液深=320组；种子589317+263*i+10000*family。开发集未知H/C为4/9/17/26/38/52/64/76/86/97%。',
        '独立测试：B=2.2/6.5/9.8 m，C=15/35/75 cm，折射率1.315/1.445，72配置×10未知液深=720组；种子691753+263*i+10000*family。未知H/C为3.5/7.5/15/23/37/49/61/73/84/97.5%。',
        '测试H范围5.25–731.25 mm。远距离子集480组，其H/B均小于约11.3%。不将相机的9.8 m距离称为9.8 m液深，也不将这些小容器结果直接外推到10 m液深。',
        '选型目标沿用整体与远距离容差归一化误差各0.5，加整体P90误差项0.2及缺测惩罚；只按传感链路与透明/轻度半透明场景全局选型，不按测试真液深逐帧择优。',
        '源码/策略在生成测试集之前冻结。正常测量使用原V26算法；上一轮收益不稳定的V27残余空间场不强制启用。基线也受到相同容量限制，不拿无法获得的高液位先验作现实基线。',
        '保持原有成像、噪声和固定几何以单独检验容量约束；已知液深标签噪声标准差为0.5 mm+0.1%×H。相机/底面几何准确已知，没有另加标定误差。','',
        '## 全部可实现的5点布置','',
        '| 布置 | H/C |','|---|---|']
    for k,values in LAYOUTS.items():lines.append(f"| {LN[k]} | {'、'.join(f'{v*100:g}%' for v in values)} |")
    lines+=['','| 场景 | 开发集选定布置 | C=15 cm时的5个标定液深 mm |','|---|---|---|']
    for k,v in policy['selection'].items():
        f,c=k.split('/');lines.append(f"| {FN[f]}/{CN[c]} | {LN[v]} | {'、'.join(f'{x*150:g}' for x in LAYOUTS[v])} |")
    lines+=['','实际软件仍允许操作员自行选择已知液深，不强制固定百分比；这些比例只是候选实验方案。C必须是可安全实现的最大液深，不是相机距离或容器容量体积。','',
        '## 独立测试：全部帧与远距离子集','',
        '| 布置 | 数值输出/全部 | MAE mm | AbsRel % | P95 mm | 全部通过 | 远距离 MAE mm | 远距离通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for stage in list(LAYOUTS)+['selected']:
        values=[r for r in rs if r['stage']==stage];a=summary(values);b=summary([r for r in values if r['bottom_m']>=6])
        lines.append(f"| {LN.get(stage,'开发集选定组合')} | {a['outputs']}/{a['frames']} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {fmt(a['p95_mm'])} | {a['passed']}/{a['frames']} | {fmt(b['mae_mm'])} | {b['passed']}/{b['frames']} |")
    lines+=['','四个单独布置的测试结果只用于验证，不得看完测试后改选；最终组合仍按已冻结开发集选择。',
        f'各布置先验拟合失败原因：{layout_fit_errors}。浅端加密4配置/40帧、端点兼顾2配置/20帧因非单调标定无法求解；不是被置信度门控删除，仍留在通过率分母。选定组合没有无法拟合的配置。','',
        '### 分场景','',
        '| 场景 | 方法 | 数值输出 | MAE mm | AbsRel % | 全部通过 | 远距离通过 |','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        a=g['overall'];b=g['far'];lines.append(f"| {FN[g['family']]}/{CN[g['condition']]} | {'均匀基线' if g['stage']=='capacity_uniform' else '选定布置'} | {a['outputs']}/{a['frames']} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {a['passed']}/{a['frames']} | {b['passed']}/{b['frames']} |")
    lines+=['','### 距离与容器最大液深交叉分组','',
        '| B m | C cm | 基线 MAE mm | 选定 MAE mm | 基线通过 | 选定通过 |','|---|---:|---:|---:|---:|---:|']
    for b in (2.2,6.5,9.8):
        for c in (.15,.35,.75):
            a=next(r for r in range_groups if r['bottom_m']==b and r['capacity_m']==c and r['stage']=='capacity_uniform')
            n=next(r for r in range_groups if r['bottom_m']==b and r['capacity_m']==c and r['stage']=='selected')
            lines.append(f"| {b:g} | {c*100:g} | {fmt(a['mae_mm'])} | {fmt(n['mae_mm'])} | {a['passed']}/{a['frames']} | {n['passed']}/{n['frames']} |")
    lines+=['','### 按实际液深分组（没有删掉低液位）','',
        '| 实际H范围 | 方法 | MAE mm | AbsRel % | 全部通过 |','|---|---|---:|---:|---:|']
    for lo,hi in [(0.,.05),(.05,.15),(.15,.35),(.35,.751)]:
        for stage in ('capacity_uniform','selected'):
            a=summary([r for r in rs if r['stage']==stage and lo<=r['true_height_m']<hi])
            lines.append(f"| [{lo*1000:g},{hi*1000:g}) mm | {'均匀基线' if stage=='capacity_uniform' else '选定布置'} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {a['passed']}/{a['frames']} |")
    lines+=['','低液位AbsRel的分母很小，必须同时看绝对误差和5 mm容差通过率；不能用小MAE直接宣称所有低液位均达到1%。','',
        '### 原目标量程内的补充结果（H≥10 cm）','',
        '以下只用于区分量程，不替代前面的全部720组主结果；本批最大液深约73 cm，并未验证10 m液深。',
        '| 方法 | 子集 | MAE mm | AbsRel % | 全部通过 |','|---|---|---:|---:|---:|']
    for a in target_range:
        lines.append(f"| {'均匀基线' if a['stage']=='capacity_uniform' else '选定布置'} | {'B≥6 m' if a['far_only'] else '所有B'}，H≥10 cm | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {a['passed']}/{a['frames']} ({a['pass_pct']:.2f}%) |")
    lines+=['',
        '## 同帧改善、退化与抽样不确定性','',
        f"选定方案修复{transition['fixed']}个基线失败，新增{transition['new_failures']}个失败，两者都失败{transition['both_failed']}个；{transition['error_reduced']}组的绝对误差降低。",
        f"72个配置为簇、2000次配对bootstrap：共同有限输出的MAE降低95%区间{transition['mae_reduction_ci95_mm']} mm；全帧通过率增加区间{transition['pass_increase_ci95_pp']}个百分点。正值表示改善，跨零不支持稳定收益。该区间不覆盖仿真到现实的域差。",'',
        '## 剩余失败统计与理想观测对照','',
        f"无法拟合先验的配置数：{len(invalid_configs)}；有效折射率到达拟合边界附近的配置数：{len(bound_configs)}。有效折射率也吸收匹配偏置，不等于已经测得真实材料折射率。",
        f"失败帧警告计数（可重叠）：{dict(reason_counts)}。没有警告也不代表误差合格，范围外警告也不自动证明唯一因果。",'',
        f"其中{failure_evidence['far_failures']}/{len(failures)}个失败发生在B≥6 m；{failure_evidence['without_outside_prior_span']}/{len(failures)}没有标定区间外警告。失败帧仍有{failure_evidence['region_min']}–{failure_evidence['region_max']}个匹配区域，不能简单归因于观测数量太少。",
        f"{len(bound_configs)}/72个配置的有效折射率接近1.2/1.6拟合边界，均为双目；其关联失败帧{failure_evidence['failed_frames_in_bound_configs']}/{len(failures)}。这支持远距离浅液位下折射率与匹配偏置可辨识性不足的解释，但边界命中本身不是独立的因果证明。",'',
        '| 仅供诊断的反事实 | 数值输出 | MAE mm | 全部通过 | 正常失败中被修复 |','|---|---:|---:|---:|---:|']
    for m in oracles:
        a=m['overall'];lines.append(f"| {MN[m['mode']]} | {a['outputs']}/{a['frames']} | {fmt(a['mae_mm'])} | {a['passed']}/{a['frames']} | {m['fixes_on_normal_failures']}/{len(failures)} |")
    lines+=['',
        '这些反事实使用评估真值，只用于分解原因，不属于可部署能力，不参与选型。理想匹配保留原采样位置/区域数，但以Snell真几何替换对应点；它同时移除了匹配噪声及空间块质心近似误差，不能把改善全部归给单一匹配器。',
        '单独理想化先验或当前匹配可能反而变差，因为既有校正吸收了二者的共同偏置，改变一侧会破坏抵消。因此需连同联合理想化和标签噪声对照解释，不能把各表行当作可相加的独立因果贡献。',
        '本链路使用可见底纹/投影穿透观测，不要求深度图里存在大量真实液面回波；不能简单把其失败归因于真实液面点太少。','',
        '仅去除先验液深标签噪声修复4/135个正常失败，整体通过数反而由585变582；标签精度不是本批首要瓶颈。同时理想化先验与测量匹配、仍保留原标签噪声时，MAE约0.410 mm、720/720通过，修复全部135个失败。主要证据指向观测偏置与窄跨度标定耦合，而不是单凭五帧数量不足。',
        '全部理想化误差近零，是前向和逆向共享几何假设的内部一致性结果，不是硬件认证。当前模拟不能判定真实设备已经触及物理极限；理想对应点也不是可部署输入。','',
        '### 几何敏感度（理论局部值，非相机规格）','',
        '| 传感链路 | B m | 0.001像素扰动对应液深变化的中位数 mm | P95 mm |','|---|---:|---:|---:|']
    for f in FN:
        for b in (2.2,6.5,9.8):
            vals=[r['height_mm_per_0_001px'] for r in sens if r['family']==f and r['bottom_m']==b and r['height_mm_per_0_001px'] is not None]
            lines.append(f"| {FN[f]} | {b:g} | {np.median(vals):.3f} | {np.percentile(vals,95):.3f} |")
    lines+=['',
        '上表取真实几何处像素残差对H的RMS导数倒数，描述等效沿敏感方向的扰动；没有把独立像素噪声直接等同于共同偏置，也不表示所有0.001像素误差都会产生相同液深误差。相机远、基线固定时，液深变化带来的视差变化非常小。','',
        '### 误差/容差比最大的20个失败组','',
        '| 配置/帧 | B m | C cm | 真H mm | 估计H mm | 误差 mm | 容差 mm | 较基线新增失败 | 警告 |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    ranked=sorted(rows,key=lambda r:r['absolute_error_m']/r['tolerance_m'] if r['absolute_error_m'] is not None else 1e12,reverse=True)
    for r in ranked[:20]:
        pred=r['predicted_height_m'];error=r['absolute_error_m']
        lines.append(f"| {r['config_id']}/{r['sample_id']} | {r['bottom_m']:g} | {r['container_max_height_m']*100:g} | {r['true_height_m']*1000:.2f} | {fmt(pred*1000 if pred is not None else None)} | {fmt(error*1000 if error is not None else None)} | {r['tolerance_m']*1000:.2f} | {'是' if r['regression_from_baseline'] else '否'} | {','.join(r['diagnostics'].get('reasons',[]))} |")
    lines+=['','其余失败未被删除，完整逐帧数据、反事实通过情况和标定参数均保存在failed-frames.json；全部成功及失败预测在test-raw.json。','',
        '## 证据范围与后续建议','',
        '本轮验证的是最大液深容量限制，不是完整有限尺寸容器。仍采用平底、平液面、无限底纹视野和已知几何；没有侧壁折射、有限容器在远处占像素缩小、动态波浪或真实硬件辐射标定。现实小容器像素不足时可能更差。',
        '继承的轻度散射代理按H/B设置消光和模糊，因此同一实际液深在远处的散射强度并不构成严格物理一致的材料模拟。本轮用于单独研究标定与几何弱信号，不宣称验证真实半透明材料的全量相位/散射能力；未混入强散射压力样本。',
        '视差精度与五帧标定可辨识性应分开处理。可进一步比较“空容器一帧+四个已知液深”的同预算设计，利用空底观测约束固定偏置；现有算法尚不接受零液深先验，本轮没有假装验证这一路线。',
        '若匹配误差主导，优先研究跨已知先验图的相对像素位移与低维局部校正、降低固定/相位共同偏移；更多拟合参数未必有效。必要时改变成像分辨率、焦距或基线，但需新的设备可行性与独立验证，不能用先验凭空补足几何信息。',
        '结构光每组为30次曝光，双目每组为一对图；没有在本轮证明采集到显示的端到端500 ms。默认工业链路及安全门控未更改，研究输出qualified=false。','',
        '## 复现与归档','',
        '开发：scripts/capacity_priors_v28.py generate_dev → select。冻结测试：scripts/run_capacity_v28.py。理想观测诊断：scripts/audit_capacity_v28.py（仅评估）。报告：scripts/report_capacity_v28.py。',
        '数据目录：'+str(DATA),
        '评估目录：'+str(OUT),
        'test-raw.json：全部帧预测及先验参数；failed-frames.json：全部剩余失败及原因对照；comparison.json：分场景/距离/容量统计；sensitivity.json：逐帧观测偏差与局部敏感度。']
    status=json.loads((OUT/'progress.json').read_text());log=(OUT/'regression.log').read_text()
    lines+=['',f"正常测试与诊断基线逐帧精确复现：{replay_equal}。冻结源码/策略一致性：{status.get('source_and_policy_unchanged')}。回归摘要：{log.strip().splitlines()[-1]}。"]
    text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text,encoding='utf-8')
    Path('docs/transparent_capacity_v28.md').write_text(text,encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
