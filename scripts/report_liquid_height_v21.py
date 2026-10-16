#!/usr/bin/env python3
"""Build current metric-contract report from archived per-frame measurements."""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
def load(name):return json.loads((root/name).read_text())
def value(x,percent=False):return '—' if x is None else f'{x*(100 if percent else 1):.2f}'
def row(label,m):return '| '+' | '.join([label,value(m['mae_mm']),value(m['p95_mm']),value(m['abs_rel'],True),
    value(m['tolerance_pass_rate'],True),value(m['output_coverage'],True),value(m['effective_pass_coverage'],True)])+' |'
header='| 场景或方法 | MAE mm | P95 mm | AbsRel % | 输出内通过率 % | 输出覆盖率 % | 全帧有效通过率 % |\n|---|---:|---:|---:|---:|---:|---:|'
names={'ordinary':'普通','transparent':'透明','translucent':'半透明','glare':'眩光','low_light':'低照',
    'floating_objects':'漂浮物','uneven_surface':'不平整','depth_failure':'深度失效全部强度','multilayer':'多层透明','compound':'复合退化'}
sections=['# 实际液深统一复评与透明半透明优化报告','日期：2026-10-11。指标契约：v3。',
'''本轮将主评价量统一为相对水平容器底面、沿重力方向的面积平均实际液深 H。
历史相机轴向深度代理的“达标”结论撤回；旧数值仅保留作深度修复诊断。
以下均为仿真研究结果，不是实机工业验收；生产默认模型和复杂场景禁用策略未升级。

## 指标和标定条件

每个像素按实测单位深度和内参反投影，再通过 OpenCV 相机到重力世界坐标变换得到点高度。
液深为点高度减底面高度，不是相机距离。Blender 到 OpenCV 的 y 和 z 轴翻转已明确实现并测试。
标定在本轮使用仿真提供的理想内参、位姿、水平底面和固定容器占地轮廓；未计入现场标定误差。
轮廓由已知容器几何提供，不是由真值液面掩码提供；真实部署需将此容器参照加入空容器标定流程。

预测只使用模型掩码，不与真值掩码求交。固定世界 XY 网格内取点中位数，再用四邻点距离加权插值
估计均匀占地网格的液深场。平均值为主评价量，最浅、最深、P05、P95另报；插值不能证明未观测波面可恢复。
接受条件固定为至少64个预测支持点、至少4个空间单元、覆盖至少12%的占地单元及正液深。
这只是宽松研究输出条件，不能排除错误回波和错误液面身份，不是安全生产资格门控。

- MAE、AbsRel、P95只统计接受且可评估输出，AbsRel的分母为每帧真实平均液深 H。
- 每帧容差为 max(5 mm, 2%×H)，不使用相机轴向距离 Z。
- 输出覆盖率为接受帧／全部帧；可评估输出率为可评估接受帧／接受帧。
- 输出内通过率为容差内帧／可评估接受帧；全帧有效通过率为容差内接受帧／全部帧。
- 无输出不记零误差。当前仿真真值齐全，可评估率高不代表现场可观测性高。

工程目标保持不变：普通 AbsRel≤1.5%、通过率≥90%、覆盖率≥95%、可评估率≥99%；
困难支持场景分别为3%、75%、80%、98%。不把仿真候选准入线当产品精度承诺。
多层透明、复合和严重／极端失效仍属于未支持条件，主要考核超容差误接受≤1%和明确拒绝原因，不能套用困难支持场景门槛后宣称上线。

## 全场景统一回归

使用既有1196帧测试划分，属于历史独立测试集的重新评价，不是本轮新盲测。
旧V3输入的非液面深度背景已在数据生成时掩蔽，属于简化输入，不能证明全帧实机泛化；新透明半透明批次另采用全帧RGB-D。
不同模型采用相同几何汇总和宽松输出门控。V8只适合普通可靠回波场景，复杂场景行是诊断对照，
不能因此将V8作为全部场景的默认模型。V17也仅为透明半透明研究方案，不因跨场景回放而获得生产资格。
''']
for variant in ('v6','v8','v17','raw'):
    report=load(f'all-scenes-{variant}.json')
    sections+=['### '+variant.upper(),header]+[row(names[s],m) for s,m in report['by_scenario'].items()]
    qualified=[]
    for s,m in report['by_scenario'].items():
        if s in ('compound','multilayer','depth_failure'):continue
        limits=(.015,.90,.95,.99) if s=='ordinary' else (.03,.75,.80,.98)
        if (m['abs_rel'] is not None and m['abs_rel']<=limits[0] and
            m['tolerance_pass_rate']>=limits[1] and m['output_coverage']>=limits[2] and
            m['evaluable_output_rate']>=limits[3]):qualified.append(names[s])
    sections.append('本方法在这批仿真中同时满足对应数值门槛的场景：'+('、'.join(qualified) if qualified else '无')+'。这不是完整产品链路或实机资格结论。')
sections+=['前三种网络修复结果均未同时达到对应工程目标，不能沿用历史相机深度代理的达标标签。RAW使用V8预测掩码及原始深度，不作网络深度修复，和V8构成同掩码对照；它不是完整时序安全链路。深度失效行合并多个强度，只作总览，不代表95%—100%失效已能处理。']
sections+=['## 目标实际液深范围内的历史测试子集',
'''全1196帧中有786帧真实平均液深处于0.1—10m，其余保留为超浅压力测试，不混同工程目标范围。
下面不改变模型输出或门控，只按真值范围事后分层评分；分层不是给推理提供真值。
该子集也不代表实际液深直到10m均匀覆盖。RAW漂浮物在全样本平均中数值过线，但在目标范围子集不满足，不能宣称整个目标量程已适配。
''']
from liquid_depth.liquid_height_metrics import summarize_frames
target_summaries={}
for variant in ('v6','v8','v17','raw'):
    report=load(f'all-scenes-{variant}.json');target_summaries[variant]={}
    sections+=['### '+variant.upper()+'：0.1—10m已采液深子集',header]
    for s in report['by_scenario']:
        records=[x for x in report['records'] if x['scenario']==s and .1<=x['reference']['mean_m']<=10]
        m=summarize_frames(records);target_summaries[variant][s]=m
        sections.append(row(names[s]+f' (n={len(records)})',m))
(root/'all-scenes-target-height-summary.json').write_text(json.dumps(target_summaries,indent=2))
sections+=['## 未支持条件的误接受诊断',
'''下表只诊断研究宽松输出门控，没有更改生产拒绝策略。超容差误接受率=接受但超出实际液深容差的帧／该条件全部帧。
95%—100%失效分组的真值区域有效比例仅作离线标注，不用于推理或门控。这些组仍应拒绝，不作为继续恢复优化目标。
''',
'| 方法与条件 | 帧数 | 输出覆盖率 % | 超容差误接受／全帧 % |',
'|---|---:|---:|---:|']
for variant in ('v6','v8','v17','raw'):
    report=load(f'all-scenes-{variant}.json')
    for s,label in (('compound','复合退化'),('multilayer','多层透明'),('extreme95','95%—100%失效')):
        records=[x for x in report['records'] if (x['scenario']==s if s!='extreme95' else
            x['scenario']=='depth_failure' and x['raw_valid_fraction_on_target']<=.05)]
        n=len(records);accepted=sum(x['accepted'] for x in records)
        bad=sum(x['accepted'] and (not x['evaluable'] or not x['within_tolerance']) for x in records)
        sections.append('| '+' | '.join([variant.upper()+' '+label,str(n),
            value(accepted/n,True) if n else '—',value(bad/n,True) if n else '—'])+' |')
v8=load('all-scenes-v8.json');oracle=v8['offline_truth_geometry_sanity']
sections+=['''## 几何流程自检

离线将真值深度与真值掩码通过同一几何汇总器，仅用于检查坐标和统计正确性，绝不作为模型输入。
''',f"1196帧自检平均液深 MAE 为 {oracle['mae_mm']:.5f} mm，P95为 {oracle['p95_mm']:.5f} mm，最大误差为 {oracle['maximum_mm']:.3f} mm。它不是模型性能或实机标定精度。",
'''## 透明半透明校准头实验

冻结V17，训练有界的小型残差头。输入仅含预测／原始点液深统计、支持率、空间覆盖及已标定底面距离；
不输入真实液深、材质标签或真值掩码。训练用768帧，验证按序列哈希分为选轮次和审计部分。
损失用真实液深容差归一化并强调尾部。空间支持门控不因真值变化。
''',header]
for batch in ('a','b'):
    for variant in ('v17','v21'):
        r=load(f'fresh-{batch}-{variant}.json')
        sections.extend(row(f'历史批次{batch.upper()} {names[s]} '+('V17' if variant=='v17' else '残差头'),m) for s,m in r['by_scenario'].items())
sections+=['残差头未获得透明场景稳定收益，不能只凭半透明MAE小幅下降上线；保留负面实验，不提升为默认。']
confidence=load('confidence-height-v8-engineering-final.json')
sections+=['''## 原置信度方案按实际液深重新校验

预先指定0.30、0.50、0.70、0.90四个阈值，仅在验证集选择；随后在1196帧历史测试划分复评。
本次选阈值与上述普通／困难工程目标一致；旧3%／50%／30%／90%研究准入线的诊断JSON另存，不作为工程选模条件。
深度失效分组已改为预测液面掩码内的原始深度有效比例，不再借助标注区域，接受门控和阈值查找均不使用真值掩码。
验证集候选资格不等于测试通过或硬件资格。下表是各场景单独应用其候选阈值的诊断，
不是把所有候选都启用后的系统覆盖率。旧置信度报告不得继续作为实际液深资格依据。
''',header]
for s,m in confidence['evaluation']['by_scenario'].items():
    m=m['metrics']
    mapped=dict(mae_mm=None if m['mae_m'] is None else m['mae_m']*1000,
        p95_mm=None if m['p95_m'] is None else m['p95_m']*1000,abs_rel=m['abs_rel'],
        tolerance_pass_rate=m['within_tolerance_rate'],output_coverage=m['coverage'],
        effective_pass_coverage=m['effective_pass_coverage'])
    sections.append(row(s,mapped))
sections+=['所有场景仍须同时核对误差、容差通过率和覆盖率；仅筛掉困难帧不能视作液深能力提升。']
qualified=[s for s,v in confidence['policy'].items() if v['qualified']]
sections+=['本次验证候选满足工程选模门槛的场景：'+('、'.join(qualified) if qualified else '无')+'。',
f"若仅允许合格候选运行，该研究置信度策略在测试集输出覆盖率为 {value(confidence['evaluation']['overall']['coverage'],True)}%；不是上表逐组候选的覆盖率之和。"]
frozen=load('network-finetune/frozen.json')
sections+=['''## 实际液深目标的网络微调

第二轮将容差归一化的液深损失直接用于深度网络。冻结定位模块，使用分材料、相机距离和真实液深均衡的训练抽样。
损失包含垂直点误差、与评估一致的可微空间汇总平均液深误差以及最差20%帧损失；监督标签只参与训练损失。
采集、区域支持和推理不使用真实液面标签。验证第0折选轮次，并要求两类材料的覆盖率、全帧有效通过率不明显回退。
''',f"训练完成6轮，验证选中第 {frozen['selected_epoch']} 轮。选定权重 SHA256：`{frozen['checkpoint_sha256']}`。",
('第0轮即原始V17回退：六轮微调未满足预先规定的改进与防回退条件。以下“液深微调”列实际采用冻结回退权重，与原始模型相同，不是优化收益。' if frozen['selected_epoch']==0 else '模型选择只使用开发验证集，不根据以下测试结果重新挑轮次。'),
'''训练集仅768帧、开发选择验证集69帧。单项MAE可以下降但浅液体相对误差和尾部容差仍回退；
因此不能用下降的绝对MAE替代通过率防回退条件。按距离和材料观察如下验证轮次，均不使用新测试选模。''',
'| 验证轮次 | 材料 | MAE mm | AbsRel % | 全帧有效通过率 % |',
'|---|---|---:|---:|---:|']
for epoch in range(7):
    validation=load(f'network-finetune/val-epoch{epoch}.json')
    for s,m in validation['by_scenario'].items():
        sections.append('| '+' | '.join([str(epoch),names[s],value(m['mae_mm']),value(m['abs_rel'],True),value(m['effective_pass_coverage'],True)])+' |')
sections+=[
'''新生成128个顶部相机全帧场景，种子20261011071，透明／半透明各64个，每材料每相机距离段16个。
新测试只在权重冻结后评估。历史A／B批次各256个场景另作回归，不把它们再称为新盲测。
''',header]
comparisons=[]
for dataset in ('new-c','historical-a','historical-b'):
    base=load(f'height-network-tests/{dataset}-baseline.json')
    cand=load(f'height-network-tests/{dataset}-height-aware.json')
    assert [r['sequence_id'] for r in base['records']]==[r['sequence_id'] for r in cand['records']]
    for s in ('transparent','translucent'):
        sections+=[row(dataset+' '+names[s]+' V17',base['by_scenario'][s]),row(dataset+' '+names[s]+' 液深微调',cand['by_scenario'][s])]
        pairs=[(x,y) for x,y in zip(base['records'],cand['records']) if x['scenario']==s]
        common=[(x,y) for x,y in pairs if x['evaluable'] and y['evaluable']]
        if common:
            delta=np.array([y['absolute_error_m']-x['absolute_error_m'] for x,y in common])*1000
            rng=np.random.default_rng(20261011);boot=np.mean(delta[rng.integers(len(delta),size=(2000,len(delta)))],axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5])
        else:lo=hi=None
        b,c=base['by_scenario'][s],cand['by_scenario'][s]
        comparisons.append(dict(dataset=dataset,scenario=s,common_frames=len(common),
            common_mae_delta_mm=float(delta.mean()) if common else None,paired_bootstrap_95_mm=[float(lo),float(hi)] if common else None,
            effective_pass_delta_pp=(c['effective_pass_coverage']-b['effective_pass_coverage'])*100,
            coverage_delta_pp=(c['output_coverage']-b['output_coverage'])*100))
(root/'paired-height-network-comparison.json').write_text(json.dumps(comparisons,indent=2))
sections+=['### 新测试分相机距离表现',
'这里0.1—10m是相机至可见液面的轴向距离分组，不是实际液深分组；每材料每距离段只有16帧。',header]
for variant in ('baseline','height-aware'):
    r=load(f'height-network-tests/new-c-{variant}.json')
    for s in ('transparent','translucent'):
        for band in ('0.1-0.3','0.3-1','1-3','3-10'):
            from liquid_depth.liquid_height_metrics import summarize_frames
            m=summarize_frames([x for x in r['records'] if x['scenario']==s and x['range_band']==band])
            sections.append(row(names[s]+' '+band+'m '+variant,m))
sections+=['### 新全帧场景的原始点对照',
'RAW使用V8定位掩码；V17-RAW使用与冻结V17完全相同的专用定位器及掩码，只保留原始深度。后者是本轮追加的诊断对照，不参与冻结模型选择。',header]
for filename,label in (('new-c-raw.json','V8掩码 RAW'),('new-c-v17-raw.json','V17掩码 RAW')):
    raw_test=load(filename)
    for s,m in raw_test['by_scenario'].items():
        sections.append(row(names[s]+' '+label,m))
sections+=['同V17掩码对照显示已有修复网络能缓解错误回波，但修复后仍有明显残差；这属于现有V17相对原始点的收益，不是本轮新增微调的收益。接受集合可能不同，应同时核对全帧有效通过率和覆盖率，不能只比较被接受子集MAE。']
sections+=['### 新测试按实际液深分组',
'本批真实液深约0.015—4.00m，不能据此宣称实际液深0.1—10m全范围已验证。下表保留不足0.1m的压力样本，并单独列出目标范围内已采到的子集，不删除失败样本。',header]
depth_test=load('height-network-tests/new-c-height-aware.json')
for s in ('transparent','translucent'):
    for low,high,label in ((0,.1,'H<0.1m'),(.1,.3,'H=0.1—0.3m'),(.3,1,'H=0.3—1m'),(1,3,'H=1—3m'),(3,10,'H=3—10m'),(.1,10,'H目标范围已采子集')):
        records=[x for x in depth_test['records'] if x['scenario']==s and low<=x['reference']['mean_m']<high]
        sections.append(row(names[s]+' '+label+f' (n={len(records)})',summarize_frames(records)))
sections+=['### 波面深浅端补充诊断',
'下表是新128帧测试冻结回退／候选权重的面积场极值误差，不用于替代主指标平均液深；插值最浅最深仍可能遗漏未观测峰谷。',
'| 材料 | 最浅液深 MAE mm | 最深液深 MAE mm | P05液深 MAE mm | P95液深 MAE mm |',
'|---|---:|---:|---:|---:|']
extrema=load('height-network-tests/new-c-height-aware.json')
for s,m in extrema['by_scenario'].items():
    records=[x for x in extrema['records'] if x['scenario']==s and x['evaluable']]
    sections.append('| '+' | '.join([names[s],value(m['min_mae_mm']),value(m['max_mae_mm']),
        value(np.mean([x['extrema_error_m']['p05_m'] for x in records])*1000) if records else '—',
        value(np.mean([x['extrema_error_m']['p95_m'] for x in records])*1000) if records else '—'])+' |')
sections+=['### 配对比较',
    '| 数据与材料 | 共同可评估帧 MAE差 mm | 配对bootstrap 95%区间 mm | 全帧有效通过率变化 百分点 | 覆盖率变化 百分点 |',
    '|---|---:|---|---:|---:|']
for c in comparisons:
    ci=c['paired_bootstrap_95_mm']
    sections.append('| '+' | '.join([c['dataset']+' '+names[c['scenario']],value(c['common_mae_delta_mm']),
        '—' if ci is None else f'[{ci[0]:.2f}, {ci[1]:.2f}]',value(c['effective_pass_delta_pp']),value(c['coverage_delta_pp'])])+' |')
new=load('height-network-tests/new-c-height-aware.json')
good=all(m['abs_rel'] is not None and m['abs_rel']<=.03 and m['tolerance_pass_rate']>=.75 and m['output_coverage']>=.8
         and m['evaluable_output_rate']>=.98 for m in new['by_scenario'].values())
sections+=['''## 结论与下一步
''',('新测试两类材料满足困难场景仿真数值门槛，但未完成实机资格验证。' if good else
        '新测试没有使两类材料同时满足困难场景工程数值门槛，不提升为生产默认。'),
'''顶部近垂直视角下 H≈相机至底面距离−相机至液面距离。因此同样的深度误差会被浅液体较小的H分母放大：
例如距离液面2m、液深0.10m，液面测距误差1%约20mm，对液深就是20%，不是1%。
本轮即使理想底面标定仍未达标，说明不能把差距仅归因于底面标定；模型修复／锚点偏置、
浅液体相对精度及困难回波泛化都仍需改善。仿真只能诊断当前模拟传感器和模型，不能据此断言真实RGB-D硬件的物理极限。
''',
'''不能仅以MAE下降判断成功，需同时看相对误差、P95、通过率和覆盖率。若微调未稳定获益，应保留冻结V17，
而不是按测试真值挑模型或放宽拒绝门槛。下一阶段优先扩展远距离浅液体和近距离半透明训练分布，
处理稳定错误回波与尺度偏置；将真实空容器标定误差、底面不水平和相机位姿误差纳入独立压力测试。
普通场景保留可靠原始点几何路线，避免透明微调污染普通路线。不以本轮替代95%至100%失效拒绝策略。

本轮没有重跑V19／V20双层折射逆问题。其上层厚度是真实长度指标，但与本轮自由液面总液深不同；
原有约21%强界面输出覆盖率和超过0.5秒的时延局限仍成立，不能合并成通用液深精度。

此前 temporal_depth_failure_v2、range_independent_verification_v3 等实验已经使用相对底面的真实液深。
其限定材质、量程、先验和时序窗口内的结论并不因本次纠正相机距离代理而自动撤回，
也不能与本次单帧、面积插值、宽松门控的结果直接混为同一产品资格结论。本轮未重新训练时序模块。

## 复现与文件

评估模块：src/liquid_depth/liquid_height_metrics.py；入口：scripts/evaluate_liquid_height_v21.py。
原置信度脚本 scripts/calibrate_scenario_confidence.py 已改为实际液深，旧JSON不覆盖。
网络微调：scripts/train_liquid_height_network_v21.py；独立无标签推理：scripts/predict_liquid_height_v21.py。
本轮全部逐帧结果、权重、训练历史、冻结配置和日志位于服务器实验目录。
新增几何／损失／空输出契约测试及全仓回归均通过，具体结果见实验目录tests-final.log；独立推理CLI仅传RGB、原始深度与标定文件，
不传液面真值或仿真材质标签，其输出与同一帧离线评估一致。代码测试通过不代表精度达标。
''',f'实验目录：`{root}`。报告由 scripts/report_liquid_height_v21.py 从原始JSON生成。']
text='\n\n'.join(sections)+'\n'
# CommonMark tables require adjacent rows, not paragraphs between rows.
text=re.sub(r'(?m)(\|[^\n]*\|)\n\n(?=\|)',r'\1\n',text)
Path('docs/liquid_height_reassessment_v21.md').write_text(text)
(root/'report.md').write_text(text)
print('REPORT_WRITTEN',root,flush=True)
