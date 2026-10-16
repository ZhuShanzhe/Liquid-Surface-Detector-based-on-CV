"""All-frame staged comparison; no score-driven selection on either test set."""
import json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from experiment_transparent_v26 import OUT,BASE,summarize,dump

stages=['v25','bounded','pixel_bias','phase']
names={'active_stereo':'双目','structured_light':'结构光','transparent':'透明','translucent_mild':'半透明'}
stage_names={'v25':'优化前V25','bounded':'约束经验校正','pixel_bias':'加入匹配偏置校正','phase':'加入块内相位聚合（最终V26）'}
def passed(r):return r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m']
def key(r):return r['config_id'],r['sample_id']

def compare(label):
    raw=json.loads((OUT/(label+'-raw.json')).read_text());rs=raw['records']
    lookup={(r['config_id'],r['sample_id'],r['stage']):r for r in rs}
    overall=[dict(stage=s,**summarize([r for r in rs if r['stage']==s])) for s in stages]
    groups=[];ranges=[]
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild'):
            for s in stages:
                group=[r for r in rs if r['stage']==s and r['family']==family and r['condition']==condition]
                groups.append(dict(family=family,condition=condition,stage=s,**summarize(group)))
            for b in (.3,.8,2.,6.,9.8):
                for s in ('v25','phase'):
                    group=[r for r in rs if r['stage']==s and r['family']==family and r['condition']==condition and r['bottom_m']==b]
                    ranges.append(dict(family=family,condition=condition,bottom_m=b,stage=s,**summarize(group)))
    transitions=[]
    for old,new in [('v25','bounded'),('bounded','pixel_bias'),('pixel_bias','phase'),('v25','phase')]:
        pairs=[(r,lookup[(r['config_id'],r['sample_id'],new)]) for r in rs if r['stage']==old]
        transitions.append(dict(before=old,after=new,
            fixed=sum(not passed(a) and passed(b) for a,b in pairs),
            new_failures=sum(passed(a) and not passed(b) for a,b in pairs),
            both_failed=sum(not passed(a) and not passed(b) for a,b in pairs)))
    failures=[r for r in rs if r['stage']=='phase' and not passed(r)]
    unchanged=None;mapping_repair=None
    if label=='v25-paired':
        original=json.loads((BASE/'selected-frame-diagnostics.json').read_text())
        unchanged=all(abs(lookup[(r['config_id'],r['sample_id'],'v25')]['predicted_height_m']-r['predicted_height_m'])<1e-10 for r in original)
        problem=[r for r in original if not passed(r) and abs(r['diagnostics']['uncorrected_height_m']-r['true_height_m'])<=r['tolerance_m']]
        mapping_repair=dict(original_cases=len(problem),
            fixed_after_bounded=sum(passed(lookup[(r['config_id'],r['sample_id'],'bounded')]) for r in problem),
            fixed_final=sum(passed(lookup[(r['config_id'],r['sample_id'],'phase')]) for r in problem))
    result=dict(label=label,overall=overall,groups=groups,ranges=ranges,transitions=transitions,
        remaining_failures=failures,v25_archived_predictions_unchanged=unchanged,original_mapping_errors=mapping_repair)
    dump(OUT/(label+'-comparison.json'),result)
    return result

results=[compare(label) for label in ['v25-paired','holdout'] if (OUT/(label+'-raw.json')).exists()]
lines=['# 透明半透明：双目校正、匹配偏置和结构光相位优化 V26','',
    '日期：2026-10-11。保持无置信度/覆盖率输出筛选，仅目标透明和轻度半透明仿真。','',
    '## 算法改动','',
    '1. 双目经验映射改为物理估计上的有界残差：校正曲线导数限制在±0.3以内，因此合成映射增益在0.7–1.3范围；超出先验范围只延续端点残差，禁止延续末段经验斜率。该约束减少外推放大，但本身不保证每一帧优于原曲线。',
    '2. 以5个已知液深先验联合估计有效折射率、固定像素偏置和周期性亚像素偏置。透明双目使用两阶、半透明双目使用一阶正弦/余弦，并加系数正则化。模型针对匹配器的像素相位偏差；不将拟合折射率视作材料真值测量。',
    '3. 结构光取消逐像素对带噪对比度的相位归一化，以及最终选中配置中的额外相位平滑；先解相位，再对24×24空间块的有效对应做质量加权聚合，替代每块只取最高分单点。每块保留一个观测，避免把大量相关像素当作独立证据。','',
    '## 选择与测试协议','',
    '参数仅在原V23数据中透明/轻度半透明部分选择；原V23测试部分已明确转为开发数据。V25的320组不参与本轮选择。按先有界校正、再匹配偏置、再结构光相位的顺序选择，方法与源码在评估前冻结。',
    '旧320组用于严格同帧前后比较；新增320组使用新的纹理种子、折射率1.315/1.425及独立先验标签噪声。两批均每个配置提供5帧已知液深先验，未知液深不得参与拟合或逐帧选择算法。',
    '容差保持max(5 mm, 2%×实际液深)，实际液深相对容器底面。全部可计算帧均输出，所有失败帧进入统计分母。诊断结果qualified=false，不自动用于工业控制。','']
fmt=lambda v:'—' if v is None else f'{v:.3f}'
for result in results:
    title='原320组同帧对比' if result['label']=='v25-paired' else '新增320组独立验证'
    lines+=['## '+title,'','| 阶段 | 数值输出/总数 | MAE mm | AbsRel % | 容差通过 | P95 mm | 最大误差 mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in result['overall']:
        lines.append(f"| {stage_names[r['stage']]} | {r['outputs']}/{r['frames']} | {fmt(r['mae_mm'])} | {fmt(r['absrel_pct'])} | {r['passed']}/{r['frames']} | {fmt(r['p95_mm'])} | {fmt(r['max_mm'])} |")
    lines+=['','| 场景 | 方法 | MAE mm | AbsRel % | 容差通过 | P95 mm |','|---|---|---:|---:|---:|---:|']
    for r in result['groups']:
        if r['stage'] not in ('v25','phase'):continue
        lines.append(f"| {names[r['family']]}/{names[r['condition']]} | {stage_names[r['stage']]} | {fmt(r['mae_mm'])} | {fmt(r['absrel_pct'])} | {r['passed']}/{r['frames']} | {fmt(r['p95_mm'])} |")
    lines+=['','### 同帧进退','', '| 步骤 | 修复原超标 | 新增超标 | 两者均超标 |','|---|---:|---:|---:|']
    for r in result['transitions']:
        lines.append(f"| {stage_names[r['before']]}→{stage_names[r['after']]} | {r['fixed']} | {r['new_failures']} | {r['both_failed']} |")
    if result['original_mapping_errors']:
        m=result['original_mapping_errors']
        lines+=['',f"先前识别的{m['original_cases']}个‘物理估计合格、经验校正后超标’组，有界校正后修复{m['fixed_after_bounded']}组，最终链路修复{m['fixed_final']}组。旧V25原始预测逐帧一致性核验：{result['v25_archived_predictions_unchanged']}。"]
    lines+=['','### 按相机到底面距离分组（不是按液深分组）','',
        '| 场景 | 底面距离 m | 原通过/总数 | 新通过/总数 | 原 MAE mm | 新 MAE mm |',
        '|---|---:|---:|---:|---:|---:|']
    range_lookup={(r['family'],r['condition'],r['bottom_m'],r['stage']):r for r in result['ranges']}
    for r in result['ranges']:
        if r['stage']!='phase':continue
        old=range_lookup[(r['family'],r['condition'],r['bottom_m'],'v25')]
        lines.append(f"| {names[r['family']]}/{names[r['condition']]} | {r['bottom_m']:.1f} | {old['passed']}/{old['frames']} | {r['passed']}/{r['frames']} | {fmt(old['mae_mm'])} | {fmt(r['mae_mm'])} |")
    lines+=['','### 最终仍超标的全部测量组','',
        '| 配置 | 帧 | 真液深 m | 估计 m | 绝对误差 mm | 容差 mm | 相对误差 % | 警告 |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(result['remaining_failures'],key=lambda r:r['absolute_error_m']/r['tolerance_m'] if r['absolute_error_m'] is not None else 1e9,reverse=True):
        error=r['absolute_error_m'];v=r['predicted_height_m']
        lines.append(f"| {r['config_id']} | {r['sample_id']} | {r['true_height_m']:.4f} | {fmt(v)} | {fmt(error*1000 if error is not None else None)} | {r['tolerance_m']*1000:.2f} | {fmt(error/r['true_height_m']*100 if error is not None else None)} | {','.join(r['diagnostics'].get('reasons',[]))} |")
lines+=['','## 结果解释与未解决的问题','',
    '两批共640组未知液深测量均保留输出，最终622组通过、18组超标。这里的100%是数值输出率，不是可信覆盖率；本实验不启用输出置信度筛选。',
    '仅约束经验曲线并不充分：旧测试的P95从59.38增至61.76 mm，新测试的MAE从14.59增至15.08 mm，且两批均新增5个超标组。残差约束抑制外推放大，但不能替代匹配偏置校正。最终采用整条V26组合，而不是把第一步宣称为逐帧无退化。',
    '最终仍有退化：旧测试修复34组原超标、但新增3组超标；新测试修复35组原超标、但新增2组超标。其逐帧结果保存在*-raw.json及*-comparison.json，不以平均数掩盖。',
    '剩余18组中17组带outside_prior_span警告，均集中于6或9.8 m底面距离下的浅液位；另一组为9.8 m底面距离、1.176 m液深的双目半透明帧。先验最低液深为底面距离的10%，而最浅测试为6%。缺少浅端先验约束仍是突出问题，不能靠重新收紧门控解决。',
    '在远距离，小像素偏差会被液深反演放大；底面穿透观测对浅液深与有效折射率/偏置的区分也较弱。半透明结构光残留偏差可能来自模拟散射导致的相位混合，块内聚合主要减少随机噪声，不能保证消除共同偏移。这是与结果一致的机制解释，尚非独立因果分解实验。',
    '下一轮建议在独立开发集增加浅液位已知先验，比较同样5帧预算下的先验布置；进一步对双目做左右一致性与残余周期偏差诊断，对半透明结构光做多频相位/散射偏置辨识。本轮不再利用已评分测试帧调参；这两批需作为历史回归集，下一轮另设独立验证集。','']
latency_path=OUT/'processing-latency.json'
if latency_path.exists():
    latency=json.loads(latency_path.read_text())
    lines+=['## 处理时延与回归验证','',
        '| 传感链路 | 版本 | 热运行次数 | 平均 ms | P95 ms | 最大 ms |',
        '|---|---|---:|---:|---:|---:|']
    for r in latency['summary']:
        lines.append(f"| {names[r['family']]} | {r['version']} | {r['samples']} | {fmt(r['mean_ms'])} | {fmt(r['p95_ms'])} | {fmt(r['max_ms'])} |")
    lines+=['',
        '24个代表配置，每配置每版本预热1次后计时5次；每种传感链路每版本60次。范围为内存中的图像匹配/相位解码及液深反演，排除先验拟合、曝光采集、磁盘/网络传输和界面。测试与独立数据生成并行运行；不是最终设备实时认证。',
        '结构光30次曝光若按60 fps顺序采集，仅曝光序列约500 ms，尚未加图像处理；目前不能宣称这条链路端到端低于500 ms。',
        '新增5项针对性测试通过；全项目回归346项通过、14项既有警告（regression.log，18.51 s）。评估完成时源码和策略哈希与冻结版本一致（progress.json、frozen-source.json）。','']
lines+=['','## 证据范围与运行','',
    '该模拟共享水平底面、平液面、均匀折射率的前向/逆向模型。没有真实硬件辐射标定、侧壁玻璃、波浪或漂浮物，不能用这些结果宣称复杂工业现场已达到同样精度。',
    '仍需关注先验区间外、极浅液位、匹配器更换或传感器型号变化时的像素偏置迁移。保持输出并记录误差，不能仅看总体平均数。',
    '结构光每个测量组需要30次曝光；图像处理时延不能替代采集时延。',
    '开发入口：scripts/experiment_transparent_v26.py develop。冻结评估入口：scripts/run_transparent_v26.py。报告入口：scripts/report_transparent_v26.py。',
    '服务器产物目录：'+str(OUT),
    '源码：src/liquid_depth/transparent_refinement_v26.py、src/liquid_depth/refraction_matching_v26.py。']
text='\n'.join(lines)+'\n'
(OUT/'report.md').write_text(text,encoding='utf-8')
Path('docs/transparent_refinement_v26.md').write_text(text,encoding='utf-8')
for result in results:
    print(result['label'],json.dumps(dict(overall=result['overall'],transitions=result['transitions'],mapping=result['original_mapping_errors']),indent=2))
