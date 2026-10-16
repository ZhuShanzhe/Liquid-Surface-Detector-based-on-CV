"""Report all frozen-test predictions, including missing, failed and regressed frames."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from experiment_transparent_v27 import OUT,DATA,LAYOUTS,summary,dump

NAMES={'active_stereo':'双目','structured_light':'结构光','transparent':'透明','translucent_mild':'半透明'}
LAYOUT_NAMES={'original':'原方案','near_balanced':'浅端加密兼顾跨度','shallow_focus':'浅液位重点','uniform_span':'近均匀跨度'}
def fmt(v):return '—' if v is None else f'{v:.3f}'
def passed(r):return r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m']
def far(r):return r['bottom_m']>=6 and r['true_height_m']/r['bottom_m']<.18
def key(r):return r['config_id'],r['sample_id']

def paired_cluster_bootstrap(pairs):
    groups={}
    for a,b in pairs:groups.setdefault(a['config_id'],[]).append((a,b))
    clusters=list(groups.values());rng=np.random.default_rng(48123);mae=[];passes=[]
    for _ in range(2000):
        sample=[p for i in rng.integers(0,len(clusters),len(clusters)) for p in clusters[i]]
        finite=[(a,b) for a,b in sample if a['absolute_error_m'] is not None and b['absolute_error_m'] is not None]
        if finite:mae.append(np.mean([(a['absolute_error_m']-b['absolute_error_m'])*1000 for a,b in finite]))
        passes.append(100*np.mean([int(passed(b))-int(passed(a)) for a,b in sample]))
    return dict(clusters=len(clusters),resamples=2000,mae_improvement_mm_ci95=np.percentile(mae,[2.5,97.5]).tolist(),
        pass_improvement_percentage_points_ci95=np.percentile(passes,[2.5,97.5]).tolist())

def main():
    policy=json.loads((OUT/'frozen-policy.json').read_text())
    records=json.loads((OUT/'test-raw.json').read_text())['records']
    records+= [dict(r,stage='v26_selected') for r in records if r['stage']=='v26_'+policy['layouts'][r['family']+'/'+r['condition']]]
    stages=['v26_original','v26_selected','v27_final']
    stage_names={'v26_original':'优化前：V26+原5帧布置','v26_selected':'仅优化5帧布置','v27_final':'布置+残余偏置（V27）'}
    lookup={(r['config_id'],r['sample_id'],r['stage']):r for r in records}
    transitions=[]
    for a,b in [('v26_original','v26_selected'),('v26_selected','v27_final'),('v26_original','v27_final')]:
        pairs=[(r,lookup[(r['config_id'],r['sample_id'],b)]) for r in records if r['stage']==a]
        transitions.append(dict(before=a,after=b,fixed=sum(not passed(x) and passed(y) for x,y in pairs),
            new_failures=sum(passed(x) and not passed(y) for x,y in pairs),both_failed=sum(not passed(x) and not passed(y) for x,y in pairs),
            bootstrap=paired_cluster_bootstrap(pairs),
            absolute_error_better=sum(x['absolute_error_m'] is not None and y['absolute_error_m'] is not None and y['absolute_error_m']<x['absolute_error_m'] for x,y in pairs)))
    overall=[dict(stage=s,overall=summary([r for r in records if r['stage']==s]),
        far_shallow=summary([r for r in records if r['stage']==s and far(r)])) for s in stages]
    groups=[];ranges=[]
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild'):
            for s in stages:
                rs=[r for r in records if r['stage']==s and r['family']==family and r['condition']==condition]
                groups.append(dict(family=family,condition=condition,stage=s,overall=summary(rs),far_shallow=summary([r for r in rs if far(r)])))
            for bottom in sorted(set(r['bottom_m'] for r in records)):
                for s in stages:
                    rs=[r for r in records if r['stage']==s and r['family']==family and r['condition']==condition and r['bottom_m']==bottom]
                    ranges.append(dict(family=family,condition=condition,bottom_m=bottom,stage=s,**summary(rs)))
    failures=[r for r in records if r['stage']=='v27_final' and not passed(r)]
    new_failures=[r for r in failures if passed(lookup[(r['config_id'],r['sample_id'],'v26_original')])]
    result=dict(overall=overall,groups=groups,ranges=ranges,transitions=transitions,remaining_failures=failures,new_failures=new_failures)
    dump(OUT/'comparison.json',result)
    lines=['# 透明半透明：远距离浅液位5帧先验布置与残余偏置优化 V27','',
        '日期：2026-10-11。测量量为相对容器底面的实际液深 H；底面距相机 B 与液深 H 严格区分。',
        '单帧容差 max(5 mm, 2%×真实 H)。保持无输出置信度/覆盖率筛选；无法求解也计入全帧分母。数值输出不等于可信工业输出。','',
        '## 本轮结论','',
        '独立测试400组全部保留数值输出。V26原布置MAE 5.769 mm、通过370/400；仅调整5帧布置后MAE 3.846 mm、通过381/400；加残余偏置模型后MAE 3.845 mm、通过383/400。主要收益来自标定液深布置，而非增加模型参数。',
        '远距离浅液位80组：原方案MAE 13.333 mm、通过51/80；仅布置优化6.538 mm、62/80；最终6.570 mm、63/80。最终该子集AbsRel为2.099%，仍不是逐帧1%精度保证。',
        '残余偏置模型相对仅优化布置，MAE降低的配置级bootstrap 95%区间为[-0.161,0.171] mm，跨零；双目透明MAE反而从3.692增至3.993 mm。因此偏置模型仅保留为诊断实验选项，不能据此推荐全面替换。透明结构光继续保留V26。',
        '最终17组超标全部位于远距离浅液位子集，8组在先验范围外、9组已经在范围内，说明覆盖标定区间并不能消除剩余误差；原方案合格而最终超标的1组也完整列出。其余320组均通过本轮容差，但仅代表这些受控仿真配置。','',
        '## 实验设计与冻结协议','',
        '每种布置仅使用5组已知液深先验。四种布置的候选液深先声明；同一配置、同一液深共享图像及带噪已知标签，未知液深测量图像对所有方案完全一致。',
        '开发集32配置、每配置10个未知液深测量组，共320组；底面距离0.4/2.5/7.5/9.5 m，折射率1.33/1.40，纹理种子371209+241*i+10000*family。所有选择仅在此开发集完成。',
        '独立测试40配置、每配置10个未知液深组，共400组；底面距离0.3/0.9/2.2/6.5/9.8 m，折射率1.305/1.455，纹理种子478921+241*i+10000*family。未知H/B为1.5/4/6.5/9.5/14.5/24/41/57/69/81%。',
        '独立测试真实H范围4.5 mm–7.938 m；远距离浅液位子集H约97.5 mm–1.421 m。本轮没有验证10 m实际液深，也不能把9.8 m底面距离称为9.8 m液深。极浅样本保留在全部帧统计中。',
        '先冻结方法、策略与源码，再生成独立测试图像及标签。测试只评分，不能重新选择最优方案。V26原源码及原策略未改动；本轮不改产品默认链路。',
        '远距离浅液位子集预定义为B≥6 m且H/B<18%，独立测试80组。选择目标为整体与该子集容差归一化误差各占0.5，加整体P90项0.2及缺测惩罚；不设覆盖率门槛。',
        '已知先验标签噪声标准差0.5 mm+0.1%×H，与前轮一致。相机/底面几何已知且无额外标定误差；相机成像仍是非设备标定的仿真代理。','',
        '## 5帧标定液深布置','',
        '| 方案 | 5个 H/B |','|---|---|']
    for name,values in LAYOUTS.items():lines.append(f"| {LAYOUT_NAMES[name]} | {'、'.join(f'{100*v:g}%' for v in values)} |")
    lines+=['',
        '百分比只是本次仿真实验相对相机到底面距离B的采样设计，不是容器容量百分比，也不作为软件强制选点限制。实际操作应在容器可实现的液深范围内覆盖预计最低、常用和最高液位；不能要求操作员填充到容器之外。','',
        '重要部署限制：本轮各方案仍包含较高液深先验，例如双目选中布置在B=9.8 m时最高先验为7.448 m。若实际是相机远离浅容器的场景，该先验不可获取，不能直接援引本轮性能作为保证；后续需单独比较受容器最大液深限制的5帧布置，并重新独立验证。','',
        '| 场景 | 开发集选择的布置 | 残余偏置选择 |','|---|---|---|']
    for k,layout in policy['layouts'].items():
        f,c=k.split('/');lines.append(f"| {NAMES[f]}/{NAMES[c]} | {LAYOUT_NAMES[layout]} | {policy['bias'][k]['name']} |")
    lines+=['',
        '残余偏置模型在原有效折射率与固定像素偏置之外，比较更高阶亚像素周期项、固定相机坐标的低阶空间场(qx,qy,r²)以及结构光16像素周期相位项。系数均正则化并限制幅度，只由5组已知液深拟合；未知帧使用固定参数。',
        '双目采用开发集选中的空间场配置；透明结构光保留V26，半透明结构光选择空间场。新增周期相位候选未获稳定开发收益，不强行启用。空间场为经验误差项，不是已经辨识出真实散射物理参数。','',
        '## 开发集：各布置的比较','',
        '| 场景 | 布置 | 全部 MAE mm | 全部通过 | 远浅 MAE mm | 远浅通过 |','|---|---|---:|---:|---:|---:|']
    for group in json.loads((OUT/'layout-candidates.json').read_text()):
        for c in group['candidates']:
            a=c['overall'];b=c['far_shallow']
            lines.append(f"| {NAMES[group['family']]}/{NAMES[group['condition']]} | {LAYOUT_NAMES[c['layout']]} | {fmt(a['mae_mm'])} | {a['passed']}/{a['frames']} | {fmt(b['mae_mm'])} | {b['passed']}/{b['frames']} |")
    lines+=['','## 独立测试：所有帧的分步对比','',
        '| 阶段 | 数值输出/全部 | MAE mm | AbsRel % | P95 mm | 容差通过 |','|---|---:|---:|---:|---:|---:|']
    for m in overall:
        a=m['overall'];lines.append(f"| {stage_names[m['stage']]} | {a['outputs']}/{a['frames']} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {fmt(a['p95_mm'])} | {a['passed']}/{a['frames']} ({a['pass_pct']:.2f}%) |")
    lines+=['','### 远距离浅液位子集（不删除其余帧）','',
        '| 阶段 | MAE mm | AbsRel % | P95 mm | 容差通过 |','|---|---:|---:|---:|---:|']
    for m in overall:
        a=m['far_shallow'];lines.append(f"| {stage_names[m['stage']]} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {fmt(a['p95_mm'])} | {a['passed']}/{a['frames']} ({a['pass_pct']:.2f}%) |")
    lines+=['','### 独立测试的全部四种布置（V26算法不变）','',
        '| 布置 | 全部 MAE mm | 全部通过 | 远浅 MAE mm | 远浅通过 |','|---|---:|---:|---:|---:|']
    for layout in LAYOUTS:
        rs=[r for r in records if r['stage']=='v26_'+layout];a=summary(rs);b=summary([r for r in rs if far(r)])
        lines.append(f"| {LAYOUT_NAMES[layout]} | {fmt(a['mae_mm'])} | {a['passed']}/{a['frames']} | {fmt(b['mae_mm'])} | {b['passed']}/{b['frames']} |")
    lines+=['','测试布置表仅用于检验开发选择的泛化，不能据此换成测试集上最优的布置。','',
        '### 分场景全部测量组','',
        '| 场景 | 阶段 | MAE mm | AbsRel % | 全部通过 | 远浅通过 |','|---|---|---:|---:|---:|---:|']
    for g in groups:
        a=g['overall'];b=g['far_shallow'];lines.append(f"| {NAMES[g['family']]}/{NAMES[g['condition']]} | {stage_names[g['stage']]} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {a['passed']}/{a['frames']} | {b['passed']}/{b['frames']} |")
    lines+=['','### 按相机到底面距离汇总（全部液位）','',
        '| B m | 原 MAE mm | 仅布置 MAE mm | 最终 MAE mm | 原通过 | 最终通过 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for bottom in sorted(set(r['bottom_m'] for r in records)):
        ms=[summary([r for r in records if r['stage']==s and r['bottom_m']==bottom]) for s in stages]
        lines.append(f"| {bottom:.1f} | {fmt(ms[0]['mae_mm'])} | {fmt(ms[1]['mae_mm'])} | {fmt(ms[2]['mae_mm'])} | {ms[0]['passed']}/{ms[0]['frames']} | {ms[2]['passed']}/{ms[2]['frames']} |")
    lines+=['','### 开发集残余偏置候选（先验布置固定）','',
        '| 场景 | 偏置候选 | 选择得分 | 全部 MAE mm | 全部通过 | 远浅 MAE mm | 是否选中 |',
        '|---|---|---:|---:|---:|---:|---|']
    for group in json.loads((OUT/'bias-candidates.json').read_text()):
        for c in group['candidates']:
            a=c['overall'];b=c['far_shallow'];name=c['setting']['name']
            lines.append(f"| {NAMES[group['family']]}/{NAMES[group['condition']]} | {name} | {c['score']:.4f} | {fmt(a['mae_mm'])} | {a['passed']}/{a['frames']} | {fmt(b['mae_mm'])} | {'是' if name==group['selected']['setting']['name'] else '否'} |")
    lines+=['','### 同帧改善与退化','',
        '| 步骤 | 修复原超标 | 新增超标 | 两者均超标 | 绝对误差变小帧数 |','|---|---:|---:|---:|---:|']
    for m in transitions:lines.append(f"| {stage_names[m['before']]}→{stage_names[m['after']]} | {m['fixed']} | {m['new_failures']} | {m['both_failed']} | {m['absolute_error_better']} |")
    lines+=['','### 成组重采样的不确定性','',
        '| 步骤 | MAE降低的95%区间 mm | 通过率增加的95%区间 百分点 |','|---|---:|---:|']
    for m in transitions:
        b=m['bootstrap'];lo,hi=b['mae_improvement_mm_ci95'];pl,ph=b['pass_improvement_percentage_points_ci95']
        lines.append(f"| {stage_names[m['before']]}→{stage_names[m['after']]} | [{lo:.3f}, {hi:.3f}] | [{pl:.2f}, {ph:.2f}] |")
    lines+=['',
        '以配置为簇进行2000次同帧配对bootstrap，避免将同一容器/纹理的多个液位当作完全独立样本。正值表示改善；区间跨零说明当前样本不足以支持稳定改善。仅描述这批模拟配置的抽样不确定性，不覆盖仿真到现实的域差。']
    lines+=['','## 最终全部超标组','',
        '| 配置/帧 | 底面距离 m | 真液深 m | 估计 m | 误差 mm | 容差 mm | 较原方案新增超标 | 警告 |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for r in sorted(failures,key=lambda r:r['absolute_error_m']/r['tolerance_m'] if r['absolute_error_m'] is not None else 1e9,reverse=True):
        old=lookup[(r['config_id'],r['sample_id'],'v26_original')]
        lines.append(f"| {r['config_id']}/{r['sample_id']} | {r['bottom_m']:.1f} | {r['true_height_m']:.4f} | {fmt(r['predicted_height_m'])} | {fmt(r['absolute_error_m']*1000 if r['absolute_error_m'] is not None else None)} | {r['tolerance_m']*1000:.2f} | {'是' if passed(old) else '否'} | {','.join(r['diagnostics'].get('reasons',[]))} |")
    counts=Counter('outside_prior_span' in r['diagnostics'].get('reasons',[]) for r in failures)
    lines+=['',f"最终超标{len(failures)}组，其中{len(new_failures)}组较原方案新增，{counts[True]}组带先验范围外警告。范围警告是诊断，不参与筛除；不能仅凭警告认定唯一因果。",'',
        '## 解释与限制','',
        '浅液位先验直接约束浅端残差，低液位密集采样与高液位跨度共同影响有效折射率和固定偏置的可辨识性；并非越集中于最低液位越好。5组先验预算仍不增加。',
        '远距离有限基线下像素偏差的液深放大仍然存在。空间项能吸收一部分可重复偏差，但不能消除随机匹配噪声或凭空增加几何信息。浅端范围外、先验标签噪声以及半透明相位共同偏移仍需逐帧检查。',
        '新增周期相位候选没有胜出，说明本轮证据不支持用更多周期参数解决半透明残差。不要将空间场的小幅指标改善解释成物理散射已解决。',
        '仅覆盖顶视、平底/平液面、单一均匀液体、底纹/投影可匹配、几何已知的仿真。没有玻璃侧壁、多层液体、动态波浪、强散射或设备实测辐射标定；前向/逆向共享Snell几何。',
        '结构光每个测量组为30张曝光，双目每组为一对图；图像数量与已知液深先验组数不同。未因此证明端到端500 ms或工业实测精度。','',
        '## 复现与归档','',
        '开发：scripts/experiment_transparent_v27.py generate_dev → select_layout；然后 scripts/select_transparent_v27.py。冻结测试：scripts/run_transparent_v27.py；报告：scripts/report_transparent_v27.py。',
        '测试原始数据：'+str(DATA/'test/manifest.json'),
        '全部逐帧预测与标定参数：'+str(OUT/'test-raw.json'),
        '完整对比与分距离指标：'+str(OUT/'comparison.json'),
        '源码/策略哈希：'+str(OUT/'frozen-source.json'),
        '回归测试：'+str(OUT/'regression.log')]
    status=json.loads((OUT/'progress.json').read_text());log=(OUT/'regression.log').read_text()
    lines+=['',f"冻结一致性：{status.get('source_and_policy_unchanged')}。回归摘要：{log.strip().splitlines()[-1]}。"]
    text='\n'.join(lines)+'\n'
    (OUT/'report.md').write_text(text,encoding='utf-8')
    Path('docs/transparent_shallow_v27.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(overall=overall,transitions=transitions,failures=len(failures),new_failures=len(new_failures)),indent=2))

if __name__=='__main__':main()
