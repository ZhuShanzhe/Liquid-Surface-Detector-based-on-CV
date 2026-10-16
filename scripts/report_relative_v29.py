"""All-frame V29 report, including paired scene-cluster intervals and failures."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from experiment_relative_v29 import OUT,DATA,REPO,LAYOUTS,summary,dump

NAMES={'wet_uniform':'原5帧有液均匀基线','absolute_empty':'空容器固定偏置＋4帧绝对双目',
    'relative_empty':'空容器参考相对位移','relative_multi':'多先验联合相对位移',
    'relative_corrected':'联合相对位移＋有界残差','selected':'开发集冻结组合'}
CN={'transparent':'透明','translucent_mild':'轻度半透明'}

def name(stage):
    if '/' not in stage:return NAMES[stage]
    layout,method=stage.split('/');return ('空＋4均衡' if layout=='empty_four' else '空＋4宽跨度')+'／'+NAMES[method]

def ok(r):return r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m']
def key(r):return r['config_id'],r['sample_id']
def fmt(x):return '—' if x is None else f'{x:.3f}'

def compare(base,new):
    lookup={key(x):x for x in base};pairs=[(lookup[key(x)],x) for x in new];clusters={}
    # Transparent and mild images share texture seeds: cluster them together.
    for a,b in pairs:clusters.setdefault('_'.join(a['config_id'].split('_')[:2]),[]).append((a,b))
    vals=list(clusters.values());rng=np.random.default_rng(109313);maes=[];passes=[]
    for _ in range(2000):
        sample=[p for i in rng.integers(0,len(vals),len(vals)) for p in vals[i]]
        finite=[(a,b) for a,b in sample if a['absolute_error_m'] is not None and b['absolute_error_m'] is not None]
        if finite:maes.append(float(np.mean([(a['absolute_error_m']-b['absolute_error_m'])*1000 for a,b in finite])))
        passes.append(float(np.mean([int(ok(b))-int(ok(a)) for a,b in sample])*100))
    return dict(clusters=len(vals),fixed=sum(not ok(a) and ok(b) for a,b in pairs),
        new_failures=sum(ok(a) and not ok(b) for a,b in pairs),both_failed=sum(not ok(a) and not ok(b) for a,b in pairs),
        mae_reduction_ci95_mm=np.percentile(maes,[2.5,97.5]).tolist() if maes else None,
        pass_increase_ci95_pp=np.percentile(passes,[2.5,97.5]).tolist())

def main():
    raw=json.loads((OUT/'test-raw.json').read_text());rs=raw['records'];policy=json.loads((OUT/'frozen-policy.json').read_text())
    base=[r for r in rs if r['stage']=='wet_uniform']
    selected=[dict(r,stage='selected',source_stage=r['stage']) for r in rs if r['stage']==policy['selection'][r['condition']]]
    assert len(base)==len(selected)==864
    allrows=rs+selected;stages=sorted({r['stage'] for r in rs})+['selected'];metrics=[]
    for stage in stages:
        subset=[r for r in allrows if r['stage']==stage]
        metrics.append(dict(stage=stage,overall=summary(subset),far=summary([r for r in subset if r['bottom_m']>=6])))
    transition=compare(base,selected);fartransition=compare([r for r in base if r['bottom_m']>=6],[r for r in selected if r['bottom_m']>=6])
    groups=[]
    for condition in CN:
        for stage in ('wet_uniform','selected'):
            rows=[r for r in allrows if r['stage']==stage and r['condition']==condition]
            groups.append(dict(condition=condition,stage=stage,overall=summary(rows),far=summary([r for r in rows if r['bottom_m']>=6])))
    failures=[r for r in selected if not ok(r)];basemap={key(r):r for r in base}
    fitmap={}
    for config in raw['fits']:
        for f in config['fits']:fitmap[config['config_id'],f['layout']+'/'+f['method'] if f['method']!='wet_uniform' else 'wet_uniform']=f['prior']
    failed=[]
    for r in failures:
        failed.append(dict(r,prior=fitmap[r['config_id'],r['source_stage']],baseline=basemap[key(r)],
            new_failure_from_baseline=ok(basemap[key(r)])))
    dump(OUT/'failed-frames.json',failed)
    dump(OUT/'selected-records.json',selected)
    reasons=Counter(reason for r in failures for reason in r['diagnostics'].get('reasons',[]))
    ranges=[]
    for b in (2.4,6.7,9.7):
        for c in (.12,.30,.60):
            for stage in ('wet_uniform','selected'):
                a=summary([r for r in allrows if r['stage']==stage and r['bottom_m']==b and r['container_max_height_m']==c])
                ranges.append(dict(bottom_m=b,capacity_m=c,stage=stage,**a))
    target=[]
    for stage in ('wet_uniform','selected'):
        for far in (False,True):
            target.append(dict(stage=stage,far=far,**summary([r for r in allrows if r['stage']==stage and r['true_height_m']>=.1 and (not far or r['bottom_m']>=6)])))
    errors=Counter(f['prior']['error'] for c in raw['fits'] for f in c['fits'] if 'error' in f['prior'])
    strict=[]
    for lo,hi in ((0.,.1),(.1,1.),(0.,1.)):
        subset=[r for r in selected if lo<=r['true_height_m']<hi]
        relerr=np.array([r['absolute_error_m']/r['true_height_m'] for r in subset if r['absolute_error_m'] is not None])
        strict.append(dict(min_height_m=lo,max_height_m=hi,frames=len(subset),passed_1pct=int(np.sum(relerr<=.01)),
            relative_error_p95_pct=float(np.percentile(relerr,95)*100),relative_error_max_pct=float(np.max(relerr)*100)))
    dump(OUT/'strict-relative-diagnostic.json',strict)
    selected_fits=[fitmap[r['config_id'],r['source_stage']] for r in selected[::12]]
    # The selected record order is grouped by configuration, with twelve frames.
    assert len(selected_fits)==72
    bound_fits=sum('index' in p and (p['index']<=1.205 or p['index']>=1.595) for p in selected_fits)
    overall=summary(selected);before=summary(base);far=summary([r for r in selected if r['bottom_m']>=6]);bf=summary([r for r in base if r['bottom_m']>=6])
    result=dict(baseline=before,overall=overall,far=far,baseline_far=bf,metrics=metrics,groups=groups,ranges=ranges,
        target_range=target,strict_relative=strict,transition=transition,far_transition=fartransition,failure_reasons=dict(reasons),
        failures=len(failures),fit_errors=dict(errors),selected_effective_index_bound_configs=bound_fits,
        source_and_policy_unchanged=json.loads((OUT/'progress.json').read_text()).get('source_and_policy_unchanged'))
    dump(OUT/'comparison.json',result)
    stable=(transition['mae_reduction_ci95_mm'] is not None and transition['mae_reduction_ci95_mm'][0]>0 and transition['pass_increase_ci95_pp'][0]>0)
    dump(OUT/'promotion-decision.json',dict(research_improvement_supported=stable,production_enabled=False,
        reason='Independent synthetic evidence only; no finite-vessel, pose-change, real-device or camera-to-display qualification'))
    lines=['# 透明半透明：空容器＋4帧先验与相对位移校正 V29','',
        '日期：2026-10-11。H是相对容器底面的实际液深，C是可实现最大液深，B是相机到底面距离。','',
        '## 结论与比较','',
        f"独立测试{overall['frames']}组全部保留。原5帧有液均匀基线MAE {fmt(before['mae_mm'])} mm、通过{before['passed']}/{before['frames']} ({before['pass_pct']:.2f}%)；冻结选型组合MAE {fmt(overall['mae_mm'])} mm、通过{overall['passed']}/{overall['frames']} ({overall['pass_pct']:.2f}%)。",
        f"B≥6 m的{far['frames']}组，基线MAE {fmt(bf['mae_mm'])} mm、通过{bf['passed']}/{bf['frames']}；选型组合MAE {fmt(far['mae_mm'])} mm、通过{far['passed']}/{far['frames']} ({far['pass_pct']:.2f}%)。",
        f"数值输出{overall['outputs']}/{overall['frames']}，不是置信度合格覆盖率。改善的配对区间是否同时支持MAE和通过率提升：{stable}；不自动启用工业生产路线。",'',
        '通过标准为误差≤max(5 mm, 2%×真实H)。所有未知测量帧计入通过率分母，不按置信度、覆盖率或误差选择帧；无法求解记失败，MAE只统计有限数值。容量外/负预测保留并告警，不静默截断。','',
        '## 五帧预算和实现','',
        '| 布置 | H/C |','|---|---|']
    for k,v in LAYOUTS.items():lines.append(f"| {k} | {'、'.join(f'{x*100:g}%' for x in v)} |")
    lines+=['',
        '每种空容器方案只采集1组空容器双目图像和4组已知液深双目图像；一组是左右目一对图，不是多次曝光平均。空容器标签为已确认的零液深，非零先验标签噪声标准差为0.5 mm+0.1%×H。不同布局复用相同液深的图像与标签，测试未知帧完全相同。',
        '绝对双目空容器方案先用零液深图像约束固定/周期像素偏置，再以4个非零液深拟合有效折射率，沿用V26有界液深残差。',
        '相对路线在同一相机通道跨液位图跟踪底纹，左右通道分别处理、双向跟踪与回环检查，再联合拟合折射几何。空参考仅使用空到各液位；多参考使用5帧先验之间的10组图像对，测量时使用全部5个参考。没有增加先验采集预算。',
        '相对路线减去每对图的平移残差中位数作为干扰项，但这不是相机位姿校正；缩放/旋转/移动与液深变形仍可能混淆。经验版本用跨先验重建的误差拟合有界残差，先验图不作为自己的匹配参考；它不是留一帧完全独立校准评估。',
        '匹配点相关度/回环检查属于观测有效性检查，不是按输出液深或置信度筛选测量帧。无可用观测时不编造结果。所有相对结果仅适用于相机、容器、底纹稳定且底部可见的均匀透明/轻度半透明液体。','',
        '## 开发选型与新测试独立性','',
        '开发：B=3.2/8.4 m、C=18/45 cm、折射率1.33/1.405，每种几何2个纹理种子，透明/轻度半透明共32配置×10未知液位=320组，种子812371+317*i。',
        '测试：B=2.4/6.7/9.7 m、C=12/30/60 cm、折射率1.315/1.455，每种几何2个纹理种子，72配置×12未知液位=864组，种子923147+317*i。',
        '未知H/C为3.5/7.5/12/18/26/36/47/58/69/80/91.5/97.5%；不与任何本轮标定液位相同。测试H=4.2–585 mm，远距离子集576组。相机距离9.7 m不等于液深9.7 m。',
        '先在开发集按整体/远距离容差归一化误差、P90与缺测惩罚选型，再冻结源码和策略，之后才生成测试图与评估真值。候选包含原基线，允许新方法失败或不获选；不根据测试结果重新选型。','',
        '| 场景 | 冻结选择 |','|---|---|']
    for c,s in policy['selection'].items():lines.append(f'| {CN[c]} | {name(s)} |')
    lines+=['','## 全候选的全部帧结果','',
        '| 方法 | 数值输出 | MAE mm | AbsRel % | P95 mm | 全部通过 | 远距离 MAE mm | 远距离通过 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        a=m['overall'];b=m['far'];lines.append(f"| {name(m['stage'])} | {a['outputs']}/{a['frames']} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {fmt(a['p95_mm'])} | {a['passed']}/{a['frames']} | {fmt(b['mae_mm'])} | {b['passed']}/{b['frames']} |")
    lines+=['','这些候选使用同一批未知图像，不是各自不同的864组。候选表仅作消融，不能以测试最优者替换冻结选择。','',
        '主要收益来自新的观测和多参考联合，而非仅多一张空图：均衡空容器固定偏置版本仅500/864通过，空容器单参考相对版本695/864；多参考联合已864/864通过、MAE 0.801 mm，加有界残差后MAE进一步降为0.691 mm、通过数不变。此结果不支持仅替换5点布置就能解决问题。','',
        '## 分场景','', '| 场景 | 方法 | MAE mm | AbsRel % | 全部通过 | 远距离 MAE mm | 远距离通过 |','|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        a=g['overall'];b=g['far'];lines.append(f"| {CN[g['condition']]} | {name(g['stage'])} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {a['passed']}/{a['frames']} | {fmt(b['mae_mm'])} | {b['passed']}/{b['frames']} ({b['pass_pct']:.2f}%) |")
    lines+=['','## 距离×容器最大液深','', '| B m | C cm | 基线 MAE mm | 新方案 MAE mm | 基线通过 | 新方案通过 |','|---|---:|---:|---:|---:|---:|']
    for b in (2.4,6.7,9.7):
        for c in (.12,.30,.60):
            a=next(x for x in ranges if x['bottom_m']==b and x['capacity_m']==c and x['stage']=='wet_uniform')
            n=next(x for x in ranges if x['bottom_m']==b and x['capacity_m']==c and x['stage']=='selected')
            lines.append(f"| {b:g} | {c*100:g} | {fmt(a['mae_mm'])} | {fmt(n['mae_mm'])} | {a['passed']}/{a['frames']} | {n['passed']}/{n['frames']} |")
    lines+=['','## 实际液深分层（主结果没有删除低液位）','', '| H范围 mm | 方法 | MAE mm | AbsRel % | 通过 |','|---|---|---:|---:|---:|']
    for lo,hi in ((0.,.05),(.05,.1),(.1,.25),(.25,.60)):
        for stage in ('wet_uniform','selected'):
            a=summary([r for r in allrows if r['stage']==stage and lo<=r['true_height_m']<hi])
            lines.append(f"| [{lo*1000:g},{hi*1000:g}) | {name(stage)} | {fmt(a['mae_mm'])} | {fmt(a['absrel_pct'])} | {a['passed']}/{a['frames']} |")
    lines+=['','小液深的相对误差分母很小，需同时看毫米误差和5 mm容差通过率；不能用很小的总体MAE宣称所有帧达到1%。','',
        '### 原目标H≥10 cm的补充分组','', '| 方法 | 距离 | MAE mm | AbsRel % | 通过 |','|---|---|---:|---:|---:|']
    for x in target:lines.append(f"| {name(x['stage'])} | {'B≥6 m' if x['far'] else '全部B'} | {fmt(x['mae_mm'])} | {fmt(x['absrel_pct'])} | {x['passed']}/{x['frames']} |")
    lines+=['','### 严格逐帧1%诊断（不是本轮通过标准）','',
        '| 实际液深 | 误差≤1%帧数 | 相对误差P95 % | 最大相对误差 % |','|---|---:|---:|---:|']
    for x in strict:
        label='全部' if x['min_height_m']==0 and x['max_height_m']==1 else ('H<10 cm' if x['max_height_m']==.1 else 'H≥10 cm')
        lines.append(f"| {label} | {x['passed_1pct']}/{x['frames']} ({100*x['passed_1pct']/x['frames']:.2f}%) | {fmt(x['relative_error_p95_pct'])} | {fmt(x['relative_error_max_pct'])} |")
    lines+=['','全部帧达到max(5 mm,2%H)不等于全部帧达到1%。本批仍有243组严格1%超标，其中206组H<10 cm、37组H≥10 cm；这些帧没有被删除或拒绝。残余标签噪声与图像跟踪偏置的对照见下文，不能把平均AbsRel<1%当作逐帧保证。']
    lines+=['','## 配对改善与不确定性','',
        f"修复基线失败{transition['fixed']}组，新增失败{transition['new_failures']}组，双方都失败{transition['both_failed']}组。",
        f"全体按{transition['clusters']}个几何/纹理场景簇（合并同种子透明与半透明）、2000次配对bootstrap：MAE降低95%区间{transition['mae_reduction_ci95_mm']} mm，通过率增加区间{transition['pass_increase_ci95_pp']}个百分点。正值表示改善。",
        f"远距离配对区间：MAE降低{fartransition['mae_reduction_ci95_mm']} mm、通过率增加{fartransition['pass_increase_ci95_pp']}个百分点。区间不包含仿真到现实域差。",'',
        '## 剩余超标与诊断线索','',
        f"冻结组合共有{len(failures)}组超标或无法求解；警告计数（可重叠）{dict(reasons)}。全候选先验失败计数{dict(errors)}。没有警告不等于误差合格，警告本身也不是独立因果证明。",'',
        f'冻结组合有效折射率接近边界的配置数{bound_fits}/72。该参数吸收成像与匹配代理偏差，不等于真实折射率的独立测定。',
        '| 场景 | B m | 剩余失败数 |','|---|---:|---:|']
    counts=Counter((r['condition'],r['bottom_m']) for r in failures)
    for (c,b),n in sorted(counts.items()):lines.append(f'| {CN[c]} | {b:g} | {n} |')
    lines+=['','### 容差余量最小的20组（没有超标时仍列出达标难帧）','', '| 配置/帧 | B m | C cm | 真H mm | 预测H mm | 误差 mm | 容差 mm | 线索 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in sorted(selected,key=lambda r:r['absolute_error_m']/r['tolerance_m'] if r['absolute_error_m'] is not None else 1e12,reverse=True)[:20]:
        p=r['predicted_height_m'];e=r['absolute_error_m'];lines.append(f"| {r['config_id']}/{r['sample_id']} | {r['bottom_m']:g} | {r['container_max_height_m']*100:g} | {r['true_height_m']*1000:.2f} | {fmt(p*1000 if p is not None else None)} | {fmt(e*1000 if e is not None else None)} | {r['tolerance_m']*1000:.2f} | {','.join(r['diagnostics'].get('reasons',[]))} |")
    oracle_path=OUT/'oracle-summary.json'
    if oracle_path.exists():
        oracle=json.loads(oracle_path.read_text());orr=json.loads((OUT/'oracle-raw.json').read_text())['records']
        normal={key(r):r for r in selected}
        def strictok(r):return r['absolute_error_m'] is not None and r['absolute_error_m']<=.01*r['true_height_m']
        lines+=['','## 仅供评估的理想观测对照','',
            '| 反事实 | MAE mm | 容差通过 | 严格1%通过 | 修复原1%失败 | 新增1%失败 |','|---|---:|---:|---:|---:|---:|']
        for mode,m in oracle['modes'].items():
            a=m['overall'];records=[r for r in orr if r['stage']==mode]
            fixes=sum(strictok(r) and not strictok(normal[key(r)]) for r in records)
            regress=sum(not strictok(r) and strictok(normal[key(r)]) for r in records)
            lines.append(f"| {mode} | {fmt(a['mae_mm'])} | {a['passed']}/{a['frames']} | {sum(strictok(r) for r in records)}/{len(records)} | {fixes}/243 | {regress} |")
        lines+=['',f"正常观测反事实逐帧精确重放：{oracle['normal_replay_exact']}。这些对照使用评估真值，只用于归因，不进入正常求解、选型或通过率主结果。理想轨迹保留原采样位置/区域数，以真几何替换对应点，联合去除了跟踪噪声、插值偏差和光照纹理混合造成的偏差，不是只改善某一个匹配器。",
            'exact_labels仅去除4个非零先验标签噪声；ideal_tracks同时理想化先验与测量图之间的匹配而保留标签噪声；ideal_tracks_and_labels二者均理想化。全理想近零误差仅证明共享前向/逆向几何的内部一致性，不代表硬件可达到该精度。',
            '残余误差仍同时受先验标签和跟踪影响：去掉标签噪声后MAE 0.475 mm，联合理想匹配后为0.462 mm，二者都明显低于正常0.691 mm。单因素理想化也新增一些严格1%失败，说明误差抵消与经验标定耦合仍存在，不能将两个改善量简单相加当作因果贡献。',
            '243个严格1%失败中206个来自H<10 cm：例如H=4.2 mm时，约0.386 mm误差已经是9.2%，但远小于5 mm容差。H≥10 cm仍有37个严格1%失败，接下来可研究更精确的先验标签、低偏差相对匹配及残差校准；优先验证有限容器ROI与固定相机假设，避免仅在理想宽视野继续压低数字。']
    lines+=['','## 延时与常驻研究接口','',
        '新增FixedCameraRelativeMeter（src/liquid_depth/relative_runtime_v29.py）接收5组左右目先验图、5个已知液深、相机/底面几何与液体类别；启动时拟合并保留参考，measure只处理当前左右图。支持4线程或4常驻进程的独立匹配，算法核心及已冻结精度参数不变。并行封装晚于精度冻结，单独用开发帧核对与串行缓存结果逐帧精确一致，不将其计时试验当作新的精度测试。',
        '| 计时条件 | 帧数 | 平均处理 ms | P95 ms | 数值一致 |','|---|---:|---:|---:|---|']
    for filename,title in [('processing-smoke.json','单线程串行，有并发评估负载'),('processing-parallel-smoke.json','4线程，有并发评估负载'),('processing-processes-idle.json','4常驻进程，空闲时处理')]:
        p=OUT/filename
        if p.exists():
            m=json.loads(p.read_text());lines.append(f"| {title} | {len(m['frames'])} | {fmt(m['mean_ms'])} | {fmt(m['p95_ms'])} | {m.get('all_predictions_exactly_equal','未单独核对')} |")
    lines+=['','六帧开发样本只属于运行冒烟测试，不是稳定延时认证；包括重新跟踪当前图和液深求解，不包括相机曝光/传输、图像读盘和界面显示。不得用缓存命中或单独求解时间宣称500 ms全链路达标。不同负载下的串行/并行计时不能作为严格受控的加速比。','',
        '## 证据边界和部署要求','',
        '一阶近似下，短基线液深视差信号约随f×baseline×H/B²变化，而相对底纹径向位移约随f×q×H/B变化；宽视野离轴底纹可提供更大的液深信号。这解释了相对路线的改善方向，但信号大小依赖底纹离轴范围，不能直接外推到小ROI。',
        '相对位移利用更宽视野上的液位折射变形，不再只依赖短基线的左右微小视差；但本轮仍是平底、平液面、无限纹理视野、已知相机/底面几何。没有玻璃侧壁、有限容器ROI缩小、底部污染、动态液面或相机运动。真实小容器远距离可见底纹不足时效果可能明显下降。',
        '例如本轮fx=450 px、宽480 px，在B=9.7 m时整个底面视野宽约10.35 m；若真实底部仅宽1.2 m，则空底投影宽约56 px。当前宽视野成绩不能当作这种小容器的已验证成绩，下一轮需以有限底纹范围/容器尺寸为独立变量验证。',
        '原V23散射代理按H/B缩放消光/模糊，非物理量纲一致的真实材料模拟；仅涉及透明及底部可见的轻度半透明，不能宣称强散射适配。空容器图使用空气/透明成像噪声；非零半透明图使用原轻散射代理。',
        '五帧相对路线需要保存完整双目先验图和已知液深，不能只保存旧点云平面。相机、内参、容器位置、底纹或光照投影条件发生显著变化后需重新标定。当前路线需要明确的安装边界与移动检测，未自动接入工业面板。',
        '研究接口中known_heights单位为米，第一项必须为确认的空容器零液深，随后4项递增且低于C；并未将15/40/65/90%写死在算法里。示例C=12 cm时开发选中方案为0/18/48/78/108 mm。左右输入为已去畸变、共用整流内参的灰度/IR图，标定和测量必须使用相同成像设置。calibration至少提供fx、cx、cy、bottom_m、container_max_height_m；当前几何假设是相机顶视水平底面，不能用任意倾斜位姿直接套用。',
        '外参和底面几何仍需独立提供；五帧液深先验不等于同时学会了未知相机位姿。缓存匹配用于避免消融重复计算，不是实际测量的零成本；尚未证明摄像头到面板500 ms。生产默认、安全门控和qualified=false保持不变。本轮无真实硬件精度认证，也未验证10 m实际液深。','',
        '## 复现与归档','',
        '核心：src/liquid_depth/transparent_relative_v29.py。实验：scripts/experiment_relative_v29.py generate_dev → evaluate_dev → run_test。报告：scripts/report_relative_v29.py。',
        f'数据：{DATA}',f'评估：{OUT}',
        'development-candidates.json和frozen-policy.json记录开发选型；test-raw.json保存所有候选全部帧；selected-records.json为冻结组合；failed-frames.json保存失败与基线、拟合参数；frozen-source.json记录冻结哈希。',
        f"冻结一致性：{result['source_and_policy_unchanged']}。回归：{(OUT/'regression.log').read_text().splitlines()[-1]}。"]
    if (OUT/'regression-final.log').exists():lines.append('常驻封装加入后的完整回归：'+(OUT/'regression-final.log').read_text().splitlines()[-1])
    text='\n'.join(lines)+'\n';(OUT/'report.md').write_text(text,encoding='utf-8');(REPO/'docs/transparent_relative_v29.md').write_text(text,encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
