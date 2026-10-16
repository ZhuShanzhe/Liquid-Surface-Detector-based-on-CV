"""Read-only audit of frozen V25 measurements and the historical V24b gate."""
import json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')
OUT=ROOT/'transparent-diagnostic-v25-20261011'
raw=json.loads((OUT/'test-raw.json').read_text())
selected=json.loads((OUT/'selected-frame-diagnostics.json').read_text())
policy=json.loads((ROOT/'prior-reliability-v24b-20261011/frozen-calibration-policy.json').read_text())
lookup={(r['config_id'],r['sample_id'],r['method']):r for r in raw['records']}
fits={(r['config'],r['method']):r for r in raw['fits']}

def failure_reasons(pred,multiplier):
    reasons=list(pred.get('reasons',[]))
    h=pred.get('height_m');u=pred.get('uncertainty_m')
    if h is None:return reasons or ['no_numeric_estimate']
    if not pred.get('observable'):return reasons or ['not_observable']
    if multiplier is None:return ['no_qualified_multiplier']
    if u is None or not math.isfinite(u):return ['invalid_uncertainty']
    if u>multiplier*max(.005,.02*max(h,0)):reasons.append('uncertainty_above_gate')
    return reasons

def stats(rows,field='absolute_error_m'):
    good=[r for r in rows if r.get(field) is not None]
    es=np.array([r[field] for r in good]);hs=np.array([r['true_height_m'] for r in good]);ts=np.maximum(.005,.02*hs)
    return dict(frames=len(rows),finite=len(good),mae_mm=float(es.mean()*1000) if len(es) else None,
        median_mm=float(np.median(es)*1000) if len(es) else None,p95_mm=float(np.percentile(es,95)*1000) if len(es) else None,
        max_mm=float(es.max()*1000) if len(es) else None,absrel_pct=float(np.mean(es/hs)*100) if len(es) else None,
        passed=int(np.sum(es<=ts)),pass_pct=float(np.mean(es<=ts)*100) if len(es) else None)

frames=[]
for row in selected:
    family=row['family'];p=policy[family+'/routes']['5'];chosen=p['selected']
    key=(row['config_id'],row['sample_id'])
    primary=lookup[key+('v23_3',)]
    method=chosen['method'] if chosen['method'].startswith('v23') else 'v24_'+chosen['method']
    fallback=lookup[key+(method,)]
    pr=failure_reasons(primary['diagnostics'],p['primary_multiplier'])
    fr=failure_reasons(fallback['diagnostics'],chosen['multiplier'])
    covered=not pr or not fr
    u=fallback['diagnostics'].get('uncertainty_m');pred=fallback['predicted_height_m']
    gate=chosen['multiplier']*max(.005,.02*max(pred or 0,0))
    primary_fit=fits[(row['config_id'],'v23_3')]
    fallback_fit=fits[(row['config_id'],method)]
    frames.append(dict(row,old_covered=covered,primary_reasons=pr,fallback_reasons=fr,
        primary_prior_failure=primary_fit['failure'],fallback_prior_failure=fallback_fit['failure'],
        old_fallback_error_m=fallback['absolute_error_m'],old_fallback_height_m=pred,
        fallback_uncertainty_mm=u*1000 if u is not None else None,fallback_gate_mm=gate*1000,
        fallback_risk_ratio=u/gate if u is not None else None,
        primary_regions=primary['diagnostics'].get('regions'),fallback_parameters=fallback_fit['prior']))
uncovered=[r for r in frames if not r['old_covered']]
failed=[r for r in frames if not r['passed']]
groups=[]
for family in ('active_stereo','structured_light'):
    for condition in ('transparent','translucent_mild'):
        rs=[r for r in uncovered if r['family']==family and r['condition']==condition]
        groups.append(dict(family=family,condition=condition,new=stats(rs),old=stats(rs,'old_fallback_error_m'),
            primary_reasons=dict(Counter(x for r in rs for x in r['primary_reasons'])),
            fallback_reasons=dict(Counter(x for r in rs for x in r['fallback_reasons']))))
ranges=[]
for bottom in (.3,.8,2.,6.,9.8):
    rs=[r for r in uncovered if r['bottom_m']==bottom]
    ranges.append(dict(bottom_m=bottom,**stats(rs)))
bins=[]
for low,high in [(0,5),(5,10),(10,20),(20,50),(50,100),(100,float('inf'))]:
    count=sum(low<=r['absolute_error_m']*1000<high for r in uncovered)
    bins.append(dict(lower_mm=low,upper_mm=high if math.isfinite(high) else None,frames=count))
primary_failures=Counter(r['primary_prior_failure'] for r in uncovered if r['primary_prior_failure'])
fallback_reasons=Counter(x for r in uncovered for x in r['fallback_reasons'])
label_check=Counter()
for r in uncovered:
    e=r['absolute_error_m'];tol=r['tolerance_m'];u=(r['fallback_uncertainty_mm'] or 0)/1000
    label_check['v25_pass' if e<=tol else 'v25_fail']+=1
    if r['old_fallback_error_m'] is not None:
        label_check['old_raw_pass' if r['old_fallback_error_m']<=tol else 'old_raw_fail']+=1
report=dict(total=len(frames),old_covered=len(frames)-len(uncovered),uncovered=stats(uncovered),
    old_uncovered_estimates=stats(uncovered,'old_fallback_error_m'),uncovered_by_scene=groups,
    uncovered_by_distance=ranges,error_bins=bins,primary_prior_failure_frames=dict(primary_failures),
    primary_reasons=dict(Counter(x for r in uncovered for x in r['primary_reasons'])),
    fallback_reasons=dict(fallback_reasons),old_rejection_vs_actual=dict(label_check),
    v25_failed=stats(failed),v25_failed_old_uncovered=sum(not r['old_covered'] for r in failed),
    match_regions_min_uncovered=min(r['regions'] for r in uncovered),
    warnings=dict(Counter(x for r in uncovered for x in r['diagnostics'].get('reasons',[]))),
    frames=uncovered)
assert len(frames)==320 and len(uncovered)==91
assert sum(g['new']['frames'] for g in groups)==len(uncovered)
assert sum(b['frames'] for b in bins)==len(uncovered)
assert report['uncovered']['passed']+report['v25_failed_old_uncovered']==len(uncovered)
(OUT/'uncovered-frame-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
names={'active_stereo':'双目','structured_light':'结构光','transparent':'透明','translucent_mild':'半透明'}
fmt=lambda v:'—' if v is None else f'{v:.2f}'
lines=['# 透明半透明：旧门控未覆盖帧液深误差核查','',
    '日期：2026-10-11。基于 V25 独立仿真测试；没有重新训练或调整门控。','',
    f"共 {len(frames)} 个测量组，历史 V24b 路线接受 {len(frames)-len(uncovered)} 组，未覆盖 {len(uncovered)} 组。V25 的无筛选诊断全部给出数值；本报告用历史门控定义未覆盖集合，在严格相同帧上统计旧补充校正和 V25 诊断估计的实际误差。",'',
    '液深相对容器底面，精度通过容差为 max(5 mm, 2%×真实液深)。未覆盖不等于无法计算，也不等于精度未通过。','',
    '| 场景 | 未覆盖组 | 旧原始 MAE mm | V25 MAE mm | 中位误差 mm | P95 mm | 最大误差 mm | AbsRel % | V25通过组 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in groups:
    s=g['new'];o=g['old']
    lines.append(f"| {names[g['family']]}/{names[g['condition']]} | {s['frames']} | {fmt(o['mae_mm'])} | {fmt(s['mae_mm'])} | {fmt(s['median_mm'])} | {fmt(s['p95_mm'])} | {fmt(s['max_mm'])} | {fmt(s['absrel_pct'])} | {s['passed']}/{s['frames']} |")
lines+=['','## 汇总','', '```json',json.dumps({k:v for k,v in report.items() if k not in ('frames','uncovered_by_scene')},ensure_ascii=False,indent=2),'```','',
    '## 原因解释','',
    '历史路线先尝试 V23 物理参数标定，再尝试 V24b 补充校正；只有两个分支都不被接受才算未覆盖。主分支的标定失败与补充分支的风险拒绝是同一帧的不同阶段原因，不能将计数相加当作总帧数。',
    '主分支：49 组所在标定的留一液位检查失败；40 组所在标定的有效折射参数不稳定；另 2 组有主分支估计但不确定度超限。前两项按受影响测量帧统计，不是独立标定失败次数。折射率与像素偏置互相补偿可使参数不稳定；留一检查失败表示用剩余先验预测被留出液位时偏差过大。',
    '补充分支：91/91 组都有有限数值，且全部因为不确定度指标超过门槛被拒绝；没有匹配点不足、无数值或几何边界拒绝。历史补充门槛为 0.5 × max(5 mm, 2%×预测液深)，它是开发集选择的风险门槛。',
    '不确定度阈值使用预测液深下的容差；实际通过率使用真实液深下的容差。预测不确定度超过阈值只表示该风险策略不愿放行，不能证明实际误差一定超限。',
    '例：相机到底 6 m、真实液深 0.72 m 的双目半透明组，旧补充误差 9.84 mm，小于实际容差 14.4 mm；不确定度指标 11.51 mm 却大于门槛 7.10 mm，因此仍被拒绝。V25 在同组误差为 0.415 mm。',
    '未覆盖集合中，V25 有 37 组位于先验范围外，其中 21 组通过、16 组失败；范围内 54 组中 44 组通过、10 组失败。外推组失败比例更高，但相关统计不是单一因果证明。',
    'V25 全部 320 组中有 40 组超出容差：平均误差 53.33 mm，AbsRel 6.10%。其中 26 组来自旧未覆盖集合，另 14 组来自旧已覆盖集合；这是与门控覆盖不同的统计集合。',
    '远距离、较浅液位与先验范围外的测量更敏感于亚像素匹配偏置。匹配区域仍存在，但相关系统偏差无法通过简单增加点数完全消除。V25 的理想对应点原因隔离试验已显示：改善匹配对应比仅消除先验标签噪声作用更明显；该结论只在当前仿真假设内成立。','',
    '## 全部未覆盖帧','',
    '| 配置 | 帧 | 真液深 m | V25估计 m | V25误差 mm | 旧原始误差 mm | 容差 mm | V25通过 | 旧补充不确定度 mm | 旧门槛 mm |',
    '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|']
for r in sorted(uncovered,key=lambda r:r['absolute_error_m'],reverse=True):
    lines.append(f"| {r['config_id']} | {r['sample_id']} | {r['true_height_m']:.4f} | {r['predicted_height_m']:.4f} | {r['absolute_error_m']*1000:.2f} | {fmt(r['old_fallback_error_m']*1000 if r['old_fallback_error_m'] is not None else None)} | {r['tolerance_m']*1000:.2f} | {'是' if r['passed'] else '否'} | {fmt(r['fallback_uncertainty_mm'])} | {fmt(r['fallback_gate_mm'])} |")
lines+=['','完整双分支拒绝原因、先验拟合状态和逐帧诊断保存在 uncovered-frame-audit.json。']
(OUT/'uncovered-frame-audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
compact={k:v for k,v in report.items() if k!='frames'}
compact['worst']= [{k:r[k] for k in ['config_id','sample_id','true_height_m','predicted_height_m','absolute_error_m','old_fallback_error_m','primary_reasons','fallback_reasons','fallback_uncertainty_mm','fallback_gate_mm']} for r in sorted(uncovered,key=lambda r:r['absolute_error_m'],reverse=True)[:6]]
print(json.dumps(compact,indent=2,allow_nan=False))
