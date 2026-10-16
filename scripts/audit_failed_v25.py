"""All-finite research output and evaluator-only diagnosis of tolerance failures.

No confidence/coverage filter is applied. Ground truth only scores frozen
predictions or supplies explicitly labelled non-deployable counterfactuals.
"""
import hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
import cv2,numpy as np
from liquid_depth import transparent_diagnostic_v25 as model
from liquid_depth.prior_reliability_v24 import _map

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/transparent-diagnostic-v25-20261011')
OUT=BASE/'all-finite-failure-audit'
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-diagnostic-v25-20261011')
OUT.mkdir(exist_ok=True)
manifest=json.loads((DATA/'manifest.json').read_text())
raw=json.loads((BASE/'test-raw.json').read_text())
selected=json.loads((BASE/'selected-frame-diagnostics.json').read_text())
original_oracle=json.loads((BASE/'oracle-ablation.json').read_text())
policy=json.loads((BASE/'frozen-selection.json').read_text())
fits={(f['config'],f['method']):f for f in raw['fits']}
by_key={(r['config_id'],r['sample_id']):r for r in selected}
oracles={(r['config_id'],r['sample_id'],r['arm']):r for r in original_oracle['records']}
input_paths=[BASE/'test-raw.json',BASE/'selected-frame-diagnostics.json',BASE/'oracle-ablation.json',BASE/'frozen-selection.json',DATA/'manifest.json']
fingerprint={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def observation(c):
    with np.load(Path(c['folder'])/'matches_optimized.npz') as f:obs={k:f[k] for k in f.files}
    obs['regions']=len(obs['weights'])
    return obs

def oracle_observation(obs,truth,cal,family):
    ps=obs['left_pixels'];prefix='right' if family=='active_stereo' else 'projector'
    target=np.column_stack([cv2.remap(truth[prefix+'_'+axis],ps[:,0].astype(np.float32).reshape(-1,1),
        ps[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel() for axis in ['u','v']])
    return dict(obs,right_q=(target-np.array([cal['reference_cx'],cal['reference_cy']]))/cal['fx'])

def metrics(rows,field='absolute_error_m'):
    valid=[r for r in rows if r.get(field) is not None]
    e=np.array([r[field] for r in valid]);h=np.array([r['true_height_m'] for r in valid]);t=np.maximum(.005,.02*h)
    return dict(frames=len(rows),finite_outputs=len(valid),passed=int(np.sum(e<=t)),
        mae_mm=float(e.mean()*1000) if len(e) else None,median_mm=float(np.median(e)*1000) if len(e) else None,
        p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,max_mm=float(e.max()*1000) if len(e) else None,
        absrel_pct=float(np.mean(e/h)*100) if len(e) else None,pass_pct=float(np.mean(e<=t)*100) if len(e) else None)

frames=[];counterfactuals=[]
for c in manifest['configs']:
    model.require_target(c['condition'])
    cs=c['captures'];cal=json.loads((Path(cs[0]['folder'])/'calibration.json').read_text())
    measured=[observation(cap) for cap in cs];ideal=[];true_heights=[];true_indices=[]
    for cap,obs in zip(cs,measured):
        with np.load(cap['evaluator_truth']) as f:
            ideal.append(oracle_observation(obs,f,cal,c['family']))
            true_heights.append(float(f['height_m']));true_indices.append(float(f['index']))
    known=[cap['known_liquid_height_m'] for cap in cs[:5]]
    method=policy['methods'][c['family']+'/'+c['condition']]
    prior=fits[c['id'],method]['prior']
    assert prior is not None
    oracle_prior=model.fit_prior(ideal[:5],known,cal,c['condition'],prior['correction'])
    for j in range(5,len(cs)):
        key=(c['id'],j);r=dict(by_key[key]);pred=r['predicted_height_m'];h=r['true_height_m'];tol=r['tolerance_m']
        # Frozen estimate is re-evaluated, without acceptance filters or refitting.
        replay=model.infer(measured[j],prior,cal)
        assert abs(replay['height_m']-pred)<1e-10
        obs=measured[j];truth_obs=ideal[j]
        signed_pixels=(obs['right_q']-truth_obs['right_q'])*cal['fx']
        medx=float(np.median(signed_pixels[:,0]));medy=float(np.median(signed_pixels[:,1]))
        raw_h=r['diagnostics']['uncorrected_height_m'];raw_e=abs(raw_h-h)
        dh=max(1e-7,cal['bottom_m']*1e-6)
        derivative=(model.pixel_residual(raw_h+dh,prior['index'],prior['bias_px'],obs,cal)-
                    model.pixel_residual(raw_h-dh,prior['index'],prior['bias_px'],obs,cal))/(2*dh)
        jac=np.median(abs(derivative[:,0]));raw_m_per_px=float(1/max(jac,1e-12))
        dx=max(cal['bottom_m']*1e-6,1e-8)
        map_gain=1. if prior['correction']=='none' else (_map(raw_h+dx,np.array(prior['x']),np.array(prior['y']),prior['correction'])-
             _map(raw_h-dx,np.array(prior['x']),np.array(prior['y']),prior['correction']))/(2*dx)
        all_fields=dict(r,gate_mode='diagnostic_all_finite',displayed=bool(np.isfinite(pred)),
            qualified_for_control=False,passes_accuracy=bool(abs(pred-h)<=tol),
            signed_error_mm=float((pred-h)*1000),raw_error_m=raw_e,
            correction_delta_mm=float((pred-raw_h)*1000),raw_passes_accuracy=bool(raw_e<=tol),
            signed_match_x_px=medx,signed_match_y_px=medy,
            inverse_sensitivity_m_per_px=raw_m_per_px,calibration_mapping_gain=float(map_gain),
            output_sensitivity_m_per_px=float(raw_m_per_px*abs(map_gain)),
            equivalent_systematic_pixel_budget=float(tol/(raw_m_per_px*max(abs(map_gain),1e-12))),
            effective_index=prior['index'],true_index_evaluator_only=true_indices[j],bias_px=prior['bias_px'],
            prior_condition_number=prior['parameter_condition_number'],
            prior_min_label_m=float(min(known)),prior_max_label_m=float(max(known)))
        frames.append(all_fields)
        # Counterfactuals use evaluator truth. They are cause-isolation experiments.
        for arm,ob,p in [('oracle_prior_real_measurement',obs,oracle_prior),
                         ('real_prior_oracle_measurement',truth_obs,prior)]:
            pcf=model.infer(ob,p,cal);v=pcf['height_m']
            counterfactuals.append(dict(config_id=c['id'],sample_id=j,arm=arm,
                family=c['family'],condition=c['condition'],true_height_m=h,predicted_height_m=v,
                absolute_error_m=abs(v-h) if v is not None else None,evaluator_only=True,deployable=False))
        for arm in ['oracle_matches_noisy_labels','measured_matches_exact_labels','oracle_matches_exact_labels']:
            counterfactuals.append(oracles[key+(arm,)])
    print('AUDITED',c['id'],flush=True)

failed=[r for r in frames if not r['passes_accuracy']];good=[r for r in frames if r['passes_accuracy']]
assert len(frames)==320 and len(failed)==40
dump('all-finite-measurements.json',dict(gate_mode='diagnostic_all_finite',confidence_gate_enabled=False,
    coverage_gate_enabled=False,measurement_scope=['transparent','translucent_mild'],records=frames))
dump('failed-frame-details.json',failed)
dump('counterfactuals.json',dict(evaluator_only=True,deployable=False,records=counterfactuals))
groups=[]
for family in ('active_stereo','structured_light'):
    for condition in ('transparent','translucent_mild'):
        all_rows=[r for r in frames if r['family']==family and r['condition']==condition]
        rs=[r for r in failed if r['family']==family and r['condition']==condition]
        groups.append(dict(family=family,condition=condition,total_frames=len(all_rows),
            failures=len(rs),failure_fraction_pct=len(rs)/len(all_rows)*100,failed_metrics=metrics(rs)))
strata=[]
for b in (.3,.8,2.,6.,9.8):
    rs=[r for r in frames if r['bottom_m']==b];bad=[r for r in rs if not r['passes_accuracy']]
    strata.append(dict(camera_to_bottom_m=b,total=len(rs),failed=len(bad),failed_metrics=metrics(bad),
        median_output_sensitivity_m_per_px=float(np.median([r['output_sensitivity_m_per_px'] for r in rs]))))
ratios=[]
for ratio in (.06,.12,.18,.31,.46,.60,.70,.76):
    rs=[r for r in frames if abs(r['liquid_fraction']-ratio)<1e-6]
    ratios.append(dict(height_to_bottom_distance=ratio,total=len(rs),failed=sum(not r['passes_accuracy'] for r in rs)))
arms=[];bad_keys={(r['config_id'],r['sample_id']) for r in failed}
for arm in sorted({r['arm'] for r in counterfactuals}):
    rs=[r for r in counterfactuals if r['arm']==arm and (r['config_id'],r['sample_id']) in bad_keys]
    arms.append(dict(arm=arm,**metrics(rs)))
correction=dict(failed_but_raw_pass=sum(r['raw_passes_accuracy'] for r in failed),
    raw_fail_corrected_pass=sum(not r['raw_passes_accuracy'] for r in good),
    raw_metrics_on_failed=metrics(failed,'raw_error_m'),
    raw_all=metrics(frames,'raw_error_m'),corrected_all=metrics(frames))
overlap=[]
for extrap in (False,True):
    rs=[r for r in frames if ('outside_prior_span' in r['diagnostics']['reasons'])==extrap]
    bad=[r for r in rs if not r['passes_accuracy']]
    overlap.append(dict(outside_prior_span=extrap,total=len(rs),failed=len(bad),failed_metrics=metrics(bad)))
summary=dict(total=len(frames),finite_outputs=sum(r['displayed'] for r in frames),passed=len(good),failed=len(failed),
    all_metrics=metrics(frames),failed_metrics=metrics(failed),scene_groups=groups,distance_groups=strata,
    relative_height_groups=ratios,extrapolation_groups=overlap,correction=correction,
    oracle_arms_on_same_40_failed_frames=arms,
    match_regions_failed_min=min(r['regions'] for r in failed),
    failure_sign=dict(Counter('overestimate' if r['signed_error_mm']>0 else 'underestimate' for r in failed)),
    warning_counts=dict(Counter(x for r in failed for x in r['diagnostics']['reasons'])),
    sources_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in fingerprint.items()))
assert summary['sources_unchanged']
dump('summary.json',summary);dump('inputs-sha256.json',fingerprint)
names={'active_stereo':'双目','structured_light':'结构光','transparent':'透明','translucent_mild':'半透明'}
fmt=lambda v:'—' if v is None else f'{v:.3f}'
lines=['# 透明半透明：放宽门控后的实际超标帧分析','',
    '日期：2026-10-11。仅针对 V25 固定仿真测试；无强散射样本。','',
    '## 输出与判定','',
    '已生成显式诊断输出 all-finite-measurements.json：confidence_gate_enabled=false、coverage_gate_enabled=false。320/320 组有数值，280 组通过、40 组超过容差；所有估计均保留。精度标签只用于事后统计，不用于决定是否展示。',
    '液深相对底面。容差 max(5 mm, 2%×实际液深)。诊断输出保留警告、qualified_for_control=false。没有把取消门控解释为工业控制已认证。结构光一组测量含30次曝光，统计单位为测量组。','',
    '## 确实超标的40组','',
    '| 场景 | 超标/总数 | 超标组MAE mm | 中位误差 mm | P95 mm | 最大误差 mm | 超标组AbsRel % |',
    '|---|---:|---:|---:|---:|---:|---:|']
for g in groups:
    m=g['failed_metrics'];lines.append(f"| {names[g['family']]}/{names[g['condition']]} | {g['failures']}/{g['total_frames']} | {fmt(m['mae_mm'])} | {fmt(m['median_mm'])} | {fmt(m['p95_mm'])} | {fmt(m['max_mm'])} | {fmt(m['absrel_pct'])} |")
lines+=['','## 距离及液位分布','',
    '| 相机到底距离 m | 超标/总数 | 超标组MAE mm | 典型输出敏感度 m/px |',
    '|---|---:|---:|---:|']
for s in strata:lines.append(f"| {s['camera_to_bottom_m']} | {s['failed']}/{s['total']} | {fmt(s['failed_metrics']['mae_mm'])} | {fmt(s['median_output_sensitivity_m_per_px'])} |")
lines+=['','| 液深/相机到底距离 | 超标/总数 |','|---|---:|']
for s in ratios:lines.append(f"| {s['height_to_bottom_distance']:.0%} | {s['failed']}/{s['total']} |")
lines+=['','## 原因隔离：只比较相同40个超标组','',
    '以下对照在已固定算法之后进行。理想对应点和精确标签来自仿真真值，只用于定位误差来源，不能作为部署性能。单独替换一个阶段会破坏原本相互补偿的偏置，不能把不同对照改善量当作可相加的贡献百分比。','',
    '| 对照输入 | 仍超标 | MAE mm | AbsRel % |', '|---|---:|---:|---:|']
arm_names={'oracle_prior_real_measurement':'先验图匹配理想化，测量图保持原样',
    'real_prior_oracle_measurement':'先验图保持原样，测量图匹配理想化',
    'oracle_matches_noisy_labels':'先验/测量匹配均理想化，标签保留原噪声',
    'measured_matches_exact_labels':'原图匹配保留，仅先验液深标签精确化',
    'oracle_matches_exact_labels':'匹配/标签均理想化'}
for s in arms:lines.append(f"| {arm_names[s['arm']]} | {s['frames']-s['passed']}/{s['frames']} | {fmt(s['mae_mm'])} | {fmt(s['absrel_pct'])} |")
lines+=['','## 经验校正是否引入误差','',
    f"40 个超标组中，有 {correction['failed_but_raw_pass']} 组在经验映射前满足容差、映射后超标；反过来，全测试中有 {correction['raw_fail_corrected_pass']} 组在映射前超标、映射后合格。这是定位候选原因的事后证据，不能用测试真值决定实机该帧是否应用映射。",'',
    '这11组全部来自双目：相机到底2 m有6组、6 m有2组、9.8 m有3组。经验校正总体把通过数从255提升到280，但在部分浅液位和上端外推段改变了原本准确的结果。','',
    '实例一：双目半透明、相机到底2 m、实际液深0.12 m。物理解算原值误差2.13 mm，满足5 mm容差；经验映射后估计0.13921 m，误差19.21 mm，变为超标。',
    '实例二：双目半透明、相机到底9.8 m、实际液深7.448 m。原值误差1.64 mm；经验映射后误差192.79 mm。该帧位于先验上界之外，说明上端外推斜率存在明显问题。','',
    '## 原因归纳与证据','',
    '1. 点数不足不是本批主要失败原因：40个超标组仍至少有149个匹配区域，所有组均成功给出数值。匹配区域是折射后的底部/投影对应，并非直接观测到液面点。',
    '2. 图像匹配偏差同时污染标定和测量：仅精确化先验液深标签仍有39/40组超标；两个阶段匹配同时理想化、保留标签噪声后0/40超标。单独改善先验匹配或单独改善测量匹配，仍分别有29/40和30/40超标，部分指标反而变差，提示原链路存在偏差补偿与相互作用，需要一起校准。',
    '3. 距离放大系统像素偏差：模型计算的输出敏感度中位数在到底2/6/9.8 m时分别约0.51/5.20/13.78 m/px。这个局部导数说明一致的微小偏差在远端会放大；不是随机像素噪声可简单线性叠加的误差上界。',
    '4. 浅液位更容易超过相对精度要求：27/40个超标组位于液深/到底距离6%或12%的分组。此处同等毫米偏差对应更大相对误差，并且6%液位低于标定最低10%液位，兼有外推因素。',
    '5. 先验外推与经验映射存在独立修复空间：范围外120组中23组超标，范围内200组中17组超标。映射确实引入11个失败，但总体救回36组，应改进斜率约束、物理锚定及工作区间标定，不能一概移除。','',
    '## 完整超标组','',
    '| 场景配置 | 帧 | 实际液深 m | 测量液深 m | 有符号误差 mm | 绝对误差 mm | 容差 mm | 相对误差 % | 外推警告 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|']
for r in sorted(failed,key=lambda r:r['absolute_error_m']/r['tolerance_m'],reverse=True):
    lines.append(f"| {r['config_id']} | {r['sample_id']} | {r['true_height_m']:.4f} | {r['predicted_height_m']:.4f} | {r['signed_error_mm']:.2f} | {r['absolute_error_m']*1000:.2f} | {r['tolerance_m']*1000:.2f} | {r['absolute_error_m']/r['true_height_m']*100:.2f} | {'是' if 'outside_prior_span' in r['diagnostics']['reasons'] else '否'} |")
lines+=['','## 证据边界与后续方向','',
    '固定短基线、有限分辨率下，远端液深对亚像素偏差更敏感。输出敏感度由当前求解模型的数值导数计算，是局部诊断量，不是相机标称精度或严格误差上界。',
    '液深较小时同样毫米偏差对应更大的相对误差。先验外推和经验映射斜率可能进一步放大匹配及标签误差。当前模拟仅水平底面、平液面与均匀折射率，不能将超标归因于本批不存在的漂浮物、玻璃侧壁或波浪。',
    '优先排查亚像素匹配随视差的系统偏置、结构光相位偏差，以及少帧先验所学经验映射在工作区间边缘的斜率。不通过测试标签选取更有利的输出方法，也不以降低覆盖率隐藏超标帧。',
    '源码：scripts/audit_failed_v25.py。服务器产物：'+str(OUT),
    '本轮不重新训练、不修改冻结模型，也不改默认工业控制门控。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2,allow_nan=False))
