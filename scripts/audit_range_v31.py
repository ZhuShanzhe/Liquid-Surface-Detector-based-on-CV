"""Post-freeze evaluation/reporting only; does not choose an inference policy."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
import cv2
from experiment_range_v31 import OUT,DATA,REPO,summary,intervals,load,dump
from liquid_depth import finite_relative_v30 as base
from liquid_depth.range_relative_v31 import wide_pair,fit_curve,infer_curve
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom


def selected(rows,policy):
    return [r for r in rows if r['stage']==policy[r['condition']]]


def metrics(rows):
    s=summary(rows);finite=[r for r in rows if r['absolute_error_m'] is not None]
    s['rmse_mm']=float(np.sqrt(np.mean([r['absolute_error_m']**2 for r in finite]))*1000) if finite else None
    s['strict_1pct_passed_positive']=sum(r['absolute_error_m']<=.01*r['true_height_m'] for r in finite if r['true_height_m']>0)
    s['positive_total_frames']=sum(r['true_height_m']>0 for r in rows)
    return s


def bootstrap(before,after):
    # Correlated heights/materials from one texture count as one cluster.
    clusters=sorted({r['texture_cluster'] for r in after});rng=np.random.default_rng(5551012)
    pairs=[]
    for c in clusters:
        b=[r for r in before if r['texture_cluster']==c];a=[r for r in after if r['texture_cluster']==c]
        pairs.append((np.mean([r['absolute_error_m'] for r in b if r['absolute_error_m'] is not None]),
                      np.mean([r['absolute_error_m'] for r in a if r['absolute_error_m'] is not None]),
                      metrics(b)['pass_pct'],metrics(a)['pass_pct']))
    p=np.array(pairs);idx=rng.integers(0,len(p),(3000,len(p)));rep=p[idx].mean(axis=1)
    return dict(clusters=len(p),mae_improvement_mm_ci95=(np.percentile(rep[:,0]-rep[:,1],[2.5,97.5])*1000).tolist(),
        pass_improvement_percentage_points_ci95=np.percentile(rep[:,3]-rep[:,2],[2.5,97.5]).tolist())


def timing():
    cv2.setNumThreads(1);cs=json.loads((DATA/'test/manifest.json').read_text())['configs'];result=[]
    chosen=[next(c for c in cs if c['condition']==condition and c['material_regime']!='legacy_depth_attenuation' and c['empty_width_px']==width)
            for condition in ('transparent','translucent_mild') for width in (128,160)]
    for c in chosen:
        condition=c['condition']
        ims,masks,cal,hs=load(c);ps=[];start=time.perf_counter()
        for a in range(5):
            for b in range(a+1,5):
                for ch in ('left','right'):ps.append((a,b,wide_pair(ims[a,ch],ims[b,ch],cal,masks[ch],ch)))
        prior=fit_curve(ps,hs,cal,condition);fitting=time.perf_counter()-start;times=[];differences=[]
        expected=json.loads((OUT/'test-configs'/(c['id']+'.json')).read_text())['records']
        expected={r['sample_id']:r['predicted_height_m'] for r in expected if r['stage']=='wide_curve'}
        for j in (0,8,24,40,56,72):
            start=time.perf_counter();gs=[]
            for a in range(5):
                for ch in ('left','right'):gs.append((a,wide_pair(ims[a,ch],ims[j+5,ch],cal,masks[ch],ch)))
            pred=infer_curve(gs,prior,cal)['height_m'];times.append((time.perf_counter()-start)*1000)
            differences.append(abs(pred-expected[j]))
        assert max(differences)<1e-8,'uncached_parity_failed'
        result.append(dict(condition=condition,samples=len(times),empty_width_px=c['empty_width_px'],prior_fit_s=fitting,
            p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=max(times),
            max_cached_difference_m=max(differences),
            acquisition_transfer_display_included=False,cpu_threads=1))
    return result


def oracle_audit():
    """Evaluator-only counterfactual; hidden correspondences never reach meter."""
    cs=json.loads((DATA/'test/manifest.json').read_text())['configs'];records=[]
    for c in cs:
        if c['material_regime']=='legacy_depth_attenuation':continue
        caps=c['priors']+c['tests'];truth=[]
        for cap in caps:
            with np.load(cap['evaluator_truth']) as f:truth.append((float(f['height_m']),float(f['index'])))
        cal=json.loads((Path(caps[0]['folder'])/'calibration.json').read_text())
        y,x=np.meshgrid(np.linspace(-.2,.2,5)*c['empty_width_px']+cal['cy'],
                         np.linspace(-.25,.25,6)*c['empty_width_px']+cal['cx'])
        pp=np.c_[x.ravel(),y.ravel()];qq=(pp-[cal['cx'],cal['cy']])/cal['fx']
        def correspondence(a,b,ch):
            centre=(0.,0.) if ch=='left' else (cal['baseline_m'],0.)
            world=bottom_map(qq,truth[a][0],truth[a][1],cal['bottom_m'],centre)
            current=inverse_bottom(world,truth[b][0],truth[b][1],cal['bottom_m'],centre)
            return dict(anchor_q=qq,current_q=current,weights=np.ones(len(qq)),regions=len(qq),channel=ch,correlation=1.)
        ps=[(a,b,correspondence(a,b,ch)) for a in range(5) for b in range(a+1,5) for ch in ('left','right')]
        for labels in ('noisy_operator_labels','perfect_labels'):
            hs=[cap['known_liquid_height_m'] for cap in c['priors']] if labels=='noisy_operator_labels' else [t[0] for t in truth[:5]]
            prior=fit_curve(ps,hs,cal,c['condition'])
            for j in range(len(c['tests'])):
                gs=[(a,correspondence(a,j+5,ch)) for a in range(5) for ch in ('left','right')]
                pred=infer_curve(gs,prior,cal)['height_m'];h=truth[j+5][0]
                records.append(dict(config_id=c['id'],labels=labels,material_regime=c['material_regime'],true_height_m=h,
                    predicted_height_m=pred,absolute_error_m=abs(pred-h),evaluator_only=True))
    dump(OUT/'oracle-correspondence-raw.json',records)
    return {label:dict(overall=metrics([r for r in records if r['labels']==label]),
        intervals=intervals([r for r in records if r['labels']==label])) for label in ('noisy_operator_labels','perfect_labels')}


def registration_audit():
    cs=json.loads((DATA/'test/manifest.json').read_text())['configs'];out={}
    for c in cs:
        if c['material_regime']=='legacy_depth_attenuation':continue
        hs=[];n=None
        for cap in c['priors']:
            with np.load(cap['evaluator_truth']) as f:hs.append(float(f['height_m']));n=float(f['index'])
        theoretical=-np.log1p(-(1-1/n)*np.asarray(hs)/c['bottom_m'])
        stats=out.setdefault(c['material_regime'],dict(prior_edges=0,paraxial_true_scale_exceeds_old_bound=0,
            old_observed_near_bound=0,wide_observed_near_bound=0))
        for kind,key,limit in [('baseline','old_observed_near_bound',.118),('wide_s3','wide_observed_near_bound',.498)]:
            with np.load(Path(c['folder'])/('tracks_'+kind+'.npz')) as f:
                for a in range(5):
                    for b in range(a+1,5):
                        for ch in ('left','right'):
                            prefix=f'{a}_{b}_{ch}_';p=dict(anchor_q=f[prefix+'anchor_q'],current_q=f[prefix+'current_q'])
                            if len(p['anchor_q'])<6:continue
                            s=base.observed_log_scale(p)
                            stats[key]+=int(s is not None and abs(s)>=limit)
                            if kind=='baseline':
                                stats['prior_edges']+=1
                                stats['paraxial_true_scale_exceeds_old_bound']+=int(abs(theoretical[b]-theoretical[a])>.12)
    return out


def fmt(v,dec=2):return '—' if v is None else f'{v:.{dec}f}'


def table(rows):
    lines=['| 场景 / 区间 | 帧数 | 原 MAE (mm) | 优化 MAE (mm) | 原通过率 | 优化通过率 | 数值输出率 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,b,a in rows:
        lines.append(f"| {name} | {a['frames']} | {fmt(b['mae_mm'])} | {fmt(a['mae_mm'])} | {fmt(b['pass_pct'])}% | {fmt(a['pass_pct'])}% | {fmt(a['numeric_output_pct'])}% |")
    return '\n'.join(lines)


def main():
    freeze=json.loads((OUT/'frozen-source.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in freeze.items()),'source_changed'
    policy=json.loads((OUT/'frozen-policy.json').read_text())['selection']
    raw=json.loads((OUT/'test-raw.json').read_text())['records'];b=[r for r in raw if r['stage']=='baseline_v30'];a=selected(raw,policy)
    regimes=sorted({r['material_regime'] for r in a});results={};tables=[]
    for m in regimes:
        bb=[r for r in b if r['material_regime']==m];aa=[r for r in a if r['material_regime']==m]
        results[m]=dict(before=metrics(bb),after=metrics(aa),before_intervals=intervals(bb),after_intervals=intervals(aa))
        tables.append((m,results[m]['before'],results[m]['after']))
    target_a=[r for r in a if r['material_regime']!='legacy_depth_attenuation'];target_b=[r for r in b if r['material_regime']!='legacy_depth_attenuation']
    results['target_total']=dict(before=metrics(target_b),after=metrics(target_a),bootstrap=bootstrap(target_b,target_a))
    results['all_including_boundary']=dict(before=metrics(b),after=metrics(a))
    tables += [('目标链路合计（不含边界材料）',results['target_total']['before'],results['target_total']['after']),
               ('全部（含底纹消失边界材料）',results['all_including_boundary']['before'],results['all_including_boundary']['after'])]
    oracle=oracle_audit();registration=registration_audit()
    result=dict(policy=policy,metrics=results,timing=timing(),oracle_evaluator_only=oracle,
        prior_registration_evaluator_audit=registration,test_measurement_groups=len(a),
        frozen_source_and_policy_unchanged=True,all_frame_retention=True)
    dump(OUT/'final-analysis.json',result)
    lines=['# 透明半透明：实际液深 0–9 米均匀采样与分段优化','',
        '日期：2026-10-12。仅受控仿真研究，未作为工业认证或默认生产算法。','',
        '## 结论先读','',
        f"底纹仍可观察的目标链路共{len(target_a)}组：MAE {results['target_total']['before']['mae_mm']:.2f}→{results['target_total']['after']['mae_mm']:.2f} mm，通过率{results['target_total']['before']['pass_pct']:.2f}%→{results['target_total']['after']['pass_pct']:.2f}%，正液深AbsRel {results['target_total']['after']['absrel_positive_pct']:.2f}%。这些结果不包括另列的光学不可观测边界材料。",'',
        '主要不足仍是0–1 m、2–3 m和空槽零位。空槽端点的5 mm通过率仅43.75%，不能因全范围平均误差好就称零位已达标。',
        '底纹消失边界材料严重失败：MAE 28.81 m、最大误差约801.84 m。这是放宽门控后弱响应反解/外推的数值发散，不是扩大实际量程；所有错误数值都保留计分，未按容量裁掉。该分支不能用于工业输出，也不能把100%数值输出解读为100%可信覆盖。原生产安全门控未被修改。',
        '透明零衰减和半透明衰减系数是仿真代理设定，未经真实相机光谱、照明和液体参数标定；不能由此保证实际RGB-D红外设备能穿透9 m液体。','',
        '## 范围与评价口径','',
        '- H 是相对容器底面的实际液深，均匀覆盖 0–9 m，含端点；B 是相机到底面的距离，始终不超过 10 m。',
        '- 开发集 B=9.5/9.9 m、C=9.15/9.25 m；独立测试 B=9.7/10 m、C=9.3/9.4 m。深槽不等于远距离浅容器。',
        '- 5组先验为1组空容器加4组已知液深；开发冻结布置为0/15/40/65/90%容量。每组含左右目，非单次曝光；人工先验标签有0.5 mm+0.1%H标准差噪声。',
        '- 开发32配置×37等间距液深（间隔0.25 m）=1184组；新测试64配置×73等间距液深（间隔0.125 m）=4672组。种子、几何、折射率、纹理尺度及半透明衰减参数均与开发不同。',
        '- 本轮最小非零测试液深125 mm；0米端点独立采集，但不能据此声称0–125 mm内毫米级液深均已验证。',
        '- 不继续小容器优化：空容器底面宽128/160原生像素。固定480×320传感器裁剪，不靠扩大传感器图像或无限底纹提高成绩。',
        '- 通过标准 abs(pred−H)≤max(5 mm,2%H)。通过率以全部测试组为分母；数值输出率不是可信覆盖率。未强制置信度/覆盖率筛选、不裁剪到容器容量。MAE/P95/AbsRel只对有限数值计算，缺失帧仍计入总帧数并作为不通过。',
        '- H=0只报绝对误差与5 mm通过率，不定义相对误差。正液深AbsRel单列，不能将绝对误差/相机距离冒充液深相对误差。','',
        '## 改进内容与选型隔离','',
        '保留V30基线原样重测。新增原生像素宽尺度配准（log尺度±0.5）、掩码外最近内部值填充防止背景泄漏、抑制固定投影散斑的空间滤波、正向/反向质量择优。',
        '5帧先验构成相对尺度图，联合求单调响应节点；多参考按图像相关性软加权，端点线性外推且不按容量截断。无测试真值、折射率或衰减系数进入推理。有效响应不等于识别出了液体物理折射率。',
        '开发集比较baseline_v30、wide_response、wide_curve、wide_curve_best。先保存第一轮结果，再仅在开发集验证滤波迭代。选型按等权液深区间误差、尾部误差和缺失惩罚，冻结源码及策略后才生成独立测试。',
        f"冻结选择：{policy}。所有数值为诊断输出qualified=false；不改变默认工业安全门控。",'',
        '## 全帧结果','',table(tables),'',
        '低/中衰减是光学代理参数，不是新的自动液体类别。旧材料extinction=0.65/m在9 m时底纹直达透射仅exp(−11.7)，单列为光程边界审计，不参加本轮优化选型。不能将此类失去底纹的输入直接宣称已由普通路线解决。','']
    for m in regimes:
        r=results[m];lines+=['## '+m+' 分液深结果','',table([(k,r['before_intervals'][k],s) for k,s in r['after_intervals'].items()]),'']
        s=r['after'];lines += [f"优化后P95绝对误差 {fmt(s['p95_mm'])} mm，最大误差 {fmt(s['max_mm'])} mm；正液深AbsRel {fmt(s['absrel_positive_pct'])}%。严格逐帧1%通过 {s['strict_1pct_passed_positive']}/{s['positive_total_frames']}。",'']
        if m!='legacy_depth_attenuation':
            weak=sorted(((k,v) for k,v in r['after_intervals'].items() if k!='zero'),key=lambda kv:(kv[1]['pass_pct'],-(kv[1]['normalized_mean'] or 0)))[:3]
            lines += ['最低通过率的三个区间：'+ '；'.join(f"{k.replace('_','–')} m：{fmt(v['pass_pct'])}%、MAE {fmt(v['mae_mm'])} mm" for k,v in weak)+'。这是诊断排序，不是按真值液深切换推理算法。','']
    boot=results['target_total']['bootstrap']
    lines+=['## 配对不确定性与延时','',
        f"目标场景按纹理种子聚类（{boot['clusters']}簇，3000次bootstrap）；MAE改善95%区间(mm)={boot['mae_improvement_mm_ci95']}，通过率改善区间(百分点)={boot['pass_improvement_percentage_points_ci95']}。同一纹理的73个液深不是73个独立场景。",'']
    for t in result['timing']:
        lines.append(f"- {t['condition']}、底纹宽{t['empty_width_px']}px：CPU单线程、不读匹配缓存、6个测试组；5先验拟合{t['prior_fit_s']:.3f}s，测量P50/P95={t['p50_ms']:.1f}/{t['p95_ms']:.1f}ms；相对缓存评估最大差{t['max_cached_difference_m']:.3g}m。未包含采集/传输/显示，不能作为500ms端到端认证。")
    lines+=['','## 剩余不足与归因边界','',
        '配准尺度上限不是唯一问题：投影纹理与底纹有不同运动、相似变换不能完全表达大视角Snell畸变、5帧校正曲线存在插值/外推误差，低位液深的2%容差又更严格。应依据上表定位欠佳区间，不能仅看全量平均数宣布0–9m全量程达标。',
        '0–1 m及零位优先方向：抑制近零残余位移偏置，使用左右通道及跨先验循环一致性联合估计尺度/偏置，诊断噪声下的零位重复性。不能通过按真值判断空槽、将负值强行归零或裁剪测量值来伪造改进。',
        '2–3 m优先方向：目前先验相邻点约1.4 m与3.7 m，中间没有标定点；尝试带Snell形状约束的单调有理响应或形状保持插值，减少直线反解的曲率误差，再处理投影纹理造成的配准非传递性。仍保持5帧预算，仅在开发集拟合，并用另一批独立测试验证。',
        '半透明深光程会使底纹对比度衰减、模糊和噪声相对增大；软加权只能降低坏配准影响，不能创造已消失的信息。无数值帧保留为失败；有限但错误的数值同样计入误差和通过率。',
        '该仿真为单一均匀液体、顶视水平液面/底面、精确固定几何、有限底纹的Snell+衰减/散斑代理。不是设备标定后的完整RGB-D传感器仿真；没有玻璃侧壁、多层液体、波浪、实际ToF多径、完整结构光相位展开或主动双目硬件解算。',
        '左右目在此分别做跨先验时间配准并融合，不应称为已验证了所有真实左右目视差/结构光链路。没有新增神经网络训练或真实相机数据验证。',
        '下一步先在开发数据上针对最弱区间研究投影/底纹分离、Snell非线性配准和低液位校正；物理不可观测样本需可靠表面回波/其他传感器或拒绝，不能靠放开门控获得真实精度。','',
        '## 复现与证据','',
        f'- 数据：`{DATA}`；记录与冻结哈希：`{OUT}`。',
        '- 入口：`scripts/experiment_range_v31.py`；后验报告：`scripts/audit_range_v31.py`；完整逐帧数据test-raw.json，汇总final-analysis.json。',
        '- 新增范围、独立零位、掩码防泄漏、宽尺度恢复、单调校准与不裁剪外推单元测试；完整回归结果见regression.log。',
        '- 保留全部V30和开发迭代证据；研究候选不自动推广到生产。','']
    lines+=['## 理想对应点反事实（仅评估器）','',
        '将匹配点替换为真值光线对应，保持相同5帧与校正方法，分有噪人工标签和理想标签。该结果只估计匹配/标签以外的校正误差，不能部署、不能计入正式性能。','',
        table([(label,results['target_total']['after'],o['overall']) for label,o in oracle.items()]),'',
        '上表原MAE/通过率列为真实图像优化链路对照，优化列为不可部署的理想对应点反事实。','']
    lines+=['## 原配准搜索范围归因（仅评估器）','',
        '使用真值折射率计算近轴理论尺度，仅用于检查旧±0.12 log尺度是否覆盖先验跨度；不是推理输入。以下计数以左右通道的先验配准边为单位，不是独立测量帧。','']
    for m,s in registration.items():
        lines.append(f"- {m}：{s['prior_edges']}条先验边中，理论超旧尺度界{s['paraxial_true_scale_exceeds_old_bound']}条；旧配准观测贴界{s['old_observed_near_bound']}条，新宽尺度配准贴界{s['wide_observed_near_bound']}条。")
    text='\n'.join(lines)
    (OUT/'report.md').write_text(text,encoding='utf-8');(REPO/'docs/transparent_range_v31.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(metrics=results,timing=result['timing']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
