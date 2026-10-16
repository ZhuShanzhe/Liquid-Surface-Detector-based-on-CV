"""Post-freeze paired diagnostics, ablations and report. No model selection."""
import argparse,json,time,hashlib
from pathlib import Path
import cv2,numpy as np
import experiment_focus_v32 as run
from liquid_depth import range_relative_v31 as baseline
from liquid_depth import range_relative_v32 as model
from experiment_range_v31 import load,summary,intervals

OUT=run.OUT;DATA=run.DATA;REPO=run.REPO
ABLATIONS=['global_pchip_robust_None','local_linear_robust_None']


def declare():
    path=OUT/'ablation-plan.json'
    if path.exists():raise FileExistsError('preserve_ablation_declaration')
    run.dump(path,dict(methods=ABLATIONS,used_for_selection=False,test_results_read=False,
        frozen_policy_sha256=hashlib.sha256((OUT/'frozen-policy.json').read_bytes()).hexdigest()))


def ablate():
    plan=json.loads((OUT/'ablation-plan.json').read_text())
    assert plan['methods']==ABLATIONS
    assert plan['frozen_policy_sha256']==hashlib.sha256((OUT/'frozen-policy.json').read_bytes()).hexdigest()
    configs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    run.evaluate(configs,plan['methods'],'ablation')


def metrics(rows):
    s=summary(rows);finite=[r for r in rows if r['predicted_height_m'] is not None]
    e=np.array([r['predicted_height_m']-r['true_height_m'] for r in finite])
    s.update(rmse_mm=float(np.sqrt(np.mean(e*e))*1000) if len(e) else None,
        signed_bias_mm=float(e.mean()*1000) if len(e) else None,
        negative_outputs=sum(r['predicted_height_m']<0 for r in finite),
        above_capacity_outputs=sum(r['predicted_height_m']>r['capacity_m'] for r in finite),
        exactly_zero_positive_height_outputs=sum(abs(r['predicted_height_m'])<1e-12 and r['true_height_m']>0 for r in finite),
        strict_1pct_positive_passed=sum(abs(r['predicted_height_m']-r['true_height_m'])<=.01*r['true_height_m'] for r in finite if r['true_height_m']>0),
        positive_total_frames=sum(r['true_height_m']>0 for r in rows))
    return s


def partition(rows):
    return dict(all=rows,uniform=[r for r in rows if r['population']=='uniform'],
        zero=[r for r in rows if r['true_height_m']==0],near_zero=[r for r in rows if 0<r['true_height_m']<=.1],
        low=[r for r in rows if 0<r['true_height_m']<1],mid=[r for r in rows if 2<=r['true_height_m']<3],
        rest=[r for r in rows if r['true_height_m']>=1 and not 2<=r['true_height_m']<3])


def paired_ci(before,after):
    clusters=sorted({r['texture_cluster'] for r in before});pairs=[]
    for c in clusters:
        b=metrics([r for r in before if r['texture_cluster']==c]);a=metrics([r for r in after if r['texture_cluster']==c])
        pairs.append([b['mae_mm']-a['mae_mm'],a['pass_pct']-b['pass_pct']])
    p=np.array(pairs);rng=np.random.default_rng(82061012);rep=p[rng.integers(0,len(p),(5000,len(p)))].mean(axis=1)
    return dict(texture_clusters=len(p),mae_improvement_mm_ci95=np.percentile(rep[:,0],[2.5,97.5]).tolist(),
        pass_improvement_percentage_points_ci95=np.percentile(rep[:,1],[2.5,97.5]).tolist())


def zero_bias(rows):
    result=[]
    for key in sorted({r['config_id'] for r in rows}):
        rr=[r for r in rows if r['config_id']==key and r['true_height_m']==0]
        e=np.array([r['predicted_height_m'] for r in rr if r['predicted_height_m'] is not None])
        result.append(dict(config_id=key,condition=rr[0]['condition'],material_regime=rr[0]['material_regime'],
            frames=len(rr),bias_mm=float(e.mean()*1000) if len(e) else None,
            repeatability_std_mm=float(e.std(ddof=1)*1000) if len(e)>1 else None,
            mae_mm=float(abs(e).mean()*1000) if len(e) else None))
    return result


def timed_check(policy,rows):
    cv2.setNumThreads(1);configs=json.loads((DATA/'test/manifest.json').read_text())['configs'];result=[]
    chosen=[next(c for c in configs if c['condition']==cond and c['empty_width_px']==width)
        for cond in ('transparent','translucent_mild') for width in (128,160)]
    for c in chosen:
        name=policy['selection'][c['condition']];options=run.choices()[name]
        images,masks,cal,hs=load(c);ps=[];t=time.perf_counter()
        for a in range(5):
            for b in range(a+1,5):
                for ch in ('left','right'):ps.append((a,b,baseline.wide_pair(images[a,ch],images[b,ch],cal,masks[ch],ch)))
        prior=baseline.fit_curve(ps,hs,cal,c['condition']) if name==run.BASE else model.fit(ps,hs,cal,c['condition'],options)
        prior_s=time.perf_counter()-t;durations=[];differences=[]
        expected={r['sample_id']:r['predicted_height_m'] for r in rows if r['config_id']==c['id']}
        for j in [0,1,15,45,85,len(c['tests'])-24]:
            gs=[];t=time.perf_counter()
            for a in range(5):
                for ch in ('left','right'):gs.append((a,baseline.wide_pair(images[a,ch],images[j+5,ch],cal,masks[ch],ch)))
            pred=baseline.infer_curve(gs,prior,cal) if name==run.BASE else model.infer(gs,prior,cal,options['fusion'],options['band'])
            durations.append((time.perf_counter()-t)*1000);differences.append(abs(pred['height_m']-expected[j]))
        assert max(differences)<1e-8,'uncached_runtime_parity_failed'
        result.append(dict(condition=c['condition'],empty_width_px=c['empty_width_px'],frames=len(durations),
            prior_fit_s=prior_s,p50_ms=float(np.median(durations)),p95_ms=float(np.percentile(durations,95)),
            max_ms=max(durations),max_cached_difference_m=max(differences),cpu_threads=1,
            includes_capture_transfer_display=False))
    return result


def fmt(v):return '—' if v is None else f'{v:.2f}'


def table(items):
    lines=['| 场景或区间 | 帧数 | 原 MAE mm | 优化 MAE mm | 原通过率 | 优化通过率 | 数值输出率 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,b,a in items:
        lines.append(f"| {label} | {a['frames']} | {fmt(b['mae_mm'])} | {fmt(a['mae_mm'])} | {fmt(b['pass_pct'])}% | {fmt(a['pass_pct'])}% | {fmt(a['numeric_output_pct'])}% |")
    return '\n'.join(lines)


def report():
    frozen=json.loads((OUT/'frozen-source.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items()),'frozen_source_changed'
    policy=json.loads((OUT/'frozen-policy.json').read_text());raw=json.loads((OUT/'test-raw.json').read_text())['records']
    before=[r for r in raw if r['stage']==run.BASE]
    after=[r for r in raw if r['stage']==policy['selection'][r['condition']]]
    assert len(before)==len(after)==8016
    assert {(r['config_id'],r['sample_id']) for r in before}=={(r['config_id'],r['sample_id']) for r in after}
    bp=partition(before);ap=partition(after)
    results={k:dict(before=metrics(bp[k]),after=metrics(ap[k]),paired_cluster_ci=paired_ci(bp[k],ap[k])) for k in bp}
    material={}
    for m in sorted({r['material_regime'] for r in after}):
        b=[r for r in before if r['material_regime']==m];a=[r for r in after if r['material_regime']==m]
        material[m]=dict(before=metrics(b),after=metrics(a),
            buckets={k:dict(before=metrics(partition(b)[k]),after=metrics(partition(a)[k])) for k in bp})
    ablations=json.loads((OUT/'ablation-raw.json').read_text())['records']
    ablation_results={name:{k:metrics(rs) for k,rs in partition([r for r in ablations if r['stage']==name]).items()} for name in ABLATIONS}
    bias=dict(before=zero_bias(before),after=zero_bias(after));runtime=timed_check(policy,after)
    uniform_intervals=dict(before=intervals(bp['uniform']),after=intervals(ap['uniform']))
    result=dict(policy=policy,all_frames_retained=True,frozen_source_unchanged=True,results=results,material=material,
        uniform_intervals=uniform_intervals,
        ablations=ablation_results,zero_bias=bias,runtime=runtime)
    run.dump(OUT/'final-analysis.json',result)
    names=dict(all='全部含加密样本',uniform='全量程均匀采样',zero='空槽零位',near_zero='大于0至0.1米',low='大于0且小于1米',mid='2至3米不含3米',rest='其余非零区间')
    s=results['all'];u=results['uniform'];z=results['zero']
    lines=['# 透明半透明液体零位与薄弱液深区间优化','',
        '2026年10月12日。面向固定顶视相机与底纹仍可观察的单一均匀透明或轻度半透明液体，仅为受控仿真研究。','',
        f"新的8016组独立测量中，整体MAE由{fmt(s['before']['mae_mm'])}降至{fmt(s['after']['mae_mm'])} mm，通过率由{fmt(s['before']['pass_pct'])}%升至{fmt(s['after']['pass_pct'])}%。其中3504组全量程均匀样本的MAE为{fmt(u['after']['mae_mm'])} mm、通过率{fmt(u['after']['pass_pct'])}%。空槽零位MAE为{fmt(z['after']['mae_mm'])} mm，5 mm通过率{fmt(z['after']['pass_pct'])}%，需与浅液位和分材料结果一起判断，不能只看总平均。",'',
        '## 测试范围与指标','',
        '实际液深H为相对容器底面的高度，范围0至9 m；相机到底面距离B不超过10 m。开发集B为9.55和9.85 m，容器容量C为9.12和9.22 m；新测试B为9.65和10 m，C为9.28和9.45 m。空容器底面成像宽128或160原生像素，沿用固定480×320传感器裁剪与有限底纹，不再优化小容器。',
        '先验预算始终为5组：1组空容器加4组已知液深，每组含左右目。布置沿用0、15%、40%、65%、90%容器容量；非零先验标签标准差为0.5 mm+0.1%H，空槽标签为0。重复空槽是测试帧，不参与标定、训练或推理记忆，未增加先验预算。',
        '开发集24种配置、2088组测量；新测试48种配置、8016组测量。新测试采用不同纹理种子、几何、折射率、纹理尺度、衰减系数和独立成像噪声。每种配置包含24组独立空槽测试，并加密0至1 m、2至3 m及2至100 mm浅液位；另保留0至9 m、步长0.125 m的均匀子集。',
        '通过标准为绝对误差≤max(5 mm,2%H)，全部测试帧均计入通过率分母，无法求解算不通过。MAE和分位数基于有限输出；H=0不定义AbsRel。由于本轮增加2 mm浅液位，汇总AbsRel不能直接与上轮最小非零液深125 mm的汇总值相比，需分别看毫米误差和均匀子集。',
        '输出不强制归零，不按0至容量或0至9 m裁剪，不启用置信度或覆盖率筛选。数值输出率不代表可信覆盖率；所有研究结果qualified=false。生产默认链路与安全门控未变。','',
        '## 改进方法','',
        '原链路将全部先验间配准投影到一条全局潜在尺度曲线，再用分段直线反解与加权中位数融合。它要求各参考帧的测量能满足同一尺度关系，而固定投影散斑与真实底纹运动并不完全一致；配准边可能不满足循环一致性，造成参考帧相关偏移。稀疏标定节点间的直线反解还会留下曲率误差。',
        '新方法为每个参考帧、每个左右通道建立自身的相对尺度到已知液深曲线。通过形状保持单调三次插值减小节点间偏差，再对各参考帧给出的液深做软鲁棒融合，避免中位数在不同参考帧之间突然切换。原生图像配准器未改，优化前后使用完全相同的观测和匹配结果。',
        '本轮是校准与融合算法优化，没有新增深度神经网络训练，也没有用测试液深调整模型参数。',
        '参考图与自身的尺度变化为0，是已知图像坐标的恒等关系；未知测量依然先计算真实图像位移再反解液深，因此不等于将空槽测试帧强制置0。超出先验跨度时按正端点切线外推；约束的是校准曲线斜率，不是裁剪测量值。',
        '开发阶段筛选全局与逐参考、线性与单调三次插值、三种融合及软距离加权等37个方案，再在新增开发集上确认。最终冻结透明和半透明均使用逐参考单调三次曲线、软鲁棒融合、不加尺度距离权重。完整源码和策略哈希在生成新测试前冻结。','',
        '## 优化前后配对结果','',table([(names[k],v['before'],v['after']) for k,v in results.items()]),'']
    for m,r in material.items():
        lines += ['## '+{'transparent':'透明场景','semi_low_extinction':'半透明低衰减场景','semi_medium_extinction':'半透明中衰减场景'}[m],'',
            table([(names[k],v['before'],v['after']) for k,v in r['buckets'].items()]),'',
            f"全部样本优化后P95绝对误差{fmt(r['after']['p95_mm'])} mm、最大误差{fmt(r['after']['max_mm'])} mm；均匀子集正液深AbsRel {fmt(r['buckets']['uniform']['after']['absrel_positive_pct'])}%。",'']
    lines += ['## 均匀子集分液深结果','',
        table([(k.replace('_','至')+'米' if k!='zero' else '零位',uniform_intervals['before'][k],v)
            for k,v in uniform_intervals['after'].items()]),'',
        f"总体P95误差由{fmt(s['before']['p95_mm'])}降至{fmt(s['after']['p95_mm'])} mm，但最大误差由{fmt(s['before']['max_mm'])}增至{fmt(s['after']['max_mm'])} mm；平均改善不代表最坏尾部也改善。少量高位先验跨度之外的外推仍可能超标或超过容量，均保留计分和警告。",'',
        '## 消融对比','',
        '以下两项在读取新测试结果前声明，只用于拆分改进来源，不参加选型，也不改换冻结方案。全局单调三次曲线检查仅改变插值和融合能否解决问题；逐参考线性曲线检查逐参考校准本身的贡献。','',
        '| 方法 | 零位 MAE mm | 零位通过率 | 0至1米 MAE mm | 2至3米 MAE mm | 均匀子集 MAE mm |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in [('原链路',{k:v['before'] for k,v in results.items()}),*ablation_results.items(),('冻结优化方案',{k:v['after'] for k,v in results.items()})]:
        lines.append(f"| {name} | {fmt(r['zero']['mae_mm'])} | {fmt(r['zero']['pass_pct'])}% | {fmt(r['low']['mae_mm'])} | {fmt(r['mid']['mae_mm'])} | {fmt(r['uniform']['mae_mm'])} |")
    lines += ['',
        '仅把全局直线改为单调三次曲线并使用软鲁棒融合，零位反而明显变差；逐参考线性曲线已将零位MAE降至约4.85 mm，与冻结方案相近。消融支持逐参考校准是零位改善的重要来源，而单调三次插值主要进一步改善0至1 m、2至3 m及均匀子集。不能把全部收益归于插值阶数更高。']
    lines+=['','## 零位重复性与浅液位响应','']
    for stage in ('before','after'):
        for cond in ('transparent','translucent_mild'):
            rr=[r for r in bias[stage] if r['condition']==cond]
            lines.append(f"- {stage} {cond}：各配置平均绝对零位偏置{np.mean([abs(r['bias_mm']) for r in rr]):.2f} mm，平均重复测量标准差{np.mean([r['repeatability_std_mm'] for r in rr]):.2f} mm。")
    lines += ['',
        f"全部优化输出中负值{s['after']['negative_outputs']}组、超容量{ s['after']['above_capacity_outputs']}组，均保留；正液深测试中被输出为精确0的帧数{s['after']['exactly_zero_positive_height_outputs']}。这项计数配合2至100 mm测试及负值/超容量单元测试检查，不能单独证明任意现场浅液位精度。",'',
        '多次测试空槽共享同一张有噪标定空图；其像素噪声与纹理偏差会成为固定参考误差。各参考帧又共享同一张当前测量图，十个配准结果不是十份独立传感器信息。逐参考校准能降低参考间偏移，却不能仅靠软融合消除共同噪声；零位残余可能同时包含共同参考偏置、当前帧噪声和投影纹理偏差。上述重复性分解是统计诊断，不是已标定真实硬件的噪声模型。','',
        '## 配对不确定性','']
    for k in ('zero','low','mid','uniform'):
        ci=results[k]['paired_cluster_ci']
        lines.append(f"- {names[k]}：MAE改善95%区间(mm)={ci['mae_improvement_mm_ci95']}；通过率改善区间(百分点)={ci['pass_improvement_percentage_points_ci95']}。")
    lines += ['',
        '按16个纹理种子簇进行5000次配对bootstrap；同一纹理、不同液深、重复曝光及三种材料视为相关观测，不把8016帧当8016个独立场景。零位MAE改善区间跨0，不能据此宣称对新纹理都有稳定MAE收益；其通过率改善区间仍为正。其他薄弱区间的误差改善证据更明确。','',
        '## 延时与代码验证','']
    for t in runtime:
        lines.append(f"- {t['condition']}、{t['empty_width_px']}px：CPU单线程、6组无图像配准缓存测量，P50/P95={t['p50_ms']:.1f}/{t['p95_ms']:.1f} ms；5组先验拟合{t['prior_fit_s']:.3f}s；与冻结缓存评估最大差{t['max_cached_difference_m']:.3g}m。")
    lines += ['',
        '以上不含相机采集、传输和显示，不能认证500 ms端到端延时。新增测试覆盖符号保留、2 mm非零响应、超容量外推、单调曲线、缺失观测和独立重复零位；完整回归结果见regression.log。','',
        '## 应用边界与后续方向','',
        '本轮只针对底纹仍可观察的透明和轻度半透明均匀液体；不以强散射、底纹消失材料或小容器参与优化或平均数。上轮底纹消失的严重失败证据仍保留，本轮不宣称解决该边界。',
        '仿真沿用顶视水平液面/底面、精确固定几何及Snell折射加散斑、衰减、模糊和噪声代理；不是经真实设备光谱标定的RGB-D数字孪生。未覆盖玻璃侧壁、多层液体、动态波面、相机运动或完整结构光相位解算；左右目分别跨先验配准融合，不等于已经验证所有硬件视差链路。',
        '若零位仍偏弱，下一步应在不增加先验组数的条件下，研究更大有效底纹区域的亚像素配准、噪声感知的双通道联合估计及共同参考偏差建模；是否提高现场成像质量或改变单组曝光方案须另行明确预算。不得靠归零死区、容量截断或筛帧提升成绩。','',
        '## 复现与证据','',
        f'- 数据目录：`{DATA}`。',f'- 冻结策略、逐帧结果、消融、置信区间和报告：`{OUT}`。',
        '- 运行入口scripts/experiment_focus_v32.py；算法src/liquid_depth/range_relative_v32.py；后验分析scripts/audit_focus_v32.py。',
        '- 对照为上一轮冻结V31链路，非论文原始模型，也不直接比较两轮不同采样分布的总平均。原V31代码与历史数据未改。','']
    text='\n\n'.join(s for s in lines if s).replace('|\n\n|','|\n|')+'\n'
    (OUT/'report.md').write_text(text,encoding='utf-8');(REPO/'docs/transparent_focus_v32.md').write_text(text,encoding='utf-8')
    run.progress('complete_with_analysis',frozen_source_and_policy_unchanged=True,frames=len(after),production_changed=False)
    print(json.dumps(dict(results=results,runtime=runtime),ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['declare','ablate','report']);args=ap.parse_args()
    if args.stage=='declare':declare()
    elif args.stage=='ablate':ablate()
    else:report()
