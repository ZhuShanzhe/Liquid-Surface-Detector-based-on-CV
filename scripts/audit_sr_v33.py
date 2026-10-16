"""Distribution, paired uncertainty, and uncached runtime audit after freezing."""
import json,hashlib,time
from pathlib import Path
import cv2,numpy as np
import experiment_sr_v33 as run
from experiment_range_v31 import summary,load
from liquid_depth import range_registration_v33 as reg
from liquid_depth import range_relative_v32 as model

OUT=run.OUT
NAMES=dict(baseline_v32='优化前链路',bicubic_x2='双三次放大两倍',swinir_x2='SwinIR轻量超分两倍',
 native_dense='原生图像加密采样',midpoint_650='对称配准650点',midpoint_dense='对称配准2000点',
 midpoint_sigma2='对称配准低通2像素',midpoint_sigma4='对称配准低通4像素',midpoint_spline5='对称配准五阶插值')
NAMES.update(midpoint_core30='对称配准中央60%区域',midpoint_bicubic2='双三次两倍加对称配准',midpoint_swinir2='SwinIR两倍加对称配准')

def stats(rows):
    result=summary(rows);finite=[r for r in rows if r['predicted_height_m'] is not None]
    e=np.array([(r['predicted_height_m']-r['true_height_m'])*1000 for r in finite]);a=abs(e)
    bounds=[2,5,10,20,50,100,np.inf];bins=[];last=-np.inf
    for b in bounds:
        n=int(sum((a>last)&(a<=b)));bins.append(dict(upper_mm=float(b) if np.isfinite(b) else None,count=n,pct=100*n/max(len(rows),1)));last=b
    result.update(absolute_bins=bins,missing=len(rows)-len(finite),median_mm=float(np.median(a)) if len(a) else None,
         p99_mm=float(np.percentile(a,99)) if len(a) else None,bias_mm=float(e.mean()) if len(e) else None,
         negative=sum(r['predicted_height_m']<0 for r in finite),above_capacity=sum(r['predicted_height_m']>r['capacity_m'] for r in finite),
         positive_exact_zero=sum(r['true_height_m']>0 and r['predicted_height_m']==0 for r in finite))
    assert sum(x['count'] for x in bins)+result['missing']==len(rows)
    pos=[r for r in finite if r['true_height_m']>0];relative=np.array([r['absolute_error_m']/r['true_height_m']*100 for r in pos])
    pn=sum(r['true_height_m']>0 for r in rows)
    result['relative_bins']=[];last=-np.inf
    for b in [.5,1,2,3,5,10,np.inf]:
        n=int(sum((relative>last)&(relative<=b)));result['relative_bins'].append(dict(upper_pct=float(b) if np.isfinite(b) else None,count=n,pct=100*n/pn if pn else None));last=b
    return result

def subsets(rows):
    return dict(all=rows,zero=[r for r in rows if r['true_height_m']==0],near=[r for r in rows if 0<r['true_height_m']<=.1],
         low=[r for r in rows if 0<r['true_height_m']<1],mid=[r for r in rows if 2<=r['true_height_m']<3],
         rest=[r for r in rows if r['true_height_m']>=1],uniform=[r for r in rows if r['population']=='uniform'])

def ci(before,after):
    clusters=sorted({r['texture_cluster'] for r in before});pairs=[]
    for c in clusters:
        b=stats([r for r in before if r['texture_cluster']==c]);a=stats([r for r in after if r['texture_cluster']==c])
        pairs.append([b['mae_mm']-a['mae_mm'],a['pass_pct']-b['pass_pct']])
    x=np.array(pairs);rng=np.random.default_rng(1900127);rep=x[rng.integers(0,len(x),(5000,len(x)))].mean(axis=1)
    return dict(clusters=len(x),mae_improvement_mm_ci95=np.percentile(rep[:,0],[2.5,97.5]).tolist(),
         pass_improvement_pp_ci95=np.percentile(rep[:,1],[2.5,97.5]).tolist())

def compare(before,after,uncertainty=False):
    assert {(r['config_id'],r['sample_id']) for r in before}=={(r['config_id'],r['sample_id']) for r in after}
    out={k:dict(before=stats(v),after=stats(subsets(after)[k])) for k,v in subsets(before).items()}
    if uncertainty:
        for k,v in out.items():v['paired_cluster_ci']=ci(subsets(before)[k],subsets(after)[k])
    return out

def runtime(policy,rows):
    cv2.setNumThreads(1);result=[];sr=None
    for condition in ('transparent','translucent_mild'):
        for width in (128,160):
            c=next(c for c in run.configs('test') if c['condition']==condition and c['empty_width_px']==width)
            ims,masks,cal,hs=load(c);name=policy['selection'][condition];factor=1 if name==run.BASE else reg.OPTIONS[name]['factor']
            mm={ch:cv2.resize(m,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST) if factor!=1 else m for ch,m in masks.items()}
            def prepare(keys):
                nonlocal sr
                if 'swinir' in name:
                    if sr is None:sr=reg.SwinIRX2()
                    ys=sr([ims[k] for k in keys]);return dict(zip(keys,ys))
                if factor==2:return {k:cv2.resize(ims[k].astype(np.float32),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) for k in keys}
                return {k:ims[k] for k in keys}
            t=time.perf_counter();pr=prepare([(j,ch) for j in range(5) for ch in ('left','right')]);ps=[]
            for a in range(5):
                for b in range(a+1,5):
                    for ch in ('left','right'):ps.append((a,b,reg.pair(pr[a,ch],pr[b,ch],cal,mm[ch],ch,name)))
            prior=model.fit_local(ps,hs,cal,condition);prior_s=time.perf_counter()-t;times=[];diff=[]
            expected={r['sample_id']:r['predicted_height_m'] for r in rows if r['config_id']==c['id']}
            for j in [0,1,20,60,120,143]:
                t=time.perf_counter();cur=prepare([(j+5,ch) for ch in ('left','right')]);gs=[]
                for a in range(5):
                    for ch in ('left','right'):gs.append((a,reg.pair(pr[a,ch],cur[j+5,ch],cal,mm[ch],ch,name)))
                pred=model.infer(gs,prior,cal);times.append((time.perf_counter()-t)*1000);diff.append(abs(pred['height_m']-expected[j]))
            assert max(diff)<1e-7,(name,max(diff))
            result.append(dict(condition=condition,width_px=width,method=name,frames=len(times),prior_s=prior_s,
                 p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=max(times),cached_difference_m=max(diff),
                 capture_transfer_display_included=False,restoration_included=True,cpu_threads=1))
    return result

def fmt(x):return '—' if x is None else f'{x:.2f}'

def table(items):
    lines=['| 场景 | 帧数 | 原MAE mm | 新MAE mm | 原P95 mm | 新P95 mm | 原通过率 | 新通过率 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,item in items:
        b=item['before'];a=item['after']
        lines.append(f"| {label} | {a['frames']} | {fmt(b['mae_mm'])} | {fmt(a['mae_mm'])} | {fmt(b['p95_mm'])} | {fmt(a['p95_mm'])} | {fmt(b['pass_pct'])}% | {fmt(a['pass_pct'])}% |")
    return '\n'.join(lines)

def audit():
    hashes=json.loads((OUT/'frozen-source.json').read_text());unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    assert unchanged,'frozen_source_changed'
    policy=json.loads((OUT/'frozen-policy.json').read_text());raw=json.loads((OUT/'test-raw.json').read_text())['records']
    before=[r for r in raw if r['stage']==run.BASE];after=[r for r in raw if r['stage']==policy['selection'][r['condition']]]
    assert len(before)==len(after)==8016
    semi_b=[r for r in before if r['condition']=='translucent_mild'];semi_a=[r for r in after if r['condition']=='translucent_mild']
    result=dict(policy=policy,source_unchanged=True,all=compare(before,after),semi=compare(semi_b,semi_a,True),material={},
                development_runtime=json.loads((OUT/'development-runtime.json').read_text()),
                development_ranking=json.loads((OUT/'development-ranking.json').read_text()))
    result['shared_noise_development']=json.loads((OUT/'shared-noise-development.json').read_text())
    for mat in sorted({r['material_regime'] for r in after}):
        result['material'][mat]=compare([r for r in before if r['material_regime']==mat],[r for r in after if r['material_regime']==mat])
    auditraw=json.loads((OUT/'sr-audit-raw.json').read_text())['records']
    result['sr_audit']={name:{k:stats(v) for k,v in subsets([r for r in auditraw if r['stage']==name]).items()}
                        for name in sorted({r['stage'] for r in auditraw})}
    result['runtime']=runtime(policy,after)
    result['worst_after']=sorted(semi_a,key=lambda r:r['absolute_error_m'] if r['absolute_error_m'] is not None else np.inf,reverse=True)[:10]
    run.dump(OUT/'final-analysis.json',result)
    names=dict(all='全部加密样本',zero='空槽',near='大于0至100毫米',low='大于0且小于1米',mid='2至3米不含3米',rest='1至9米',uniform='0至9米均匀子集')
    chosen=policy['selection']['translucent_mild'];s=result['semi']
    lines=['# 半透明空槽与浅液位超分辨率和配准优化','',
        '2026年10月12日。固定顶视、底纹仍可观察的单一均匀液体仿真；误差对象为相对容器底面的实际液深。','',
        f"开发阶段冻结半透明方案为{NAMES[chosen]}，透明仍沿用优化前链路。新独立测试中，半透明空槽MAE由{fmt(s['zero']['before']['mae_mm'])}变为{fmt(s['zero']['after']['mae_mm'])} mm，0至100 mm非零浅液位由{fmt(s['near']['before']['mae_mm'])}变为{fmt(s['near']['after']['mae_mm'])} mm。以下同时列出误差区间和长尾，不以通过率作为唯一判断。",'',
        '## 实验预算与隔离','',
        '保持5组先验：空槽加15%、40%、65%、90%容量的4组已知液深，每组包含左右目。先验标签噪声沿用0.5 mm加0.1%液深，空槽已知标签为0。测试空槽为独立噪声图像，不重用标定空图。实际液深0至9 m，相机到底面不超过10 m，空槽底纹宽128或160原生像素，不扩展小容器。',
        '新增开发24种配置，初筛9种方法后，在开发阶段补充超分与对称配准的组合和区域大小消融，共12种。选型只使用开发集，并要求两种开发成像尺寸的计算P95不超过450 ms，为500 ms端到端目标预留余量；小样本计算计时不是端到端认证。源码、权重和策略冻结后，生成48种新测试配置、8016组独立测量图像。另按事前声明的纹理簇与H小于1 m子集，对两倍双三次插值、官方SwinIR及各自对称配准组合做独立消融。子集的真值仅用于划定评估人群，不参与推理或方法路由。',
        '不强制归零，不裁剪负值或超容量输出，不做强制覆盖率或置信度筛选。所有帧计入分布和通过率分母；有限误差统计另报告无法求解数。全部研究输出qualified=false，生产默认配置未修改。','',
        '通过标准沿用绝对误差不超过max(5 mm,2%实际液深)。MAE及分位数按有限数值输出计算，无法求解的帧仍计入通过率分母并单列。本轮主测试与超分消融均无无法求解帧。空槽不定义相对误差；极浅液位的相对误差需与毫米误差同时解读。','',
        '## 为什么增加像素不一定增加可信观测','',
        '在近轴折射近似下，底纹尺度对液深的敏感度约为(1减1除以折射率)除以相机到底面距离。约10 m距离、折射率约1.3至1.45时，5 mm液深变化在距图像中心60像素处仅对应约0.007至0.009像素位移。毫米精度依赖原始信号的亚像素配准，不能靠重复插值像素获得独立信息。',
        '当前少帧先验折射校正链路用的是底纹图像变化到实际液深的响应曲线，不是将相机失效液面点云补出真实回波。本轮增加的是图像匹配采样；即使最终液深误差下降，也不能解释为恢复了更多真实液面三维点。',
        '原配准使用底纹中央约60%宽高、最多约650点。加密采样可减小数值采样误差，但仍相关；扩大可用半径有助于提高尺度变化的可观测性。空图标定噪声被多次测量共同引用，各先验又共享当前图像噪声，不能把十个配准当成十次独立采集。',
        f"开发集16种半透明配置、每种12次独立空槽测量中，同一目不同先验的重复误差平均相关系数为{result['shared_noise_development']['average']['same_channel_reference_error_correlation_mean']:.3f}，左右目之间为{result['shared_noise_development']['average']['cross_channel_error_correlation_mean']:.3f}。相关系数按配置中心化，固定参考偏置不计入此相关；该结果支持同一当前图像的共同误差不可忽视，但不等于真实硬件噪声因果辨识。",
        '仿真对半透明类别在空槽时也沿用较高噪声，并保留该类别的轻微成像模糊；空槽本身没有液体，不能将类别差异解释成实际空槽液体散射。新旧方法使用完全相同图像，因此比较仍成立，但现场归因需真实相机数据。','',
        '## 对照方法','',
        '双三次两倍放大是插值对照，不称为学习式超分。SwinIR使用官方DIV2K轻量两倍预训练模型，重复灰度为三通道，未训练或微调；该自然图像模型和本项目IR样式仿真存在域差异。放大坐标按半像素中心关系还原到原相机坐标，再使用原焦距计算；低通核保持原生像素尺度，新增像素不按独立样本提高置信度。',
        '原生加密采样保持原相机分辨率。对称配准将两张含噪图像映射到中间坐标系、同时插值并归一化光度；对称方案使用约72%宽高、最多2000点，并比较650点、不同低通宽度和五阶样条。校准与融合均沿用上一轮逐参考单调曲线和软鲁棒融合，以隔离配准改动。','',
        '## 开发集方案比较','',
        '| 方法 | 空槽MAE mm | 0至100 mm MAE mm | 100 mm至1 m MAE mm | 1至9 m MAE mm | 计算延时合格 |',
        '|---|---:|---:|---:|---:|---|']
    for item in result['development_ranking']:
        b=item['buckets']
        lines.append(f"| {NAMES[item['name']]} | {fmt(b['zero']['mae_mm'])} | {fmt(b['near']['mae_mm'])} | {fmt(b['shallow']['mae_mm'])} | {fmt(b['rest']['mae_mm'])} | {'是' if item['latency_eligible'] else '否'} |")
    lines += ['', '开发结果用于选型，不能当成新测试结果。五阶样条与三阶的开发误差几乎相同，但五阶配准计算更慢，因而需要与精度同时权衡。','',
        '## 新测试半透明结果','',table([(names[k],v) for k,v in s.items()]),'']
    for mat,item in result['material'].items():
        lines += ['## '+{'transparent':'透明对照','semi_low_extinction':'半透明低衰减','semi_medium_extinction':'半透明中衰减'}[mat],'',table([(names[k],v) for k,v in item.items()]),'']
    lines += ['## 半透明毫米误差分布','',
        '区间左开右闭，首档包含0。以下各组分别使用自身全部帧作为分母；低液位组包含0至100 mm子集，不能相加。','']
    for k in ('zero','near','low','uniform'):
        item=s[k];lines += ['### '+names[k],'','| 绝对误差 | 原占比 | 新占比 |','|---|---:|---:|']
        for label,b,a in zip(['不超过2 mm','2至5 mm','5至10 mm','10至20 mm','20至50 mm','50至100 mm','超过100 mm'],item['before']['absolute_bins'],item['after']['absolute_bins']):
            lines.append(f"| {label} | {b['pct']:.2f}% | {a['pct']:.2f}% |")
        lines += ['',f"原/新无法求解帧数：{item['before']['missing']}/{item['after']['missing']}。原/新P99：{fmt(item['before']['p99_mm'])}/{fmt(item['after']['p99_mm'])} mm；最大误差：{fmt(item['before']['max_mm'])}/{fmt(item['after']['max_mm'])} mm。",'']
    ab=result['sr_audit'];b=ab['baseline_v32'];sr=ab['swinir_x2'];bc=ab['midpoint_bicubic2'];sc=ab['midpoint_swinir2']
    lines += ['## 独立浅液位超分消融','',
        f"该配对消融包含{b['all']['frames']}组测量，来自8个纹理簇、16种半透明配置：空槽{b['zero']['frames']}组，非零且小于1 m为{b['low']['frames']}组，其中不超过100 mm为{b['near']['frames']}组。它是主测试的预声明子集，不能将这里的MAE直接与上方全部5344组半透明结果跨人群比较。",'',
        '| 方法 | 空槽MAE mm | 非零0至100 mm MAE mm | 非零0至1 m MAE mm | 非零0至1 m P95 mm |','|---|---:|---:|---:|---:|']
    for name,item in result['sr_audit'].items():
        lines.append(f"| {NAMES[name]} | {fmt(item['zero']['mae_mm'])} | {fmt(item['near']['mae_mm'])} | {fmt(item['low']['mae_mm'])} | {fmt(item['low']['p95_mm'])} |")
    lines += ['',
        f"独立子集上，单纯SwinIR放大的空槽MAE为{fmt(sr['zero']['mae_mm'])} mm，对照为{fmt(b['zero']['mae_mm'])} mm；非零0至1 m MAE分别为{fmt(sr['low']['mae_mm'])}与{fmt(b['low']['mae_mm'])} mm。加入对称配准后，普通双三次与SwinIR的非零0至1 m MAE分别为{fmt(bc['low']['mae_mm'])}与{fmt(sc['low']['mae_mm'])} mm。需看这些差值和延时，而不能仅比较图像视觉质量。",
        '以上方法在同一预声明子集上配对评估；消融不重新选型。一次超分模型失效不代表所有超分算法无效，图像观感改善也不等于计量精度改善。','',
        '## 统计不确定性','']
    for k in ('zero','near','low','uniform'):
        x=s[k]['paired_cluster_ci'];m=x['mae_improvement_mm_ci95'];p=x['pass_improvement_pp_ci95']
        lines.append(f"- {names[k]}：MAE改善95%区间[{m[0]:.2f}, {m[1]:.2f}] mm；通过率改善区间[{p[0]:.2f}, {p[1]:.2f}]个百分点。")
    lines += ['', '按16个纹理种子簇进行5000次配对bootstrap，同纹理不同材料和不同液深不视为独立场景。区间跨0时不能宣称稳健改善。','',
        '## 单帧计算延时','']
    for x in result['runtime']:
        lines.append(f"- {x['condition']}，{x['width_px']}px，{NAMES[x['method']]}：6帧无配准缓存，P50/P95为{x['p50_ms']:.1f}/{x['p95_ms']:.1f} ms，先验准备与拟合{x['prior_s']:.3f}s，缓存一致性最大差{x['cached_difference_m']:.3g}m。")
    lines += ['', '延时包括所选方法的图像处理与测量，不含相机曝光、传输和显示；小样本不能认证端到端500 ms。','',
        '## 实验接口与后续优化','',
        'src/liquid_depth/range_runtime_v33.py封装FivePriorImageMeter：输入原相机几何、左右目固定ROI、5组先验图像及已知液深、操作人员指定的透明类别和可选配准方法；初始化完成先验拟合，measure只接收当前左右目灰度图，返回height_m及诊断。它不记忆测试帧，不增加标定图，不读取测量真值，未接入生产GUI默认链路。相机移动后须重新标定，不据此宣称运动场景适配。',
        '超分和加密采样应分别看待。扩大有效尺度观测半径、对两张含噪图像对称处理，可针对当前零位残差；超分若改变纹理相位则可能引入偏差，需继续用毫米误差、P95、误差分档和相邻液深响应共同验证。任何后续去噪或项目专用超分训练，都应保持原始像素重投影一致性，并另用新的独立测试集评估。','',
        '本轮更适合保留原生图像对称配准作为半透明实验主方案：在独立浅液位子集中，加入SwinIR或双三次放大没有超过该方案，且计算更慢；单纯SwinIR放大仅有小幅平均收益。空槽与浅液位仍有毫米级残差，不能据此承诺极浅液位1%相对精度。后续应优先研究共同参考偏置、固定投影纹理与底纹解耦，以及保持几何一致性的专用去噪，而非直接扩大超分模型。','',
        '## 适用边界与参考','',
        '本轮未新增现场数据，也未训练项目专用超分网络；未解决底纹完全消失、多层玻璃、动态波面或相机运动。仿真是折射、散斑、模糊和噪声代理，不是光谱标定后的真实RGB-D数字孪生。',
        '[SwinIR官方实现](https://github.com/JingyunLiang/SwinIR)提供本轮轻量预训练模型与Apache 2.0许可；其训练域不是本项目传感器图像。',
        '[Interpolation artifacts in sub-pixel image registration](https://pubmed.ncbi.nlm.nih.gov/19116201/)讨论插值带来的配准局部极值与低通、高阶样条等对策，支持本轮对插值误差进行单独消融。',
        '[The Perception-Distortion Tradeoff](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)说明视觉感知质量与失真之间的权衡；本项目以液深误差而非视觉锐度选择方法。','',
        '## 复现位置','',f'数据：`{run.DATA}`。',f'冻结策略、逐帧误差、开发排名、超分消融、分布和运行时间：`{OUT}`。',
        '代码入口scripts/experiment_sr_v33.py；配准src/liquid_depth/range_registration_v33.py；分析scripts/audit_sr_v33.py。历史算法与数据未改。']
    formatted=[]
    for line in lines:
        if line and formatted and formatted[-1] and not line.startswith(('|','-','#')) and not formatted[-1].startswith(('|','-','#')):
            formatted.append('')
        formatted.append(line)
    text='\n'.join(formatted)+'\n';(OUT/'report.md').write_text(text,encoding='utf-8')
    (run.REPO/'docs/transparent_sr_v33.md').write_text(text,encoding='utf-8')
    run.progress('complete_with_analysis',source_unchanged=True)
    print(json.dumps(dict(selection=policy['selection'],semi=s,runtime=result['runtime'],sr_audit=result['sr_audit']),ensure_ascii=False))

if __name__=='__main__':audit()
