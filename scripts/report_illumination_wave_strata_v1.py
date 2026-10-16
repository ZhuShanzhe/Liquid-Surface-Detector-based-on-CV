"""Deterministic Chinese report from measured results, never edits meter code."""
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from special_scene_protocol_v1 import SCENES,SENSORS,LEVELS,BAND_NAMES,tolerance

NAMES={'low_light':'低照','glare':'眩光','uneven_surface':'不平整液面',
       'tof':'ToF','structured_light':'结构光','all':'合并','mild':'低','moderate':'中','severe':'强'}


def f(x):return '—' if x is None else f'{x:.2f}'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])+'\n\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];docs=repo/'docs'
    summary=json.loads((a.out/'stratified-summary.json').read_text())
    rows=[json.loads(s) for s in (a.out/'records.jsonl').read_text().splitlines()]
    attempts=[json.loads(s) for s in (a.out/'capture-eligibility-attempts.jsonl').read_text().splitlines()]
    audit=json.loads((a.out/'input-audit.json').read_text())
    assert len(rows)==4752 and all(r['scope_included'] for r in rows)
    assert all(r['input_evidence']['capture_origin']['eligible'] for r in rows)
    assert audit['successful_calibrations']==132
    # Independent recount of each metric denominator and tolerance, including all crossed strata.
    checks=[]
    for scene in SCENES:
        for sensor in ('all',*SENSORS):
            for level in ('all',*LEVELS):
                for band in ('all',*BAND_NAMES):
                    q=[r for r in rows if r['scenario']==scene and (sensor=='all' or r['sensor']==sensor)
                       and (level=='all' or r['level']==level) and (band=='all' or r['band_name']==band)]
                    m=summary[f'{scene}|{sensor}|{level}|{band}']['liquid']
                    good=[r for r in q if r['liquid_depth_m'] is not None]
                    ok=sum(abs(r['liquid_depth_m']-r['truth']['liquid_height_m'])<=tolerance(scene,r['truth']['liquid_height_m']) for r in good)
                    assert m['frames']==len(q)
                    assert abs(m['all_frame_pass_pct']-100*ok/len(q))<1e-9
                    assert abs(sum(m['absolute_bins_all_frame_pct'])-m['numeric_coverage_pct'])<1e-9
                    if level!='all' and band!='all' and sensor!='all':assert len(q)==24
                    checks.append(f'{scene}|{sensor}|{level}|{band}')
    # Baseline source/config/document archive stays immutable; new evaluation files are additive.
    baseline=Path('/root/autodl-tmp/liquid-depth-artifacts/determined-all-scenes-v1-20261016')
    old=json.loads((baseline/'manifest.json').read_text())['source_sha256']
    changes=[p for p,h in old.items() if hashlib.sha256((repo/p).read_bytes()).hexdigest()!=h]
    assert not changes,changes
    independent=dict(cross_strata_recounted=len(checks),all_432_denominators_and_pass_counts_verified=True,
        fixed_baseline_source_files_verified=len(old),changed_baseline_files=changes,
        selected_frames=len(rows),selected_visual_calls=sum(r['visual_module_executed'] for r in rows))
    (a.out/'independent-recount-and-baseline-audit.json').write_text(json.dumps(independent,indent=2))

    def m(scene,sensor='all',level='all',band='all',kind='liquid'):
        return summary[f'{scene}|{sensor}|{level}|{band}'][kind]
    def results(q,kind='liquid'):
        return [q['frames'],f(q['mae_mm']),f(q['p95_mm']),f(q['max_mm']),f(q['all_frame_pass_pct']),f(q['numeric_coverage_pct']),f(q['absrel_pct'])]
    headers=['帧数','MAE mm','P95 mm','最大误差 mm','通过率 %','数值覆盖率 %','AbsRel %']
    # Capture quality belongs to a separate denominator from conditional measurement accuracy.
    def capture_table(scene):
        data=[]
        for level in LEVELS:
            q=[r for r in attempts if r['scenario']==scene and r['level']==level]
            for sensor in SENSORS:
                qualified=sum(r['evidence'][sensor]['eligible'] for r in q)
                data.append([NAMES[level],NAMES[sensor],len(q),qualified,len(q)-qualified,
                             f(100*qualified/len(q)),f(100*sum(r['paired_eligible'] for r in q)/len(q))])
        return table(['强度','相机','原始尝试帧','自身证据合格帧','自身不合格帧','自身合格率 %','成对采样成功率 %'],data)
    main_text='# 低照 眩光与不平整液面扩样评估\n\n'
    main_text+='当前确定版完成三类场景的独立仿真扩样。正式评分集共 4,752 帧，每类 1,584 帧；ToF 和结构光各占一半。先按输入证据判定采样是否有效，再对两种相机共同合格的采样计分，未按测量误差删除帧。算法、液深数值、视觉启用边界均未修改。结果用于限定仿真模型内的能力评估，不是工业计量认证或硬件验收。\n\n'
    main_text+='## 测试范围和评价口径\n\n'
    main_text+='66 个物理安装实例，132 个相机标定实例。每个液深层有 6 个物理实例，覆盖圆形、椭圆和近矩形容器以及近、中、远视角；每个相机、强度、液深组合为 24 帧，即 6 个实例各 4 次噪声采样。每档强度 528 帧，已大于此前普通场景 396 帧的整组测试规模。低照和眩光的低、中、强严格等额；不平整也按三档波幅及伴随退化强度等额采样。\n\n'
    main_text+='实际液深分层为：空槽、0.01–0.1、0.1–1、1–2、2–3、3–4、4–5、5–6、6–7、7–8、8–9 m。各层等额，不等于连续 0–9 m 上按长度均匀抽样。相机至底面不超过 10 m。液深较大时，相机至液面的距离受该约束而缩短，因此深液位成绩好不能解释为所有远距离测量都好。\n\n'
    main_text+='两种相机共用同一空槽深度参考和成对的操作员扰动。参考沿用既有密集空槽噪声，逐点标准差为 1 mm + 0.5 mm × 距离米值的平方，并有 4% 随机缺失；它是受控标定先验，不是两种设备各自原生单帧误差的实测等价物。轮廓输入模拟 0.3 像素点击扰动，独立距离锚点扰动标准差为 1 mm + 0.1% 距离。没有重新训练空槽图像识别器。\n\n'
    main_text+='低照、眩光液深通过容差为 max(20 mm, 2% 实际液深)；不平整液深改为 max(50 mm, 5% 实际平均液深)，最高高度也用 max(50 mm, 5% 实际最高高度)。液深和最高高度都相对容器内底面，波动液深真值为整个物理液面的面积平均高度，最高真值为波形几何峰值。不平整主试验不含漂浮物，不将其结果冒充漂浮物识别成绩。\n\n'
    main_text+='MAE、P95、最大误差基于有数值的有效测试帧；通过率分母是全部正式测试帧，若合格输入仍未输出也计失败。数值覆盖率是正式测试帧中输出数值的比例，不是采样成功率。相对误差只统计非零真值，空槽仍计入绝对误差和通过率。绝对误差分布分母为全部正式帧，相对误差分布分母为非零真值且有数值的帧。没有归零、裁剪结果或增加测量拒绝门槛。\n\n'
    main_text+='## 有效输入筛选与补采\n\n'
    main_text+='用户在首批全量诊断后要求排除证据不足帧。正式口径因此采用两级采集审核。第一级仅使用空槽标定与当前原始深度，在标定容器内部要求至少 64 个可用回波，覆盖 4×4 物理底面投影分区中的至少 6 区，两个水平方向的跨度均至少覆盖容器尺寸的 25%；该级不读取液深真值、预测液深、误差或语义标签。\n\n'
    main_text+='第二级只审核仿真采集是否真的提供了可重建的信息：利用离线几何类别和回波生成来源，要求至少 64 个直接液面来源的有效回波，分布覆盖可见液面 4×4 分区中的至少 6 区且两方向跨度均至少 25%；空槽控制改为底面来源。错误层回波、混合边缘回波不冒充直接液面证据，但噪声很大或共同偏置很大的直接液面回波仍保留，不按数值误差筛选。这些是工程测试输入条件，不是行业标准。来源标签只用于采集有效性审核，绝不进入测量算法；实地没有这类来源真值，不能把本测试采集审核声称为已实现的在线可观测性判定。\n\n'
    main_text+='每个物理实例和强度条件按固定种子依次采样，取前 4 组两种相机都满足上述条件的回波；最多尝试 64 次，未达到配额时必须报错，不能降低强度或阈值。排除帧不进入正式精度分母；自身证据合格但另一相机不合格的帧也不进入成对比较，单独记录。有充足物理液面来源回波、但算法仍误解液面、选错候选或存在测距偏置的帧必须计分。\n\n'
    cap=json.loads((a.out/'capture-eligibility-summary.json').read_text())
    main_text+=f"共尝试 {cap['attempted_pairs']} 组成对采样，选入 {cap['selected_pairs']} 组，排除 {cap['rejected_pairs']} 组；成对有效采样率 {f(cap['paired_input_acquisition_success_pct'])}%。这是仿真采样可用性，与筛选后的条件精度不同，也不等于实地采样成功率。筛选仅控制本次测试数据，不改正式应用的测量或拒绝链路。原始未筛选批次保留为诊断档案，不再作为本轮正式成绩。\n\n"
    for scene in SCENES:
        main_text+='### '+NAMES[scene]+'采样可用性\n\n'+capture_table(scene)
    main_text+='## 各场景总体结果\n\n'
    main_text+=table(['场景','相机',*headers],[[NAMES[scene],NAMES[sensor],*results(m(scene,sensor))] for scene in SCENES for sensor in ('all',*SENSORS)])
    main_text+='“合并”是同一场景下两种相机、三档强度、全部液深层的等额汇总，不将不同场景容差混成一个所谓总体准确率。AbsRel 容易被厘米级浅液位的分母放大，需与绝对误差分布共同解读。\n\n'
    main_text+='## 按强度和相机分层\n\n'
    main_text+=table(['场景','相机','强度',*headers],[[NAMES[scene],NAMES[sensor],NAMES[level],*results(m(scene,sensor,level))] for scene in SCENES for sensor in SENSORS for level in LEVELS])
    main_text+='## 不平整液面的最高高度\n\n'
    main_text+=table(['相机',*headers],[[NAMES[sensor],*results(m('uneven_surface',sensor,kind='highest'))] for sensor in ('all',*SENSORS)])
    main_text+=f"同一批不平整液深预测，以旧 max(20 mm, 2%) 容差重计通过率为 {f(m('uneven_surface')['reference_20mm_2pct_all_frame_pass_pct'])}%，新容差为 {f(m('uneven_surface')['all_frame_pass_pct'])}%。差别来自评分要求变化，不是算法被优化。最高高度采用几何链路，未开启 V8；V8 仍只允许用于明确选择的漂浮物场景最高高度。\n\n"
    main_text+='## 分层误差分布\n\n'
    for scene in SCENES:
        name=f'{scene}_expanded_strata_20261016.md'
        text='# '+NAMES[scene]+'扩样分层结果\n\n'
        text+='有效输入均按固定证据条件选入；每个相机、强度、液深单元 24 帧，合并强度后每个相机、液深单元 72 帧。各单元仅有 6 个物理实例，4 次噪声重复不是 24 个独立容器。误差真值、筛选和传感器适用边界见[主报告](special_scene_expanded_evaluation_20261016.md)。\n\n'
        kinds=('liquid','highest') if scene=='uneven_surface' else ('liquid',)
        for kind in kinds:
            title='液深' if kind=='liquid' else '最高高度'
            text+='## '+title+'按实际液深与相机分层\n\n'
            text+=table(['相机','实际液深层',*headers],[[NAMES[sensor],band,*results(m(scene,sensor,band=band,kind=kind))] for sensor in ('all',*SENSORS) for band in BAND_NAMES])
            for sensor in SENSORS:
                for level in LEVELS:
                    text+='## '+NAMES[sensor]+NAMES[level]+'强度'+title+'误差\n\n'
                    text+=table(['实际液深层',*headers],[[band,*results(m(scene,sensor,level,band,kind))] for band in BAND_NAMES])
                    text+='绝对误差区间为 mm，占全部正式测试帧的百分比。\n\n'
                    text+=table(['实际液深层','0至5','5至20','20至50','50至100','不小于100'],[[band,*map(f,m(scene,sensor,level,band,kind)['absolute_bins_all_frame_pct'])] for band in BAND_NAMES])
                    text+='相对误差区间为 %，占非零真值且有数值帧的百分比；空槽记为不适用，不记为零误差。区间左闭右开。\n\n'
                    text+=table(['实际液深层','0至1','1至2','2至5','5至10','不小于10'],[[band,*(list(map(f,m(scene,sensor,level,band,kind)['relative_bins_nonzero_numeric_pct'])) or ['—']*5)] for band in BAND_NAMES])
        (docs/name).write_text(text,encoding='utf-8')
        (a.out/name).write_text(text,encoding='utf-8')
        main_text+=f'- [{NAMES[scene]}全部液深和交叉分层]({name})：包含每一液深层、两种相机、三档强度的绝对和相对误差区间占比。\n'
    main_text+='\n完整机器可读结果为 `stratified-summary.json`，包括 95% 安装实例聚类重采样置信区间、偏差、P99 和未输出原因。\n\n'
    main_text+='## 远距离浅液位单列\n\n'
    far_rows=[]
    for scene in SCENES:
        for sensor in SENSORS:
            k=f'{scene}|{sensor}|all|subset=far_shallow_wet'
            far_rows.append([NAMES[scene],NAMES[sensor],*results(summary[k]['liquid'])])
    main_text+=table(['场景','相机',*headers],far_rows)
    main_text+='远距浅液位定义为相机至底面至少 6 m 且实际平均液深大于 0、不超过 1 m。它仍在正式总体分母中，没有因难测而排除。结构光小基线视差代理在远距离时单点距离噪声显著增大，浅液位又是两个大距离的差；此类误差继续留作下一阶段统一优化，本轮不改算法。\n\n'
    main_text+='## 误差尾部与输入证据\n\n'
    worst=[]
    for scene in SCENES:
        pool=sorted((r for r in rows if r['scenario']==scene and r['liquid_depth_m'] is not None),key=lambda r:abs(r['liquid_depth_m']-r['truth']['liquid_height_m']),reverse=True)[:3]
        for r in pool:
            e=r['input_evidence']
            worst.append([NAMES[scene],NAMES[r['sensor']],NAMES[r['level']],r['physical_id'],r['repeat'],
                f(r['truth']['liquid_height_m']),f(r['liquid_depth_m']),f(1000*abs(r['liquid_depth_m']-r['truth']['liquid_height_m'])),e['interior_returns'],e['occupied_4x4_tiles']])
    main_text+=table(['场景','相机','强度','实例','采样','实际液深 m','估计 m','绝对误差 mm','输入内部回波','覆盖区数'],worst)
    main_text+='这些帧已满足输入支撑与物理回波来源审核，不能因误差大再次剔除。较多直接液面回波并不等于无系统偏置：视差到距离的非线性、回波量化、范围截断和候选解释可使点云整体偏移；眩光下空间集中缺失还会降低有效约束。上述归因分别来自传感器模型结构和候选日志，并非已通过单因素消融证明的唯一原因。\n\n'
    cal={c['case']:c['audit'] for c in audit['calibrations']}
    diagnostic={}
    for sensor in SENSORS:
        q=[r for r in rows if r['sensor']==sensor and r['liquid_depth_m'] is not None]
        err=np.array([r['liquid_depth_m']-r['truth']['liquid_height_m'] for r in q])
        anchor=np.array([cal[r['case']]['bottom_distance_error_mm']/1000 for r in q])
        diagnostic[sensor]={'error_vs_bottom_calibration_error_pearson':float(np.corrcoef(err,anchor)[0,1]),
            'liquid_numeric_frames':len(q),'insufficient_support_flag_frames':sum('insufficient_current_returns' in r['diagnostics'].get('flags',[]) for r in q)}
    (a.out/'calibration-error-correlation.json').write_text(json.dumps(diagnostic,indent=2))
    main_text+=f"ToF 液深误差与底面标定距离误差的相关系数为 {f(diagnostic['tof']['error_vs_bottom_calibration_error_pearson'])}，结构光为 {f(diagnostic['structured_light']['error_vs_bottom_calibration_error_pearson'])}。相关不证明全部误差来自标定，但可防止把受共同底面误差限制的毫米级成绩全归功于液面模型。\n\n"
    main_text+='不平整最高高度还受几何极值估计的尾部放大、波峰回波稀疏、平滑或选点遗漏影响；液深均值通过并不保证真实波峰被准确找到。既有回波错误类型数组的标签可能被后续范围无效标签覆盖，因此不以标签值 4 的数量冒充真实眩光面积；亮度与 RGB 饱和率、实际有效回波数分别记录。\n\n'
    main_text+='## 相机类型建议\n\n'
    main_text+=table(['场景','仿真内结论','部署建议与边界'],[
        ['低照','本批 ToF 液深更稳定，结构光主要受远距离与联合回波退化影响','两者都是主动红外技术，RGB 暗不必然造成深度坏。暗室且红外回波充足时两者均可；长距离优先验证长距 ToF，近中距结构光仍可用。'],
        ['眩光','结构光强档采样可用性和尾部风险更差；ToF 代理更好','不能据此认定 ToF 抗镜面可靠。真实 ToF 也有多径、饱和和反射偏转；先控制光路和曝光，再按目标液体实测选择。'],
        ['不平整','平均液深较稳，最高高度尾部仍需注意','长距可优先验证 ToF；近中距且回波空间覆盖充分时可用结构光。快速波动还需检查曝光、运动与同步，本批静态波形未验证。']])
    main_text+='结构光现用 35 mm 基线和 0.26 px 视差噪声代理，480×270 计算分辨率；ToF 是距离噪声和缺失代理，本场景未完整模拟调制波形、多径与温漂。两者不是同档商用硬件的物理等价模型，ToF 的样本 100% 通过不能作为现实零失败承诺。低照、眩光的档位同时改变 RGB 与深度退化，不是按 lux 或真实红外功率标定的照明档位。\n\n'
    main_text+='Orbbec 对散斑结构光与 iToF 的官方说明区分了前者的相关匹配与红外纹理可见性限制、后者的多径风险；据此不能把 RGB 亮度直接转换成统一的深度缺失率。[Orbbec 技术说明](https://www.orbbec.com/blog/structured-light-vs-itof-depth-cameras/)\n\n'
    main_text+='Astra 2 官方量程为 0.6–8 m，不能用本次泛化至 10 m 的结构光代理结果宣称该产品支持 10 m；实际采购必须核对相机至液面及空槽底面的工作距离。[Astra 2 官方参数](https://www.orbbec.com/products/structured-light-camera/astra-2/)\n\n'
    main_text+='Basler blaze 101 官方长距模式标为 0.3–10 m，但典型 ±5 mm 精度的列示条件是 0.5–5.5 m，并非全程 10 m 的承诺。[blaze 101 官方参数](https://docs.baslerweb.com/blaze-101) 镜面、透明材质和凹容器的多次反射也会影响 ToF，不能靠本批简化仿真排除这些风险。[Basler 测量影响因素](https://docs.baslerweb.com/influencing-factors)\n\n'
    main_text+='## 统计可靠性与复现\n\n'
    main_text+='每层 6 个物理实例，重复噪声与同容器不同条件相互相关。95% 区间按实际液深层内重采样物理安装实例计算，保持配对相机和重复帧在一起，共 1,000 次。它反映当前有限仿真样本间的变异，不覆盖模型失配、未采到的失效模式或真实硬件偏差；观察到零失败时，退化为 100% 的重采样区间也不意味着总体失败率为零。\n\n'
    main_text+=f"{len(checks)} 个全交叉和合并单元已独立复核帧数、通过数和分布分母；原确定版 {len(old)} 份源码、配置和文档哈希未变。新增输入审核与统计回归测试 7 项通过，全项目 726 项测试通过，14 项警告保留于日志。测量时关闭视觉，未训练模型。耗时表位于机器可读分层结果中，仅是服务器算法调用时间，含多进程资源竞争，不含采集、USB、网络或完整终端延时。\n\n"
    main_text+=f"服务器数据目录：`{a.root}`。正式证据目录：`{a.out}`。逐帧结果为 `records.jsonl`；标定为 `calibrations`；输入筛选与补采为 `capture-eligibility-attempts.jsonl`；固定规则和源码哈希为 `protocol.lock.json`。首次未筛选诊断位于同级 `evaluation-final`，仅作输入可用性与尾部调查，不混入正式分数。\n\n"
    main_text+='复现入口为 `scripts/run_special_scene_strata_v1.py --root <数据目录> --out <新的结果目录> --workers 6`，使用项目 Python 环境并配置 `PYTHONPATH=src`。没有连接 USB 相机。后续正式测试以本轮有效输入均衡集及新容差为准，旧小样本报告保留为历史，不作优化前后配对比较。\n'
    (docs/'special_scene_expanded_evaluation_20261016.md').write_text(main_text,encoding='utf-8')
    (a.out/'report.md').write_text(main_text,encoding='utf-8')
    registry={'baseline_version':'determined_all_scenes_v1_v8_peak_20261016','algorithms_changed':False,
        'report':'docs/special_scene_expanded_evaluation_20261016.md','new_evidence_root':str(a.out),
        'supersedes_small_cohort_scores_for':list(SCENES),'other_scenes_retained_from_baseline':True,
        'input_scope':'paired input_evidence_v1 qualified balanced sampling; raw acquisition availability separate',
        'additional_offline_return_origin_capture_audit':True,'offline_labels_used_for_meter_inference':False,
        'uneven_surface_liquid_tolerance':'max(50mm,5% actual area-mean liquid depth)',
        'low_light_and_glare_liquid_tolerance':'max(20mm,2% actual liquid depth)',
        'highest_tolerance':'max(50mm,5% actual physical highest height)',
        'field_metrology_qualified':False}
    (repo/'configs/current_scene_evaluation_v1.json').write_text(json.dumps(registry,indent=2))
    print(json.dumps(independent),flush=True)
    print('REPORTS_COMPLETE',flush=True)


if __name__=='__main__':main()
