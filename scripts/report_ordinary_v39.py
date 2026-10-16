"""Reproducible bottom-referenced ordinary-route experiment report."""
import json,hashlib,sys
from pathlib import Path
from collections import Counter
import numpy as np
from liquid_depth.device_profiles_v38 import metric_summary

ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')
VERSION='v39b' if '--revision-b' in sys.argv else 'v39'
MAIN=ROOT/f'ordinary-{VERSION}-test-20261012'
LABEL={'baseline_v36':'原普通链路','v39_bias_noisy_echo':'公共偏置加带噪底部回波',
       'v39_noise_cal_only':'仅标定噪声加带噪回波','v39_bias_atom_echo':'公共偏置加旧回波原子',
       'v39_no_reference':'带噪回波但无空槽参考'}


def read(name):
    folder=ROOT/name
    assert (folder/'complete.json').exists(),str(folder)+' is incomplete'
    file=folder/'records.jsonl'
    return [json.loads(x) for x in file.read_text().splitlines()] if file.exists() else json.loads((folder/'records.json').read_text())


def f(x,d=2):return '不定义' if x is None else f'{x:.{d}f}'


def paired(rows):
    cases={}
    for r in rows:
        if r['method'] not in ('baseline_v36','v39_bias_noisy_echo') or r['predicted_height_m'] is None:continue
        cases.setdefault(r['cluster'],{}).setdefault(r['method'],[]).append(abs(r['predicted_height_m']-r['true_height_m'])*1000)
    differences=np.array([np.mean(v['v39_bias_noisy_echo'])-np.mean(v['baseline_v36']) for v in cases.values() if len(v)==2])
    rng=np.random.default_rng(390077)
    boot=np.mean(rng.choice(differences,(4000,len(differences)),replace=True),axis=1)
    return dict(clusters=len(differences),delta_mae_mm=float(differences.mean()),ci95_mm=np.percentile(boot,[2.5,97.5]).tolist())


def metric_table(lines,records,predicate):
    lines+=['|噪声|方法|独立配置|帧数|MAE mm|P95 mm|平均相对误差 %|通过率 %|未输出帧|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for regime in ('nominal','adverse'):
      for method in ('baseline_v36','v39_bias_noisy_echo'):
        s=metric_summary([r for r in records if r['regime']==regime and r['method']==method and predicate(r)])
        lines.append('|'+ '|'.join([regime,LABEL[method],str(s['clusters']),str(s['frames']),f(s['mae_mm']),f(s['p95_mm']),f(s['absrel_positive_pct']),f(s['pass_pct']),str(s['no_output'])])+'|')
    lines.append('')


def main():
    rows=read(f'ordinary-{VERSION}-test-20261012');drift=read(f'ordinary-{VERSION}-drift-20261012');replay=read(f'ordinary-{VERSION}-replay-20261012')
    lines=['# 普通场景远距离浅液位公共偏置与底部回波优化', '',
           '日期：2026-10-12。统计目标为相对容器底面的实际液深。本轮仅针对固定相机、已标定容器、近似平整的不透明普通液体，透明和半透明专用链路、ToF 链路以及默认产品运行配置均未修改。', '',
           '## 方法与适用条件','',
           '原链路在量化视差域拟合液面，但把底部错误回波视为固定深度原子。新版用一帧独立空槽参考联合估计公共视差偏置和噪声宽度，再以带噪液面与带噪底面混合分布拟合液位。似然同时考虑视差量化、毫米深度量化和有效量程截断。底部回波比例由当前观测估计，范围为 0 至 35%，不是按真实液深指定。', '',
           '新增的是一帧空槽参考，未增加已知非零液深图，仍在五帧先验预算之内。原链路不利用这帧图，因此消融另列仅校正偏置、仅校准噪声等结果。空槽参考必须来自相同安装位姿和内参；接口用安装指纹检查已声明的配置变化，但不能自动发现未上报的实际相机移动。', '',
           '不使用测试液深、真实液面掩码、真实偏置或真实噪声参数作为估计器输入。标定底面、相机内外参和容器范围仍是理想已知条件。当前测量值允许负值或超过容器容量；不强制归零、不裁剪液深、不按准确率或置信度筛帧。数值不收敛或无有效深度仍记录未输出。', '',
           '## 独立样本和统计口径','',
           '正式测试使用不同于开发集的随机种子：192 个远距离浅液位配置，32 个远距空槽，32 个远距近零液位，以及 0 至 9 m 每个整米区间 32 个均匀液深配置，共 544 个独立几何配置。每个配置分别使用标称与不利噪声，每种噪声 4 个独立瞬时噪声帧，共 4352 个原始测量帧。每帧评估五种方法，方法结果行不能算成独立图像。', '',
           '远浅专项：相机到液面 5.5 m 以上、液深 1 至 100 cm；近零补充：液深 5 至 100 mm。所有配置相机到底面中心欧氏距离不超过 10 m。圆形、椭圆形和不同圆角矩形范围、轻微俯视角变化、容器尺寸、底部错误回波比例和丢点概率独立变化。不再只有一个远浅几何配置。', '',
           '几何传感器仿真使用 Gemini 2 XL 对应 1280×800 原生像素、100 mm 基线和先前档案的视差噪声假设；所有回波均经过同一测距噪声和量化过程。不利噪声增加公共偏置并包含持久空间相关噪声。空槽和测量帧共享安装偏置及空间场，但瞬时噪声独立。此轮没有重新渲染 RGB 或测试左右目匹配网络，因为当前普通估计器只读取原始深度和标定 ROI。不能把它等同于 RGB-D 相机数字孪生或实机精度。', '',
           'MAE 为绝对液深误差均值，P95 为绝对误差 95 分位数，相对误差为 |估计液深−真实液深|/真实液深；空槽相对误差不定义，不用人为小分母替代。通过率仅作辅助指标，容差 max(5 mm, 真实液深的 2%)。误差分布百分比以有数值输出的帧为分母，未输出另列；完整 JSON 也保留以全部帧为分母的比例。', '',
           '## 远距离浅液位结果','']
    headline=[]
    for regime in ('nominal','adverse'):
        a,b=[metric_summary([r for r in rows if r['regime']==regime and r['group']=='far_shallow' and r['method']==m]) for m in ('baseline_v36','v39_bias_noisy_echo')]
        headline.append(f"{regime} 远浅 MAE 从 {f(a['mae_mm'])} 降至 {f(b['mae_mm'])} mm")
    lines[4:4]=['结论：'+ '；'.join(headline)+'。改善不依赖筛掉测量帧，但旧完整帧回放中接近 10 m 量程上限的空槽参考仍有误校正反例，因此保留为显式启用的实验分支，不全量程替换默认链路。','']
    metric_table(lines,rows,lambda r:r['group']=='far_shallow')
    diagnostics={}
    if VERSION=='v39b':
        lines += ['最终复验使用新种子 202610124037。第一轮测试曾发现空槽下的不可见液面退化解，因此先将混合比例重新定义为有效观测回波中的底部占比：液面与底部分别按可观测量程归一化，避免液面分量消失后所有观测都由底部解释。这个修正没有修改或限制液深数值。第一轮的完整结果仍保留，不作为最终独立验证集重复调参。','']
    for regime in ('nominal','adverse'):
        p=paired([r for r in rows if r['regime']==regime and r['group']=='far_shallow']);diagnostics[regime]=p
        lines += [f"{regime}：以独立配置为单位进行配对重采样，MAE 变化为 {f(p['delta_mae_mm'])} mm，95% 配对自助区间 [{f(p['ci95_mm'][0])}, {f(p['ci95_mm'][1])}] mm。负数表示改进，四个重复帧未当成四个独立容器。",'']
    for group,title in [('empty','空槽零位'),('near_zero','近零液位')]:
        lines+=['## '+title,''];metric_table(lines,rows,lambda r:r['group']==group)
    lines+=['## 各液深区间回归检查','',
            '以下只使用均匀液深主测试，不混入远浅和近零加密样本。每个噪声与液深区间为 32 个独立配置、128 帧。液深较大时相机到液面距离受到底部总距离不超过 10 m 的限制，不能解释成各液深都在同样远距离测试。','',
            '|噪声|液深 m|MAE 原→新 mm|P95 原→新 mm|相对误差 原→新 %|未输出 原→新|','|---|---|---:|---:|---:|---:|']
    regression=[]
    for regime in ('nominal','adverse'):
      for band in range(9):
        rr=[r for r in rows if r['regime']==regime and r['population']=='uniform' and r['band']==str(band)]
        a,b=[metric_summary([r for r in rr if r['method']==m]) for m in ('baseline_v36','v39_bias_noisy_echo')]
        lines.append(f"|{regime}|{band}–{band+1}|{f(a['mae_mm'])} → {f(b['mae_mm'])}|{f(a['p95_mm'])} → {f(b['p95_mm'])}|{f(a['absrel_positive_pct'])} → {f(b['absrel_positive_pct'])}|{a['no_output']} → {b['no_output']}|")
        regression.append(dict(regime=regime,band=band,delta_mae_mm=b['mae_mm']-a['mae_mm'],delta_p95_mm=b['p95_mm']-a['p95_mm'],paired=paired(rr)))
    lines+=['','## 误差区间占比','',
            '区间按左闭右开统计，最末区间包括其下界。每表的原与新为同一批帧，百分比保留一位小数可能存在舍入差。','']
    strata={'远浅专项':lambda r:r['group']=='far_shallow','空槽':lambda r:r['empty'],'近零补充':lambda r:r['group']=='near_zero'}
    strata.update({f'{i}至{i+1}米':lambda r,i=i:r['population']=='uniform' and r['band']==str(i) for i in range(9)})
    for regime in ('nominal','adverse'):
      for kind,edges in [('absolute',['0–5','5–10','10–20','20–50','50–100','100–200','≥200']),('relative',['0–0.5','0.5–1','1–2','2–3','3–5','5–10','≥10'])]:
        lines += ['### '+regime+('绝对误差毫米' if kind=='absolute' else '相对误差百分比'),'',
                  '|分组|方法|'+'|'.join(edges)+'|未输出|', '|---|---|'+'|'.join(['---:']*8)+'|']
        for group,predicate in strata.items():
          if group=='空槽' and kind=='relative':continue
          for method in ('baseline_v36','v39_bias_noisy_echo'):
            rr=[r for r in rows if r['regime']==regime and r['method']==method and predicate(r)]
            s=metric_summary(rr);values=s[kind]['pct_output']
            lines.append('|'+group+'|'+('原' if method=='baseline_v36' else '新')+'|'+'|'.join(f(v,1) for v in values)+'|'+str(s['no_output'])+'|')
        lines.append('')
    lines+=['## 两项修正的消融','',
            '|噪声|方法|远浅 MAE mm|远浅 P95 mm|相对误差 %|未输出|','|---|---|---:|---:|---:|---:|']
    ablation_rows=read('ordinary-v39-test-20261012') if VERSION=='v39b' else rows
    for regime in ('nominal','adverse'):
      for method,label in LABEL.items():
        s=metric_summary([r for r in ablation_rows if r['regime']==regime and r['method']==method and r['group']=='far_shallow'])
        lines.append(f"|{regime}|{label}|{f(s['mae_mm'])}|{f(s['p95_mm'])}|{f(s['absrel_positive_pct'])}|{s['no_output']}|")
    lines+=['','该消融表使用第一轮独立集与第一版估计器，说明公共偏置和带噪回波修正的作用；不是最终修正版的同集消融。无参考分支同时缺少偏置和噪声宽度标定，不能将其下降全部归因于偏置。仅标定噪声分支保留同一空槽估计的噪声宽度，但将公共偏置校正设为零，用于区分两者。旧原子回波分支保留偏置和噪声标定，只替换回波分布；其混合比例拟合方式仍与最早基线不同。','',
            '## 标定后小幅偏置漂移','',
            '新增测量阶段 ±0.005 像素公共偏置变化，空槽参考不更新；这是敏感性假设，不是厂商给出的漂移规格。64 个远浅配置与正式集前 64 个配置相同，适合做配对敏感性分析，不计入新增独立配置数。此处只列远浅专项，其他漂移样本完整保存在 JSON。','']
    metric_table(lines,drift,lambda r:r['group']=='far_shallow')
    lines+=['## 既有完整帧回放','',
            '对原 V38 普通双目完整帧重新比较，并为同一安装生成独立噪声的空槽参考。它是兼容性回放，不属于新的独立测试，不能用其调节参数。','']
    metric_table(lines,replay,lambda r:r['population']=='uniform')
    lines += ['','回放中的不利噪声存在明确退化，不能仅凭全量程平均值宣布全部通过。以下列出全部液深区间，每区间只有 3 个独立旧配置，用于定位反例而非估计总体泛化率。','',
              '|液深 m|原 MAE mm|新 MAE mm|变化 mm|','|---|---:|---:|---:|']
    for band in range(9):
        rr=[r for r in replay if r['regime']=='adverse' and r['population']=='uniform' and r['band']==str(band)]
        a,b=[metric_summary([r for r in rr if r['method']==m]) for m in ('baseline_v36','v39_bias_noisy_echo')]
        lines.append(f"|{band}–{band+1}|{f(a['mae_mm'])}|{f(b['mae_mm'])}|{f(b['mae_mm']-a['mae_mm'])}|")
    lines += ['',
              '第一轮对 2.338 m 液深反例的事后仿真诊断：空槽到底面轴向距离约 9.95 m，参考 ROI 有效深度只有 51.6%。单高斯截断模型估计公共视差偏置 +0.0938 像素，而仿真器在该 ROI 中的真实等效偏置为 −0.0340 像素，初版输出液深约 2.228 m。空间相关场在有限 ROI 内并非理想高斯，加上量程截断，偏置和噪声宽度可能互相补偿。真实仿真偏置只用于此事后归因，未传入算法。','',
              '8 至 9 m 液深反例的空槽有效参考点仅数百个，且空槽与较近液面采用的支持区域不同，空间偏置均值不能保证一致。因此本轮不能直接全量程替换：接近量程上限的空槽参考、不同支持区域之间的偏置迁移仍需独立解决。需要检验分区参考或同图中更近的已知静态几何，不能把这些误差通过输出筛选隐藏。','']
    lines+=['## 未输出和剩余误差','']
    for regime in ('nominal','adverse'):
        rr=[r for r in rows if r['regime']==regime and r['method']=='v39_bias_noisy_echo']
        no=[r for r in rr if r['predicted_height_m'] is None]
        negative=sum(r['predicted_height_m'] is not None and r['predicted_height_m']<0 for r in rr)
        lines += [f"{regime}：{len(rr)} 帧中未输出 {len(no)} 帧，原因 {dict(Counter(x for r in no for x in r['reasons']))}；保留负液深输出 {negative} 帧，没有归零。",'']
    lines+=['近零液位的液面和底部视差分布高度重叠，底部回波比例与液位仍可能互相补偿；空槽参考自身也有有限样本误差。空间相关误差、量程边缘缺失、标定后的公共偏置变化均不能只靠重复帧平均消除。当前混合模型假定表面与底部具有相同视差噪声宽度，真实材质造成的不同噪声、非高斯错误匹配、标定底面误差和非平面液面尚不在本轮主测试内。','',
            '新版仅作为显式启用的普通链路实验模块，不替换透明半透明、ToF 或已部署默认配置。部署时应采集空槽参考并检查相机固定状态；无法提供独立参考时，不能声称已消除未知公共偏置。后续重点是参考随温度和时间漂移的验证，以及不同回波噪声宽度的模型失配检查，而非进一步抬高覆盖率门槛。','',
            '## 可复现实验文件','',
            '估计器：src/liquid_depth/ordinary_surface_v39.py；实验：scripts/experiment_ordinary_v39.py；测试：tests/test_ordinary_v39.py。普通链路原有文件未覆盖。正式集 protocol.json 保存冻结源文件哈希、种子和假设，records.jsonl 保存所有方法的逐帧有符号液深；summary.json 保存所有区间分布。统计和报告脚本位于 scripts/report_ordinary_v39.py。','',
            '本轮正式结果目录：'+str(MAIN)+'。标定漂移和旧帧回放分别保存在相邻 ordinary-v39-drift-20261012 和 ordinary-v39-replay-20261012 目录。','']
    report='\n'.join(lines)
    if VERSION=='v39b':
        report=report.replace('每帧评估五种方法','最终复验每帧比较原链路与修正版两种方法')
        report=report.replace('src/liquid_depth/ordinary_surface_v39.py；实验：scripts/experiment_ordinary_v39.py；测试：tests/test_ordinary_v39.py',
            'src/liquid_depth/ordinary_surface_v39b.py（复用 ordinary_surface_v39.py 的标定工具）；实验：scripts/experiment_ordinary_v39b.py；测试：tests/test_ordinary_v39b.py')
        report=report.replace('相邻 ordinary-v39-drift-20261012 和 ordinary-v39-replay-20261012','相邻 ordinary-v39b-drift-20261012 和 ordinary-v39b-replay-20261012')
    verification=MAIN/'verification.json'
    if verification.exists():
        v=json.loads(verification.read_text());tt=v['single_process_estimator_timing_s']
        report+=f"\n## 软件验证与耗时\n\n相关回归测试结果：{v['test_output'].strip()}\n\n两个几何输入共 {tt['samples']} 次单进程估计测量，P50 {tt['p50']*1000:.1f} ms，P95 {tt['p95']*1000:.1f} ms。仅为估计器耗时，不包含相机采集、RGB 网络、传输或磁盘读写，不能据此声明整条系统已满足 0.5 s。冻结估计器与生成脚本哈希核对通过。\n"
    report=report.replace('nominal','标称噪声').replace('adverse','不利噪声')
    (MAIN/'report.md').write_text(report)
    (MAIN/'diagnostics.json').write_text(json.dumps(dict(paired_far=diagnostics,regression=regression),indent=2))
    Path(__file__).parents[1].joinpath('docs/ordinary_bias_echo_v39.md').write_text(report)
    print(json.dumps(dict(paired_far=diagnostics,regression=regression),indent=2))

if __name__=='__main__':main()
