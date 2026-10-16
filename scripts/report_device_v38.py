"""Export all strata and a human-readable device-profile evaluation report."""
import csv,json,math
from pathlib import Path
from collections import Counter
import numpy as np
from liquid_depth.device_profiles_v38 import metric_summary,PROFILES

OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/device-v38-test-20261012')
NAMES={'ordinary':'普通','low_light':'低照','glare':'眩光','floating_objects':'漂浮物','uneven_surface':'不平整',
       'depth_failure':'大面积失效','transparent':'透明','translucent':'半透明','translucent_mild':'半透明',
       'multilayer':'多层透明','compound':'极端复合'}
CAM={'blaze101_long':'ToF','gemini2xl_dense':'双目'}
MAIN={'ordinary':'ordinary_profile','low_light':'lowlight_enhanced'}
BANDS=['空槽','0.01–0.10 加密','0–1','1–2','2–3','3–4','4–5','5–6','6–7','7–8','8–9']
ABS_LABELS=['<5','5–10','10–20','20–50','50–100','100–200','≥200']
REL_LABELS=['<0.5','0.5–1','1–2','2–3','3–5','5–10','≥10']

def fmt(x,n=2):return '—' if x is None else f'{x:.{n}f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def csvwrite(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    native=json.loads((OUT/'letterbox/test-raw.json').read_text())['records']
    prior=json.loads((OUT/'five-prior-independent/test-raw.json').read_text())['records']
    summaries=json.loads((OUT/'letterbox/summary.json').read_text());ps=json.loads((OUT/'five-prior-independent/summary.json').read_text())
    flattened=[]
    for key,groups in summaries.items():
        profile,scene,regime,variant,method=key.split('|')
        for group,s in groups.items():
            if group=='no_output_reasons' or not s['frames']:continue
            row=dict(profile=profile,scenario=scene,noise=regime,variant=variant,method=method,stratum=group)
            for k in ['frames','outputs','no_output','clusters','numeric_output_pct','mae_mm','p95_mm','signed_bias_mm','absrel_positive_pct','relative_outputs','relative_undefined_empty','pass_pct']:row[k]=s[k]
            for kind,labels in [('absolute',ABS_LABELS),('relative',REL_LABELS)]:
                for i,label in enumerate(labels):
                    row[kind+'_'+label+'_count']=s[kind]['counts'][i]
                    row[kind+'_'+label+'_pct_output']=s[kind]['pct_output'][i]
                    row[kind+'_'+label+'_pct_all']=s[kind]['pct_all'][i]
            flattened.append(row)
    for scene,groups in ps.items():
        for group,s in groups.items():
            if not s['frames']:continue
            row=dict(profile='gemini2xl_dense',scenario=scene,noise='image_proxy',variant='base',method='five_prior',stratum=group)
            for k in ['frames','outputs','no_output','clusters','numeric_output_pct','mae_mm','p95_mm','signed_bias_mm','absrel_positive_pct','relative_outputs','relative_undefined_empty','pass_pct']:row[k]=s[k]
            for kind,labels in [('absolute',ABS_LABELS),('relative',REL_LABELS)]:
                for i,label in enumerate(labels):
                    row[kind+'_'+label+'_count']=s[kind]['counts'][i]
                    row[kind+'_'+label+'_pct_output']=s[kind]['pct_output'][i]
                    row[kind+'_'+label+'_pct_all']=s[kind]['pct_all'][i]
            flattened.append(row)
    csvwrite(OUT/'all-strata.csv',flattened)
    records=[]
    for r in native+prior:
        h=r['true_height_m'];v=r['predicted_height_m'];e=abs(v-h) if v is not None else None
        records.append(dict(profile=r['profile'],scenario=r['scenario'],regime=r['regime'],variant=r['variant'],
            method=r['method'],cluster=r['cluster'],population=r['population'],band=r['band'],
            observation_id='|'.join(str(x) for x in (r['profile'],r['cluster'],r.get('repeat',h),r['regime'],r['variant'])),
            actual_liquid_height_m=h,measured_liquid_height_m=v,absolute_error_mm=e*1000 if e is not None else None,
            relative_error_pct=e/h*100 if e is not None and h>0 and not r['empty'] else None,
            numeric_output=v is not None,within_tolerance=e<=max(.005,.02*h) if e is not None else False,
            camera_to_bottom_m=r['camera_to_bottom_m'],camera_to_surface_m=r['camera_to_surface_m'],
            reasons=json.dumps(r.get('reasons',r.get('diagnostics',{}).get('reasons',[])),ensure_ascii=False)))
    csvwrite(OUT/'all-frame-errors.csv',records)
    text=['# 十米相机档案下各场景液深误差分析\n',
      '本次采用 Basler blaze-101 Long 和 Orbbec Gemini 2 XL Dense 两套参考档案，评估相机到底面不超过10 m、实际液深0–9 m的顶置测量。数值是给定噪声及材质假设下的仿真结果，用于确定算法优化顺序；尚未构成两款实机的液位精度标定。\n',
      '## 相机档案和参数依据\n',
      table(['参数','ToF 档案','红外增强双目档案'],[
        ['型号及模式','Basler blaze-101 Long','Orbbec Gemini 2 XL Unbinned Dense'],
        ['深度分辨率','640×480','1280×800'],['使用帧率','20 fps','10 fps'],
        ['工作距离','0.3–10 m','本实验0.4–10 m'],['水平视场','67°','91°'],
        ['基线','不适用','100 mm'],['红外波长','940 nm','850 nm'],
        ['RGB输入','假定外置RGB理想配准','假定RGB与深度理想配准']]),
      '\n[Basler 官方档案](https://docs.baslerweb.com/blaze-101)的典型±5 mm准确度和小于1 mm时间噪声只覆盖特定条件下的0.5–5.5 m；5.5–10 m不能继承这一精度。[准确度与精密度说明](https://docs.baslerweb.com/accuracy-and-precision)区分均值偏差与时间噪声。[Gemini 官方资料](https://orbbec.com/wp-content/uploads/2023/12/ORBBEC_Datasheet_Gemini-2-XL.pdf)的≤2%@4 m指标以最佳拟合平面RMS描述，不能解释为液深误差上限。双目档案属于红外增强双目，未模拟单相机多相位结构光设备。\n',
      '仿真采用水平视场计算的理想去畸变方形像素内参，垂直视场由长宽比确定，和规格书名义角度有小幅差异。真实镜头畸变、逐像素工厂校正表、专有投射图案、自动曝光及多频ToF解调没有公开参数，未作为已知量使用。\n',
      '## 噪声假设\n',
      '双目在原生像素域加入0.18 px或0.36 px视差标准差，视差量化为1/16 px，再导出毫米单位深度；没有另外叠加旧版Z²深度噪声。两组假设分别加入±0.025 px、±0.05 px的共同视差偏置。\n',
      'ToF在径向距离上加入噪声再转回轴向深度。基础标准差为0.8 mm + 1.2 mm×max(距离−5.5 m, 0)²；不利假设乘3。另有跨帧共同零偏及远端偏差斜率。不利组50%的噪声方差来自持续空间块状偏差。以上数值均为敏感性假设，不代表厂家提供了这些误差模型。\n',
      '可见光低照只改变RGB外观，不自动增加主动红外深度空洞。其余材质空洞、错误回波、多径及混合像素沿用项目的参数化场景模型，未完成辐射传输或设备幅值标定。\n',
      '## 采样和评价口径\n',
      f'原生RGB场景660组；传感器观测6864帧。0–9 m每个整米区间等量采样，每个相机和场景在每米区间有3组独立几何配置，每组4次静态采样。空槽和1–10 cm加密组独立统计。双目折射链路另有{len(prior)}个测试帧，使用24组独立容器及纹理配置，每组严格5帧先验。',
      '\n每帧均以相对内底面的实际液深评价。保持放宽门控，保留负测量值和超容器范围的测量值；不强制归零，不裁剪液深。绝对误差是|测量液深−真实液深|；相对误差为绝对误差/真实液深，空槽不定义相对误差。通过容差为max(5 mm, 2%×真实液深)，通过率以全部测试帧为分母。\n',
      '表中的“输出率”仅代表存在有限数值；这些研究链路没有完成工业置信度标定。误差分箱占比以对应可计算误差的输出帧为分母，CSV同时保存全部帧分母的占比及未输出数量。4次静态重复不视作4组独立场景，每米3组配置的结果主要用于筛查薄弱点。\n',
      '场景类型由预先配置指定。普通使用量化感知原生深度拟合；低照使用增强图像提供的预测掩码；其余表面链路使用冻结V8预测掩码和原生深度拟合。不平整场景采用分区表面，同时保存平均、最浅、最深和分位液深误差。背景、漂浮物和玻璃壁参与全画面模拟；GT液面掩码只进入离线诊断。\n',
      '网络输入按比例缩放并补边至320×256，预测结果去除补边后映射回原生网格；传感器误差和几何拟合均在原生深度网格处理。该处理保持相机原有纵横比，另保存直接拉伸输入的对照结果。主处理方式由几何要求预先指定，没有按测试误差选择。误差分箱采用左闭右开区间，最后一箱含全部更大误差。\n',
      '相机到液面距离、液深和相机到底面距离满足近似关系D=Z+H。因此在D≤10 m时，高液深对应较短的相机到液面距离；各液深区间的难度差异同时受到观测距离和相对误差分母影响。两类距离分层均记录在CSV中。\n',
      '## 表面测量链路的分场景结果\n',
      '下表只统计等量的0–9 m主采样，空槽和加密组不混入总体均值。透明、半透明和多层透明的表面链路属于适用性诊断；双目透明液体的五帧先验结果见后表。\n']
    top=[]
    for regime in ['nominal','adverse']:
        text.append('### '+('基础噪声假设' if regime=='nominal' else '不利噪声假设')+'\n');trs=[]
        for scene in NAMES:
            if scene=='translucent_mild':continue
            for profile in CAM:
                method=MAIN.get(scene,'native_refined');s=summaries['|'.join([profile,scene,regime,'base',method])]['uniform']
                trs.append([NAMES[scene],CAM[profile],fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['absrel_positive_pct']),fmt(s['pass_pct']),fmt(s['numeric_output_pct'])])
                top.append(dict(profile=profile,scenario=scene,regime=regime,method=method,**s))
        text.append(table(['场景','相机','MAE mm','P95 mm','平均相对误差 %','容差通过 %','输出 %'],trs)+'\n')
    text+=['## 双目透明和半透明的五帧先验结果\n',
       '该实验使用Gemini档案的焦距、100 mm基线和原生像素裁剪，按Snell折射映射生成有限底纹左右图。先验为1帧空槽＋4帧已知液深（容器容量的15%、40%、65%、90%），先验高度含测量扰动，测试帧噪声独立。保持原有透明配准及半透明对称配准策略。此处不含额外透明容器壁、多层液体或强散射。',
       '\n这一链路读取图像，不经过厂商视差解算器，因此上面的深度图噪声档位不直接作用于这组结果；它有独立的图像噪声和透射衰减假设。ToF目前缺少对应的单目强度／相位折射逆模型，尚不能套用左右目先验模块。\n']
    trs=[]
    for scene in ps:
        for pop in ['uniform','empty','near_zero_enrichment']:
            s=ps[scene][pop];trs.append([NAMES[scene],{'uniform':'0–9 m均匀','empty':'空槽','near_zero_enrichment':'5/20/50 mm'}[pop],s['frames'],fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['absrel_positive_pct']),fmt(s['pass_pct']),fmt(s['numeric_output_pct'])])
    text.append(table(['场景','分组','帧数','MAE mm','P95 mm','相对误差 %','通过 %','输出 %'],trs)+'\n')
    text+=['## 分液深误差区间\n',
       '以下按液深分层列出绝对误差和相对误差分布。每行占比的分母为有效输出；空槽只计算绝对误差。主表使用基础噪声，完整不利噪声、传感器距离分层、基线和GT掩码诊断保存在all-strata.csv。\n']
    for profile in CAM:
      for scene in ['ordinary','low_light','glare','floating_objects','uneven_surface','depth_failure','multilayer','compound']:
        groups=summaries['|'.join([profile,scene,'nominal','base',MAIN.get(scene,'native_refined')])]
        text.append('### '+CAM[profile]+NAMES[scene]+'\n')
        absrows=[];relrows=[]
        for b in range(11):
            s=groups['band_'+str(b)]
            absrows.append([BANDS[b],s['frames'],s['no_output'],fmt(s['mae_mm']),fmt(s['p95_mm'])]+[fmt(v,1) for v in s['absolute']['pct_output']])
            if b:relrows.append([BANDS[b],s['relative_outputs'],fmt(s['absrel_positive_pct'])]+[fmt(v,1) for v in s['relative']['pct_output']])
        text.append(table(['液深m','总帧','无输出','MAE mm','P95 mm']+[x+'mm %' for x in ABS_LABELS],absrows)+'\n')
        text.append(table(['液深m','相对误差样本','平均 %']+[x+'%占比' for x in REL_LABELS],relrows)+'\n')
    for scene,groups in ps.items():
        text.append('### 双目五先验'+NAMES[scene]+'\n');a=[];b=[]
        for i in range(11):
            s=groups['band_'+str(i)]
            a.append([BANDS[i],s['frames'],s['no_output'],fmt(s['mae_mm']),fmt(s['p95_mm'])]+[fmt(v,1) for v in s['absolute']['pct_output']])
            if i:b.append([BANDS[i],fmt(s['absrel_positive_pct'])]+[fmt(v,1) for v in s['relative']['pct_output']])
        text.append(table(['液深m','总帧','无输出','MAE mm','P95 mm']+[x+'mm %' for x in ABS_LABELS],a)+'\n')
        text.append(table(['液深m','平均 %']+[x+'%占比' for x in REL_LABELS],b)+'\n')
    text+=['## 复现和结果文件\n',
       '服务器实验目录：`'+str(OUT)+'`。`protocol.json`及`adapter-plan.json`保存参数、随机种子和输入适配方式；`letterbox/test-raw.json`及`five-prior-independent/test-raw.json`保存正式逐帧结果；根目录`test-raw.json`保留直接拉伸输入的对照。`all-frame-errors.csv`用于逐帧排查；`all-strata.csv`保存全部误差分箱。`frozen-source.json`和`verification.json`核验代码及权重。先验图像试验按预先记录的种子排除计划，以新纹理种子等量替换试跑配置，最终独立评估仍为24组960帧。\n',
       '测试前440项回归通过。完整测试只运行冻结模型和估计器，没有根据测试结果更新权重或选择新阈值。原生表面链路的批量诊断耗时包含多个比较方法，不等同于部署延时。双目4帧在10 fps下跨300 ms，实际端到端延时还需包含曝光、传输和计算。\n',
       '当前底面、位姿、容器轮廓与RGB配准使用理想安装标定。真实标定偏差、持续运动、长时间失效重定位和厂商内部解码器未由本次测试验证。大面积失效项是当前帧表面链路的检查，尚不能作为既有长时记忆模块的全链路成绩。多层与复合项用于识别现有表面链路的缺口，不代表已经集成专门的多层逆模型。\n']
    analysis=Path(__file__).resolve().parents[1]/'docs/camera_device_v38_analysis.md'
    if analysis.exists():text.insert(2,analysis.read_text(encoding='utf-8'))
    (OUT/'camera-profile-performance-report.md').write_text('\n\n'.join(text),encoding='utf-8')
    (OUT/'headline.json').write_text(json.dumps(dict(surface=top,prior=ps),indent=2))
    print('REPORT',len(records),'records',len(flattened),'strata',flush=True)
if __name__=='__main__':main()
