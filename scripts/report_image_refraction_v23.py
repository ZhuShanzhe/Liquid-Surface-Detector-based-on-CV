#!/usr/bin/env python3
"""Postprocess frozen V23 evaluation; never tunes models or gates."""
import json
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np
from liquid_depth.refraction_matching_v23 import stereo_matches, decode_structured
from liquid_depth.refraction_inverse_v23 import infer
from experiment_image_refraction_v23 import gate,summarize

ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/image-refraction-v23-20261011/final')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/image-refraction-v23-final-20261011')
REPO=Path(__file__).resolve().parents[1]

def number(value):
    return '无输出' if value is None else f'{value:.2f}'

def main():
    cv2.setNumThreads(4)
    progress=json.loads((ROOT/'progress.json').read_text())
    assert progress['status']=='complete' and progress['source_unchanged']
    result=json.loads((ROOT/'evaluation/summary.json').read_text())
    raw=json.loads((ROOT/'evaluation/raw-evaluation.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text())
    gates=json.loads((ROOT/'evaluation/frozen-gates.json').read_text())
    configs={c['id']:c for c in manifest['configs']}
    fits={(p['config_id'],p['optimized'],p['prior_count']):p for p in raw['priors']}
    timings=[]
    for config in manifest['configs']:
        if config['split']!='test' or config['camera_to_bottom_m'] not in (.8,6.):continue
        if config['material_seed']%2:continue
        prior=fits[(config['id'],True,3)]
        if not prior['accepted']:continue
        capture=config['captures'][8]
        folder=Path(capture['folder'])
        cal=json.loads((folder/'calibration.json').read_text())
        images={p.stem:cv2.imread(str(p),0) for p in folder.glob('*.png')}
        times=[]
        for i in range(6):
            start=time.perf_counter()
            obs=(stereo_matches(images['left'],images['right'],cal,True)
                 if config['family']=='active_stereo' else decode_structured(images,cal,True))
            prediction=infer(obs,prior['parameters'],cal)
            if i:times.append(1000*(time.perf_counter()-start))
        timings.append(dict(config_id=config['id'],family=config['family'],condition=config['condition'],
            bottom_m=config['camera_to_bottom_m'],repeats=5,compute_ms_p50=float(np.median(times)),
            compute_ms_max=float(max(times)),includes_matching_and_height_inverse=True,
            excludes_io_capture_prior_calibration_and_gui=True,observable=prediction['observable']))
    (ROOT/'evaluation/timing.json').write_text(json.dumps(timings,indent=2))
    # Actual rendered observations, without ground-truth geometric overlays.
    panels=[]
    labels=['transparent','translucent_mild','translucent_strong']
    for condition in labels:
        config=next(c for c in manifest['configs'] if c['split']=='test' and c['family']=='active_stereo'
            and c['condition']==condition and c['camera_to_bottom_m']==.8)
        image=cv2.imread(str(Path(config['captures'][9]['folder'])/'left.png'))
        cv2.putText(image,condition,(12,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
        panels.append(image)
    cv2.imwrite(str(ROOT/'rendered-observations.png'),np.hstack(panels))
    lines=['# 透明与半透明液体：图像匹配及少帧已知液深先验验证（V23）','',
        '日期：2026-10-11。状态：研究实验；生产默认流程未启用；无真实相机精度认证。','',
        '## 主要结论','',
        '少帧已知液深先验能明显减小折射带来的系统偏差，图像前端优化也降低了对应点误差。'
        '但尚不能据此将透明／半透明全量程标记为已适配：三帧物理校正加门控，六组场景中仅结构光透明组整体同时达到困难场景误差、通过率与覆盖率目标。',
        '',
        '物理校正后的低平均误差附带明显的选择性输出。主动双目透明／轻度半透明的覆盖率约 48%，结构光轻度／强散射半透明为 50%／40%。'
        '主要损失来自先验标定失败，不是最后一道不确定度门槛。五帧并不总优于三帧：结构光轻度半透明覆盖率改善至 60%，'
        '主动双目三类的覆盖率反而下降；当前整帧留一检查、视差偏差模型和散射处理仍需改进。',
        '',
        '简单标定同样值得保留：结构光轻度半透明的单帧比例校正达到 AbsRel 1.27%、通过率 85%、覆盖率 100%，'
        '全帧有效通过率高于当前物理反演。不能仅按接受帧误差最低就选择物理分支，也不能根据这批测试结果直接拼接一个“最优”路由后宣称验证通过。','',
        '## 实验回答的问题','',
        '本轮从合成左右目／投影编码灰度图像提取对应点，再根据已知容器底面和折射几何估计实际液深。'
        '它验证的是“透过液体看到底面时，能否利用几帧已知液深校正后续测量”，不是直接把底面回波当作液面点拟合平面。','',
        '双目使用局部对比度归一化、SGBM 初值、二维 LK 跟踪、反向一致性及相关性筛选；'
        '结构光使用 Gray 码与四步相移解码，并对归一化相位矢量平滑。半透明采用双程衰减、模糊和散射幕近似。'
        '先验拟合共享折射率及水平匹配偏置，执行整帧留一验证；输出再经过几何与不确定度门控。','',
        '## 边界与数据划分','',
        '- 相机位于容器上方，底面与液面水平，底面距离及双目／投影器内外参已知；液体折射率均匀。',
        '- 这是解析 Snell 图像渲染，非 Blender 光谱红外、非完整辐射传输、非真实设备标定。正反模型共享光学假设，结果可能偏乐观。',
        '- 未包含玻璃侧壁、弯月面、波浪、漂浮物、镜面高光、真实多径、运动中的编码序列或传感器内外参误差；不能据此宣称这些场景达标。',
        '- 同一配置内先验与测试共用底面纹理、相机和材质；开发集与测试集使用不同随机种子和液体折射率组合。',
        '- 每个传感器／场景的 60 帧来自 10 个配置，帧之间存在相关性；当前未给出独立配置自助抽样置信区间，点估计不是统计保证。',
        '- 独立测试相机到底面距离为 0.3、0.8、2、6、9.8 m，折射率为 1.30、1.45；每一传感器／场景 60 个测试液位。实际液深为底面距离的 8%–72%，不是 10 m 液深验证。',
        '- 已知先验液位为底面距离的 10%、22%、35%、50%、65%；三帧用第 1、3、5 帧，五帧全部使用。液深标签噪声标准差为 0.5 mm + 0.1% 液深。',
        '- 共有 858 组采集，其中 360 组为独立测试液位；每组双目两张图，结构光 30 张图。额外保存匹配结果及仅供评分的真值。',
        '- 为控制光学厚度，散射衰减系数按底面距离归一化；因此远距离组不等同于同一种真实浑浊液体加深后的透射能力。','',
        '## 指标与门控','',
        '液深 H 是相对容器底面的实际高度。MAE（mm）、AbsRel（%）和通过率仅在有效输出上计算；'
        '逐帧通过容差为 max(5 mm, 2% × 真实液深)。覆盖率为输出数／全部测试帧；有效通过率为满足容差的输出数／全部帧。'
        '无输出时误差和通过率不定义。仿真输出均有真值，可评估输出率为 100%，这不等于部署可靠率。','',
        '支持的复杂场景目标为 AbsRel ≤ 3%、通过率 ≥ 75%、覆盖率 ≥ 80%、可评估输出率 ≥ 98%。'
        '门控仅在开发集上选择：开发集输出通过率 ≥ 90%、误接受帧占全部帧 ≤ 5%，在满足者中最大化有效通过率。'
        '测试集不参与门槛拟合；测试误接受率仍可能超过开发约束。三帧拟合出的同一门槛也用于五帧敏感性比较。','',
        '无先验同时报告两种基线：直接按表面三角测距换算液深；假定液体折射率固定为 1.333 的物理反演。'
        '单帧仅校正比例，三帧线性校正为数据驱动对照。物理模型中折射率及偏置共同拟合，单帧不足以可靠标定两者。','']
    names={'active_stereo':'主动双目','structured_light':'结构光',
           'transparent':'透明','translucent_mild':'轻度半透明','translucent_strong':'强散射半透明'}
    methods={'no_prior_raw_surface':'无先验：直接测距','no_prior_fixed_water':'无先验：固定 n=1.333',
             'one_prior_gain':'1 帧比例校正','three_prior_affine':'3 帧线性校正',
             '3_prior_physical':'3 帧物理校正','5_prior_physical':'5 帧物理校正'}
    lines+=['## 三帧物理校正加门控：目标核对','',
        '| 场景 | AbsRel % | 通过率 % | 覆盖率 % | 整组目标 |',
        '|---|---:|---:|---:|---|']
    for r in result['summary']:
        if not (r['optimized'] and r['method']=='3_prior_physical' and r['gate']=='validation_gate'):continue
        ok=(r['absrel_pct'] is not None and r['absrel_pct']<=3 and r['pass_pct']>=75
            and r['coverage_pct']>=80 and r['evaluable_output_pct']>=98)
        lines.append('| '+names[r['family']]+'/'+names[r['condition']]+' | '+
            ' | '.join(number(r[k]) for k in ('absrel_pct','pass_pct','coverage_pct'))+' | '+
            ('满足本轮限定仿真目标' if ok else '未满足：覆盖不足')+' |')
    lines+=['','## 同一批接受帧的先验前后对比','',
        '以下固定使用三帧物理校正加门控接受的帧，在完全相同的帧上比较各方法 AbsRel，隔离筛选不同样本造成的表面收益。'
        '该表不代表全帧覆盖表现，完整分母见后面的各场景表。','',
        '| 场景 | 相同帧数 | 无先验直接测距 % | 1 帧比例 % | 3 帧线性 % | 3 帧物理 % |',
        '|---|---:|---:|---:|---:|---:|']
    paired=[]
    for family in ('active_stereo','structured_light'):
        for condition in labels:
            rs=[r for r in raw['records'] if r['split']=='test' and r['optimized']
                and r['family']==family and r['condition']==condition]
            selected={(r['config_id'],r['sample_id']) for r in rs if r['method']=='3_prior_physical'
                and gate(r,gates[family]['selected_multiplier'])}
            values=[]
            for method in ('no_prior_raw_surface','one_prior_gain','three_prior_affine','3_prior_physical'):
                common=[dict(r,accepted=True) for r in rs if r['method']==method
                    and (r['config_id'],r['sample_id']) in selected]
                assert all(r['predicted_height_m'] is not None for r in common)
                metrics=summarize(common);values.append(metrics['absrel_pct'])
                paired.append(dict(family=family,condition=condition,method=method,**metrics))
            lines.append('| '+names[family]+'/'+names[condition]+f' | {len(selected)} | '+
                ' | '.join(number(v) for v in values)+' |')
    (ROOT/'evaluation/paired-summary.json').write_text(json.dumps(paired,indent=2))
    lines+=['']
    for family in ('active_stereo','structured_light'):
        for condition in labels:
            lines+=['## '+names[family]+'／'+names[condition],'',
                '| 方法 | MAE mm | AbsRel % | 通过率 % | 覆盖率 % | 有效通过率 % |',
                '|---|---:|---:|---:|---:|---:|']
            rows=[r for r in result['summary'] if r['optimized'] and r['family']==family and r['condition']==condition]
            for r in rows:
                title=methods[r['method']]+('＋验证集门控' if r['gate']=='validation_gate' else '')
                lines.append('| '+title+' | '+' | '.join(number(r[k]) for k in
                    ('mae_mm','absrel_pct','pass_pct','coverage_pct','effective_pass_pct'))+' |')
            lines+=['']
    lines+=['## 图像匹配本身的改进','',
        'EPE 为图像提取的对应点与渲染真值之间的二维像素距离；真值仅用于离线评分，匹配与推理均不读取真值对应点。'
        '误差为逐帧均值再平均，匹配区域数量变化会改变其样本集合，不能单独将 EPE 降低解释为整体覆盖提高。','',
        '| 传感器／场景 | 前端 | 平均匹配区域 | EPE px | 匹配耗时 P95 ms |',
        '|---|---|---:|---:|---:|']
    for r in result['matching']:
        lines.append('| '+names[r['family']]+'/'+names[r['condition']]+' | '+('优化' if r['optimized'] else '基线')+' | '+
            ' | '.join(number(r[k]) for k in ('regions_mean','frame_mean_epe_px','matching_ms_p95'))+' |')
    lines+=['','## 三帧物理校正的距离分解（含验证集门控）','',
        '| 传感器／场景 | 底面距离 m | AbsRel % | 通过率 % | 覆盖率 % |',
        '|---|---:|---:|---:|---:|']
    for r in result['by_range']:
        if r['optimized'] and r['method']=='3_prior_physical' and r['gate']=='validation_gate':
            lines.append('| '+names[r['family']]+'/'+names[r['condition']]+' | '+
                ' | '.join(number(r[k]) for k in ('camera_to_bottom_m','absrel_pct','pass_pct','coverage_pct'))+' |')
    lines+=['','## 标定失败与延时','']
    for family in ('active_stereo','structured_light'):
        failures=Counter()
        total=0
        for p in raw['priors']:
            c=configs[p['config_id']]
            if c['split']=='test' and c['family']==family and p['optimized'] and p['prior_count']==3:
                total+=1
                if not p['accepted']:failures[p['reason']]+=1
        lines.append(f'- {names[family]}：三帧标定失败 {sum(failures.values())}/{total} 个配置；原因 {dict(failures)}；门控倍率 {gates[family]["selected_multiplier"]}。')
    lines+=['',
        '三帧方案的 180 帧主动双目测试中，90 帧因所属配置先验标定失败而无输出，另 5 帧匹配区域不足；'
        '最终不确定度门控仅在剩余 85 帧中再拒绝 2 帧。结构光的 180 帧中，72 帧因先验失败无输出，后置门控没有进一步拒绝。'
        '因此直接放宽最终门控无法修复主要覆盖损失。',
        '',
        '原因分析：双目基线为 55 mm、焦距为 450 px，6–9.8 m 时底部视差很小。'
        '固定像素级偏差会被放大成明显米制误差，液深、折射率和匹配偏置之间也更难分离。'
        '这是当前短基线、分辨率和参数化下的敏感性限制，不证明所有 RGB-D 硬件都无法测量。'
        '半透明散射使对比度降低、匹配失真，并引入与液深相关的偏差；单一共享水平偏置不足以解释全部变化。'
        '目前该因果解释依据几何敏感性和失败分布，尚未通过逐项消融排除标定门槛过严等因素。',
        '',
        '后续建议：先用空容器／刚性目标固定内外参和设备匹配偏置，再让液体先验主要估计折射率；'
        '对散射引入由图像对比度、调制度与光学厚度驱动的低维残差，并约束其幅度与单调性。'
        '按先验质量选择有信息量的液位跨度，使用可剔除单个异常先验的稳健拟合；置信度需在独立开发数据上按距离和可见性校准。'
        '物理校正与简单标定的路由也应在新开发集学习，冻结后用新测试集复验，而非在当前测试上挑优。','']
    lines+=['','下表为从已载入图像开始的匹配＋液深反演，5 次重复的中位数／最大值；不含曝光、传输、读盘、离线标定、界面和其他生产模块。'
        '仅抽取成功标定配置，不能代替生产系统端到端延迟验收。','',
        '| 场景 | 底面距离 m | 计算中位数 ms | 计算最大值 ms |',
        '|---|---:|---:|---:|']
    for r in timings:
        lines.append('| '+names[r['family']]+'/'+names[r['condition']]+' | '+
            ' | '.join(number(r[k]) for k in ('bottom_m','compute_ms_p50','compute_ms_max'))+' |')
    lines+=['',
        '结构光每次测量需要 30 次曝光：60 fps 理想采集下限为 500 ms，120 fps 为 250 ms，之后仍需计算与传输。'
        '当前版本不能宣称结构光满足 0.5 s 全链路目标；需要减少图案数、投影与曝光同步或更高速硬件。'
        '通用封闭式结构光相机未必提供可编程投影与原始序列，需在选型时确认。主动双目需要同步原始双目图像和正确标定。','',
        '## 复现与审计','',
        f'- 数据清单：`{DATA}/manifest.json`。SHA256：`{raw["manifest_sha256"]}`。',
        f'- 冻结源文件指纹：`{ROOT}/frozen-source.json`；完成时源代码一致性检查通过。',
        f'- 原始逐帧结果、门槛、汇总与延时：`{ROOT}/evaluation/`。',
        '- 本轮新增 8 项测试；完整回归为 323 项通过、14 项既有弃用警告。',
        '- 入口：`scripts/run_image_refraction_v23.py`。数据生成／评测／报告脚本分离；同名输出目录禁止覆盖，复现实验需要指定新目录或修改副本。',
        '- 模块：`refraction_images_v23.py`、`refraction_matching_v23.py`、`refraction_inverse_v23.py`；本次无神经网络重新训练，属于图像前端与物理校正优化。',
        '- 本报告结果不可替换既有 V17/V21 全帧网络评测；问题、输入和仿真假设不同。','']
    report='\n'.join(lines)
    (REPO/'docs/transparent_image_matching_v23.md').write_text(report,encoding='utf-8')
    print('Report saved; timing cases:',len(timings),flush=True)

if __name__=='__main__':main()
