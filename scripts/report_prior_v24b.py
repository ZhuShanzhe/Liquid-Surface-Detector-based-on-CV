"""Read-only evaluation summary and repository report; never selects on test."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from experiment_image_refraction_v23 import summarize,gate

BASE=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/prior-reliability-v24b-20261011')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/prior-reliability-v24b-20261011')
REPO=Path(__file__).resolve().parents[1]
NAMES={'active_stereo':'主动双目','structured_light':'结构光','transparent':'透明',
       'translucent_mild':'轻度半透明','translucent_strong':'强散射半透明'}

def n(x):return '—' if x is None else f'{x:.2f}'

def main():
    assert json.loads((BASE/'progress.json').read_text())['status']=='complete'
    raw=json.loads((BASE/'test-raw.json').read_text());summary=json.loads((BASE/'test-summary.json').read_text())
    routes=json.loads((BASE/'routed-test-raw.json').read_text());policy=json.loads((BASE/'frozen-calibration-policy.json').read_text())
    audit=json.loads((BASE.parent/'prior-reliability-v24-20261011'/'calibration-audit.json').read_text());pairs=[]
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild','translucent_strong'):
            group=[r for r in routes if r['family']==family and r['condition']==condition and r['method']=='route_5']
            old=[dict(r,accepted=gate(r,2 if family=='active_stereo' else 16)) for r in raw['records']
                 if r['family']==family and r['condition']==condition and r['method']=='v23_3']
            baseline={(r['config_id'],r['sample_id']):r for r in old}
            retained=[r for r in group if baseline[r['config_id'],r['sample_id']]['accepted']]
            assert all(r['accepted'] and r['predicted_height_m']==baseline[r['config_id'],r['sample_id']]['predicted_height_m'] for r in retained)
            added=[r for r in group if not baseline[r['config_id'],r['sample_id']]['accepted']]
            pairs.append(dict(family=family,condition=condition,baseline=summarize(old),optimized=summarize(group),
                added=summarize(added),retained_count=len(retained),retained_predictions_identical=True,
                rejection_counts=dict(Counter(reason for r in group if not r['accepted'] for reason in r['diagnostics'].get('reasons',[]) or ['uncertainty_gate']))))
    (BASE/'comparison.json').write_text(json.dumps(pairs,indent=2))
    old_all=[dict(r,accepted=gate(r,2 if r['family']=='active_stereo' else 16)) for r in raw['records'] if r['method']=='v23_3']
    new_all=[r for r in routes if r['method']=='route_5']
    old_metrics=summarize(old_all);new_metrics=summarize(new_all)
    extra_metrics=summarize([r for r in new_all if r['source']!='v23_preserved'])
    (BASE/'overall-comparison.json').write_text(json.dumps(dict(old=old_metrics,new=new_metrics,extra=extra_metrics),indent=2))
    lines=['# 透明／半透明：先验标定失败诊断与覆盖率优化（V24b，第二轮）','',
        '日期：2026-10-11。研究验证完成；默认生产系统和 V17 权重未替换。','',
        '## 主要结论','',
        '第二轮在保持旧输出不变的情况下有限提高覆盖：360 次测试从 180 次输出增加到 202 次，新增 22 次中 21 次满足容差。'
        '新输出的相对误差为 0.70%、通过率为 95.45%。强散射双目新增 3 次中仍有 1 次超差；结构光强散射没有新增输出，不能宣称强散射问题已解决。','',
        '| 整体指标 | 旧三帧 | 新五帧 |',
        '|---|---:|---:|',
        *['| '+label+' | '+n(old_metrics[key])+' | '+n(new_metrics[key])+' |' for label,key in
          [('覆盖率 %','coverage_pct'),('MAE mm','mae_mm'),('P95 绝对误差 mm','p95_mm'),('AbsRel %','absrel_pct'),
           ('输出内通过率 %','pass_pct'),('全帧有效通过率 %','effective_pass_pct')]],'',
        '覆盖率提升 6.11 个百分点，整体相对误差基本保持，但不能说所有精度指标都没有变化：'
        'MAE 从 2.01 mm 上升到 2.67 mm，P95 绝对误差从 6.13 mm 上升到 14.70 mm。'
        '这些统计使用不同的接受集合，旧的 180 次预测未改变，尾部变化来自新增输出。新增输出的风险仍需单独跟踪。'
        '结果为本批仿真点估计，未提供跨配置统计置信区间，尚无实机验证。','',
        '新五帧的有效光学校准映射在两类相机的 30/30 个配置中均可建立；旧三帧物理标定成功率分别为双目 13/30、结构光 18/30。'
        '但是能建立映射不等于输出可靠，最终仍需按每次测量的预测风险拒绝。因此标定成功率不能替代最终覆盖率或准确度。','',
        '## 两轮验证的关系','',
        '第一轮 V24 提高了整体覆盖率，但双目及轻度半透明新增输出的独立通过率不足，因此没有启用生产配置。'
        '第二轮扩大开发集并将新增输出通过率约束提高到 90%。旧 V23 测试数据明确重新归类为开发数据，不能再用于宣称本轮独立验证；'
        'V24 第一轮的新测试集也不作为最终验证结果。V24b 使用第三组图像随机种子 245801 系列及新折射率组合，算法与策略先冻结，再生成图像并评分。'
        '两轮原始结果分别保存，最终表格均来自 V24b 同一批新测试数据。','',
        '## 本轮解决的问题','',
        'V23 的低覆盖既包含真实观测不足，也包含标定策略的整组拒绝。V24 保留旧方案通过门控的结果，'
        '只在旧方案拒绝时尝试新增的液深校准分支；使用全新图像种子和折射率组合复评，未依据新测试标签修改算法或门槛。','',
        '## 为什么标定会失败','',
        '| 旧测试集三帧标定失败原因 | 主动双目／30 配置 | 结构光／30 配置 |',
        '|---|---:|---:|',
        '| 匹配区域不足 | 2 | 0 |',
        '| 折射率到达边界或留一参数标准差超过 0.04 | 9 | 4 |',
        '| 留一液深预测超出 max(5 mm, 5% × 已知液深) | 4 | 8 |','',
        '以上为旧测试集诊断，不能和下文新测试集覆盖率混用。对非匹配不足的失败组，折射率与水平匹配偏置的留一相关系数绝对值中位数：'
        '双目约 0.86，结构光约 0.92。它们会互相补偿，单独对折射率标准差设硬门槛，不能完整表示液深预测的不确定度。','',
        '另一方面，简单取消标定检查并不安全。旧失败组的诊断性强制输出如下；此处只用于定位问题，不是部署结果。','',
        '| 失败组 | 强制输出帧数 | AbsRel % | 通过率 % |',
        '|---|---:|---:|---:|']
    for family in ('active_stereo','structured_light'):
        for failure,label in [('unstable','参数不稳定'),('loo','留一液深失败')]:
            values=[v for r in audit if r['split']=='test' and r['count']==3 and r['family']==family and r['failure']==failure
                    for v in r.get('diagnostic_bypass_outputs',[])]
            lines.append('| '+NAMES[family]+'/'+label+f' | {len(values)} | '+n(100*np.mean([v['rel'] for v in values]))+' | '+n(100*np.mean([v['passed'] for v in values]))+' |')
    lines+=['',
        '根因分为三类：① 强散射使深液位先验的有效匹配变少，旧方法要求每帧都合格，单帧失败会影响整组；'
        '② 远距离短基线下，亚像素匹配偏差被放大，液深、折射率和偏置的联合辨识变弱；'
        '③ 散射及图像匹配误差随液位变化，固定折射率加单一偏置不能描述全部残差。'
        '开发集中 6 m 双目观测的残差随液位改变方向，表明它并非一个固定比例或固定偏置就能完全校正。'
        '当前噪声和散射仍是代理模型，因此这些是模拟器与算法中的诊断，不能直接归因于所有真实相机的硬件极限。','',
        '## 实施的改动','',
        '1. 将匹配光线转换为参考折射率 1.37 下的有效光学液深坐标；先在空间块内稳健聚合，再跨块汇总。该坐标只是可标定观测量，不声称测得真实折射率。',
        '2. 用已知液深建立比例偏置／单调分段／PCHIP 候选映射；方法和门槛只在开发集选择。局部非线性模型不自动视为更优。',
        '3. 五帧先验允许丢弃匹配不充分的帧，仍要求至少三帧、足够液位跨度与单调关系；互相矛盾的标签仍拒绝。尚未自动识别所有数值错误但仍保持单调的先验标签。',
        '4. 从留一液深误差、空间块差异和外推距离构造风险分数。它不是经过真实设备标定的概率置信区间。',
        '5. 保留已通过旧门控的预测；新增分支只补充旧拒绝帧。开发集还单独检查新增输出的误差，避免用大量旧的正确帧掩盖新增错误。','',
        '## 冻结策略与输入预算','',
        '第二轮开发集包含 V23 的原开发集和原测试集，每个传感器共 234 个未知液位，场景帧存在相关性。'
        '整体门控选择要求 AbsRel ≤ 3%、通过率 ≥ 90%、误接受帧／全部帧 ≤ 5%；新增输出至少 3 帧、通过率 ≥ 90%、AbsRel ≤ 3%。'
        '这些小样本开发条件不构成误接受率的统计保证。','']
    for family in ('active_stereo','structured_light'):
        for count in (3,5):
            p=policy[family+'/routes'][str(count)];s=p['selected']
            lines.append(f'- {NAMES[family]}／{count} 帧：主分支 `{p["primary_method"]}`；'+
                (f'补充分支 `{s["method"]}`，风险倍率 {s["multiplier"]}，开发集补回 {s["added"]["outputs"]} 帧。' if s else '开发集无合格补充分支，保持原方案。'))
    lines+=['',
        '新五帧方案可用五个已知液位，但主分支使用开发集选出的旧三帧子集，新增分支才使用全部可用先验。'
        '下表同时列旧三帧、旧五帧与新五帧，避免将增加两帧先验的收益全部归因于算法。','',
        '## 新测试集设计及指标','',
        '- 60 个配置：两种成像方式 × 三种散射条件 × 五个相机到底面距离 × 两个折射率。每配置 5 个先验液位＋6 个未知液位，共 660 组采集、360 组测试测量。',
        '- 距离 0.3、0.8、2、6、9.8 m；新折射率 1.31、1.43；图像随机种子与旧测试／开发集不同。',
        '- 实际测试液深是底面距离的 8%、15%、28%、42%、58%、72%，范围 0.024–7.056 m。不能把相机距离近 10 m 当成 10 m 液深已验证。',
        '- 先验为 10%、22%、35%、50%、65% 底面距离；标签噪声标准差 0.5 mm＋0.1% 液深。未增加空容器图像或额外真实量程信息；底面距离及相机标定仍假定已知。',
        '- 新旧方法使用同一批 PNG 提取的对应点，图像前端保持 V23 不变。模型和风险策略在生成新测试图像之前冻结。',
        '- 仍是水平液面、水平底面、均匀折射率的解析 Snell 图像仿真。散射为衰减／模糊／散射幕近似，系数按底面距离归一化；未验证真实红外、侧壁、多径、波浪和现场外参误差。',
        '- MAE 和 AbsRel 在接受输出上统计；逐帧通过容差 max(5 mm, 2% × 真实液深)。覆盖率是接受输出数／全部测试帧，有效通过率是正确输出数／全部帧。仿真可评估输出率为 100%。','',
        '## 分场景：旧三帧与新五帧','',
        '| 场景 | 覆盖率旧→新 % | AbsRel 旧→新 % | 通过率旧→新 % | 有效通过率旧→新 % |',
        '|---|---:|---:|---:|---:|']
    for p in pairs:
        a,b=p['baseline'],p['optimized']
        lines.append('| '+NAMES[p['family']]+'/'+NAMES[p['condition']]+' | '+' | '.join(n(a[k])+' → '+n(b[k]) for k in ('coverage_pct','absrel_pct','pass_pct','effective_pass_pct'))+' |')
    lines+=['','## 输入数量对照与绝对误差','',
        '| 场景 | 方法 | MAE mm | AbsRel % | 通过率 % | 覆盖率 % |',
        '|---|---|---:|---:|---:|---:|']
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild','translucent_strong'):
            for method in ('v23_3','v23_5','route_3','route_5'):
                r=next(r for r in summary['summary'] if r['family']==family and r['condition']==condition and r['method']==method and r['gate']=='gated')
                lines.append('| '+NAMES[family]+'/'+NAMES[condition]+' | '+method+' | '+' | '.join(n(r[k]) for k in ('mae_mm','absrel_pct','pass_pct','coverage_pct'))+' |')
    lines+=['','## 新增输出的独立检查','',
        '旧三帧已接受的预测在新方案中逐帧完全相同，因此这部分没有精度回退；整组精度变化来自新增输出。'
        '以下只统计新增输出，不把原有正确结果混入新增输出的通过率。','',
        '| 场景 | 保留旧输出数 | 新增输出数 | 新增 AbsRel % | 新增通过率 % |',
        '|---|---:|---:|---:|---:|']
    for p in pairs:
        a=p['added'];lines.append('| '+NAMES[p['family']]+'/'+NAMES[p['condition']]+f' | {p["retained_count"]} | {a["outputs"]} | '+n(a['absrel_pct'])+' | '+n(a['pass_pct'])+' |')
    lines+=['','## 五帧优化后的距离分段','',
        '| 场景 | 底面距离 m | MAE mm | AbsRel % | 通过率 % | 覆盖率 % |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary['by_range']:
        if r['method']=='route_5':lines.append('| '+NAMES[r['family']]+'/'+NAMES[r['condition']]+' | '+' | '.join(n(r[k]) for k in ('bottom_m','mae_mm','absrel_pct','pass_pct','coverage_pct'))+' |')
    lines+=['','## 研发目标核对','',
        '困难场景要求 AbsRel ≤ 3%、输出内通过率 ≥ 75%、覆盖率 ≥ 80%、可评估输出率 ≥ 98%。这是项目研发目标，不是软件著作权登记的法定精度条件。','',
        '| 场景 | 新五帧是否满足整组目标 | 仍未满足项 |',
        '|---|---|---|']
    for p in pairs:
        m=p['optimized'];failed=[]
        if m['absrel_pct'] is None or m['absrel_pct']>3:failed.append('相对误差')
        if m['pass_pct'] is None or m['pass_pct']<75:failed.append('通过率')
        if m['coverage_pct']<80:failed.append('覆盖率')
        lines.append('| '+NAMES[p['family']]+'/'+NAMES[p['condition']]+' | '+('否' if failed else '限定仿真中满足')+' | '+('、'.join(failed) if failed else '无')+' |')
    lines+=['','## 复现与限制','',
        '- 332 项回归测试通过，14 条既有弃用警告。新增 9 项测试覆盖已知液深校准、异常先验、观测缺失、外推、真值隔离和旧结果保留。',
        '- 新测试仅用于评分，没有用于调整门控；若新增输出在某场景未达到精度目标，应保留研究状态，并在新开发／验证集上改进，不能用当前测试标签反选输出。',
        '- 本轮优化为标定与风险控制，无神经网络重新训练。相机图像采集要求和结构光 30 图案延时限制仍与 V23 相同；未重新进行生产端到端计时。',
        '- 未解决单一错误先验数值仍单调、跨液体性质变化、温度引起参数漂移、相机移动后失配等部署问题；相机、容器或液体发生变化需检查／重做标定。',
        '- 后续应优先改善真实可观测性：独立设备偏置标定、量程适配基线与采样分辨率、匹配偏差建模，并以新增输出风险为主要验证项，而非直接放宽拒绝限制。',
        f'- 数据：`{DATA}`；清单 SHA256 `{hashlib.sha256((DATA/"manifest.json").read_bytes()).hexdigest()}`。',
        f'- 审计、开发集策略、新测试逐帧结果及源文件指纹：`{BASE}`。',
        '- 入口：`scripts/audit_prior_v24.py`、`scripts/experiment_prior_v24b.py`、`scripts/run_prior_v24b.py`；核心模块 `src/liquid_depth/prior_reliability_v24.py`。独立测试入口拒绝覆盖既有冻结结果。','']
    report='\n'.join(lines)
    (REPO/'docs/transparent_prior_reliability_v24b.md').write_text(report,encoding='utf-8')
    print('REPORT COMPLETE',flush=True)
    for p in pairs:print(p['family'],p['condition'],p['baseline']['coverage_pct'],p['optimized']['coverage_pct'],p['optimized']['absrel_pct'],p['optimized']['pass_pct'],p['added']['outputs'],p['added']['pass_pct'],flush=True)

if __name__=='__main__':main()
