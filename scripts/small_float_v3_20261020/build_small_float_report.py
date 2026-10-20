import json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020')
s=json.loads((ROOT/'summary.json').read_text());live=json.loads((ROOT/'live-benchmark.json').read_text())['summary'];diagnosis=json.loads((ROOT/'failure-diagnosis.json').read_text())
passed=s['numerical_gate_pass'] and not live['parity_mismatches'] and live['all']['p95_ms']<=500

def table(headers,rows):
    text='.. list-table::\n   :header-rows: 1\n\n'
    for row in [headers]+rows:
        text+='   * - '+str(row[0])+'\n'
        for v in row[1:]:text+='     - '+str(v)+'\n'
    return text+'\n'

def f(v):return f'{v:.2f}'
rows=[]
for bits in ('8','9','10','12','24','0','1','2','4','16'):
    p=s['by_condition'][bits];a,b=p['old'],p['new']
    rows.append([p['name'],str(a['n'])+'/100',f(a['mae_mm'])+' → '+f(b['mae_mm']),f(a['p95_mm'])+' → '+f(b['p95_mm']),f(a['max_mm'])+' → '+f(b['max_mm']),f(a['pass_rate']*100)+' → '+f(b['pass_rate']*100),p['new_activations']])
a,b=s['primary_float']['old'],s['primary_float']['new']
text='''小漂浮物最高高度独立优化与正式选版
==================================

日期：2026-10-20。对象：固定机位 ToF、普通漫反射液体中的小漂浮物及已支持复合因素。
原正式版为 182bbd73aacd5bd1731e5321faf2c427421b4259（formal v2）。
本轮不改液深算法，不触及透明/半透明原始 AMCW 相位链路，不扩展新的场景物理边界。

结论
----

'''
text+=('数值与串联延时检查通过，选为最高高度增强版 formal v3。' if passed else '未通过预声明合并条件，保留 formal v2，不替换正式数值版。')+'\n\n'
text+=f"漂浮物主统计 {a['n']} 帧：MAE {f(a['mae_mm'])} → {f(b['mae_mm'])} mm，P95 {f(a['p95_mm'])} → {f(b['p95_mm'])} mm，通过率 {f(a['pass_rate']*100)}% → {f(b['pass_rate']*100)}%。\n"
text+=f"最大误差仍为 {f(b['max_mm'])} mm，不能声称大误差已消除。全部 1000 帧新旧液深逐帧完全相等。\n\n"
text+='''数据、真值与公平比较
--------------------

新增 260 安装、2600 帧：训练128安装/1280帧；验证32安装/320帧；独立测试100安装/1000帧。
安装ID与随机种子隔离；同安装的各工况有相关性，不把1000帧伪称1000独立容器。
测试10类各生成100帧，其中500帧有漂浮物、500帧无漂浮物对照。
漂浮物数量1/3/12，扩充小、混合尺寸和细高顶部；低照、眩光、起伏、缺失配对比较。
相机到底面约1.7–9.7m；容器最大液深限制约3m。本次不是0–9m全液深重新认证。

标签由仿真几何/回波来源自动生成。场景RGB-D、独立带扰动空槽几何和旧液深输出可以进入推理；
当前真值、语义、实例、回波来源不得进入推理。候选分类训练可以使用标签，独立测试仅推理后读取。
小物体实例使用可见连通区域作代理，接触/重叠实例可能合并，并非完美实例身份标注。
没有超分插值制造独立测距点；仍至少需要3个二维分布的真实深度支撑点。

训练：小物体/最高可见区域加权、语义与可靠回波双头；完整原生画幅训练，23轮验证早停，采用第15轮。
旧视觉特征与新特征两个模型家族均只由320帧验证集选择，测试前冻结选版文件。
原特征候选训练3618个，仅3个可靠漂浮物正例；新特征48276个候选，918个正例，376个最高区域正例。
候选对象、最高区域与高度修正分别训练ExtraTrees。最高区域概率门槛最终由验证集选为0，
因此本版不声称该排序头已经有效；实际使用可信候选的米制最高高度排序，而非面积或语义分数排序。
原生像素头误报依然较多，不能单独用像素置信度代替计量置信度。

统计规则：液深仍按相对内底面的实际高度；最高高度容差为max(50mm,5%真实最高高度)。
在线原始支撑审计与离线回波来源审计不依赖测量误差。缺失工况另外限定实际液面缺失≤50%（一像素舍入容差）。
978/1000帧进入主统计，另22帧仍保留原始结果和全帧统计；不因误差大删除帧。
其中含漂浮物486帧、无漂浮物492帧；眩光合格子类92帧、漂浮物+缺失94帧，不冒充合格分母100。
低照本身可能导致大面积失效；未将其悄悄归入0–50%缺失能力认证。

逐场景独立结果
--------------

旧→新，误差单位mm，通过率单位%。补启用指实际采用新候选且原场景路由未启用视觉，非正确分类率。

'''
text+=table(['场景','合格/生成','MAE','P95','最大误差','通过率%','补启用'],rows)
text+='误差分布（486帧漂浮物主统计）\n--------------------------------\n\n'
text+=table(['绝对误差','原版帧数/占比','新版帧数/占比'],[[k,f"{a['abs_bins'][k]}/{a['abs_bins'][k]/a['n']*100:.2f}%",f"{b['abs_bins'][k]}/{b['abs_bins'][k]/b['n']*100:.2f}%"] for k in a['abs_bins']])
text+=table(['相对误差','原版帧数/占比','新版帧数/占比'],[[k,f"{a['relative_bins'][k]}/{a['relative_bins'][k]/a['n']*100:.2f}%",f"{b['relative_bins'][k]}/{b['relative_bins'][k]/b['n']*100:.2f}%"] for k in a['relative_bins']])
text+='相对误差分母为真实最高高度，浅液位分母小时会放大相对误差；没有用相机到底距离作分母。\n\n'
ci=s['installation_clustered_bootstrap']
text+=f"按安装聚类自助法3000次，MAE变化95%区间 {ci['mae_change_mm_95ci']} mm；通过率变化区间 {ci['pass_change_percentage_points_95ci']} 个百分点。\n\n"
text+='''新模块的作用与边界
------------------

1. 最高高度单独使用容器开口投影及物理外缘约束，不再复用液深内圈0.86半径和底面峰过滤。
   液深选点、平面、回波竞争全部保持原样，禁止把更宽最高搜索范围反向用于液深。
2. 增加漂浮物存在/可靠回波候选探测。即使场景分类漏报漂浮物，也可以提出最高高度候选。
   探测器可在不透明链路运行，但数值视觉只能修改highest_height_m；不能修改liquid_depth_m。
3. 连通、二维支撑、局部一致性、候选对象可信度和修正幅度共同检查；修正超过3倍ToF噪声直接拒绝，
   不把输出裁剪到阈值。不按面积选实例，以合格候选的实际高度选最高者。
4. 只在新证据高于旧最高值时合并，避免这轮漏顶部修复损害已有可靠高点；因此无法降低旧链路的错误高值。
   这个限制是显式安全取舍，不是已解决所有错误回波。

正式手工入口只有floating_objects且visual_enabled时启用新视觉；其他手工场景不启用。
统一自动入口对已确认普通漫反射材质探测小漂浮物，以弥补场景分类漏启用。
透明/半透明相位、未完成物理确认或已被阻断的组合均不进入此辅助模块。
统一入口可small_float_peak=False回放formal v2；手工入口同名选项。旧版完整源码与数据冻结在服务器。
建议初始化调用warm_optional_peak_models，冷启动单独计时；旧运行进程需重启才会加载新代码，不主动中断用户服务。

未解决原因（本轮测试后诊断，不用于重调门槛）
------------------------------------------

'''
labels={'highest_candidate_confidence_or_correction_rejected':'最高区域候选可信度/修正检查拒绝','highest_direct_support_insufficient':'最高顶部真实回波空间支撑不足','inherited_high_outlier_not_demoted':'旧链路错误高值未被本轮单向合并降低','highest_pixel_presence_or_reliability_missed':'最高区域像素存在性/可靠性漏检','highest_component_spatial_or_coherence_filter':'最高区域未通过成组/空间/一致性约束'}
text+=table(['主要诊断类别','剩余超差帧数'],[[labels.get(k,k),v] for k,v in diagnosis['counts'].items()])
text+='''类别按明确优先级分配，不能将“有一些候选但未接受”简单解释成门控过严；候选可能是错误区域。
有真实最高支撑的412帧：MAE18.39→15.30mm，通过率92.96→95.15%，最大87.55→81.86mm。
这表明问题不仅是传感器无返回，还存在顶部小区域漏成组、可靠度偏保守和实例/顶部表达不足。
继承的大误差帧0021_09：真最高0.5865m、输出1.9674m；液面约98.05%缺失，顶部仅1个直接回波。
配对无漂浮物低照帧0021_01也有1.403m最高误差。本轮只上调更可信高点，无法修复此类旧错误高峰；
该极端帧未被隐藏，也不据此宣称支持95%–100%严重失效，不在本轮继续优化。

候选补启用不等于完美存在性分类：492个无漂浮物合格对照中66帧也采用了新候选，
其MAE变动为亚毫米量级、未新增容差失败，仍说明存在性误报需要后续校准，不能用于安全联锁。
含漂浮物486帧实际更新146帧，其中109帧原场景路由未启用视觉；并非完全修复所有视觉漏启用。
后续若继续：优先改善有顶部直接支撑却被漏成组/拒绝的区域；真无足够支撑时采用多帧或更高分辨率实际采样，
不通过真值辅助、放宽空间硬约束或强制填值掩盖问题。本轮不再用该独立测试集调参。

延时与复现
----------

'''
text+=table(['直接顺序CPU串联测试','帧数','中位数ms','P95 ms','最大ms','>500ms帧数'],[[name,v['n'],f(v['median_ms']),f(v['p95_ms']),f(v['max_ms']),v['over_500ms']] for name,v in [('全部1000帧',live['all']),('漂浮物500帧',live['floats']),('无漂浮物500帧',live['controls']),('初始化100次',live['initialization'])]])
text+=f"串联测量使用2个CPU线程，推理不使用GPU；模型读取和预热单列初始化，图像磁盘读取不计入测量延时。\n1000帧实际接入与冻结候选评估输出逐项一致，差异项 {len(live['parity_mismatches'])}。\n"
text+='''P95<0.5s是本轮预声明合并条件，并非硬实时最坏情况保证；最大值和越界数量如实记录。
此前baseline_ms+extra_ms只是估算，本报告以上表直接串联结果为准。
证据：reports/evidence/small-float-v3-20261020（原始逐帧记录、summary、协议、训练曲线、失效诊断与延时）。
服务器保留完整数据、冻结formal v2、脚本和模型；仓库只包含推理权重与可复现的小型记录，不上传大体积原始仿真图像。
依赖：pip install -e '.[tof-vision]'；scikit-learn固定1.9.0匹配受信项目序列化资产，载入校验SHA256与版本。
所有结论仅限本次独立仿真，不等于真实ToF设备计量认证或工业场景保证。
'''
(ROOT/'SMALL_FLOAT_HIGHEST.rst').write_text(text,encoding='utf-8')
(ROOT/'release-decision.json').write_text(json.dumps(dict(merge_allowed=passed,numerical_gate=s['numerical_gate_pass'],live_p95_gate=live['all']['p95_ms']<=500,live_parity=not live['parity_mismatches']),indent=2))
print(json.dumps(dict(merge_allowed=passed,live=live),ensure_ascii=False))
