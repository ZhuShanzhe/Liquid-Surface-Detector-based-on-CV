"""Evidence-backed conclusions: range0.1–9m with fixed10mm calibration."""
import argparse
import json
import os
from pathlib import Path

REVISION = 'range_0p1_9m_fixed_10mm'
MARKER = '\n.. BEGIN_FIXED_RANGE_CONCLUSIONS\n'


def number(value, digits=3):
    return '不适用' if value is None else f'{value:.{digits}f}'


def percent(value):
    return '不适用' if value is None else f'{100 * value:.2f}%'


def derive(data):
    if data.get('evaluation_revision') != REVISION:
        raise ValueError('Conclusions require the fixed10mm range0.1–9m revision')
    primary = data['summary']
    ideal = data['by_calibration_variant']['ideal']['summary']
    provenance = data['evaluation_provenance']
    sampling = data['depth_sampling_audit']
    q = data['qualification']['all']
    notes = [
        '本轮固定标定协议：所有米制输入标定量采用零均值截断高斯，截断前σ=10/3mm、绝对界限±10mm，不再按距离或尺寸百分比放大。物理空槽标签0固定，底面anchor承载零基准误差，像素点击与重力扰动另列；该预算不是实测工业相机误差分布。',
        f"本轮选入{primary['n']}张合格图：复用历史输入{provenance['reused_input_images']}张、新采{provenance['new_input_images']}张；全部{provenance['new_measurement_runs']}次ideal/operator_noise均重新测量，旧数值复用{provenance['reused_measurement_runs']}次。同安装多帧相关，比例只作描述，不给逐帧独立置信区间。",
        f"目标液深0.1–9m，实际{number(primary['actual_liquid_height_range_m'][0],4)}–{number(primary['actual_liquid_height_range_m'][1],4)}m；九带实际图像数为{sampling['overall_counts']}，每类首带12张、其余八带各11张，配额核对={'通过' if sampling['quotas_met'] else '未完成'}。这是分带均衡测试，不是连续量程逐点验证或真实部署频率认证。",
        f"相机到底面协议范围0.9–10m，实际{number(sampling['camera_bottom_actual_range_m'][0],4)}–{number(sampling['camera_bottom_actual_range_m'][1],4)}m。传感器到液面的距离是B−H，高液深与远距浅液位不是同等测距难度。",
        f"同图标定对照：ideal液深MAE {number(ideal['liquid']['mae_mm'])}mm、通过率{percent(ideal['liquid_pass'])}；固定±10mm输入标定版本MAE {number(primary['liquid']['mae_mm'])}mm、通过率{percent(primary['liquid_pass'])}，后者是主体。功能通过分别为{percent(ideal['functional_pass'])}和{percent(primary['functional_pass'])}；同图变化用于描述标定影响，不将旧协议范围/配额变化混成算法提升。",
        f"复用输入已被历史资格和本次范围/配额条件筛选，不构成新随机候选。新采候选{q['new_observed_attempts']}张，合格{q['new_qualified']}张、前置排除{q['new_skipped']}张，新采资格比例{percent(q['new_candidate_qualification_fraction'])}；旧库配额外合格图未删除，不算本轮资格失败。合格输入液深数值覆盖率{percent(primary['liquid']['coverage'])}。",
        '功能通过要求液深通过，且真实波面或漂浮物存在时最高高度也通过。普通液深采用max(20mm,2%真值)，波面液深和最高高度采用各自max(50mm,5%真值)；不同通过率不能脱离容差直接排序成精度。分类组合不作硬门槛，全部合格超差、异常与无输出保留。',
    ]
    categories = data.get('by_category', {})
    applicable = [value for key,value in categories.items() if key not in ('transparent','translucent')]
    if applicable:
        total = sum(value['n'] for value in applicable)
        numeric = sum(value['highest']['numeric'] for value in applicable)
        notes.append(f"原始全帧最高高度覆盖率{percent(primary['highest']['coverage'])}需结合输出边界：透明/半透明相位链路不提供该字段；适用几何链路最高数值为{numeric}/{total}帧，不能把相位不适用字段误述为漂浮物漏输出。")
    worst = sorted(categories.items(),key=lambda item:(item[1]['functional_pass'] is None,item[1]['functional_pass'] or 0))[:3]
    if worst:
        labels=data.get('category_names',{})
        notes.append('本次功能通过最低子类：'+'；'.join(
            f"{labels.get(key,key)} {percent(value['functional_pass'])}（液深MAE {number(value['liquid']['mae_mm'])}mm、最高MAE {number(value['highest']['mae_mm'])}mm）"
            for key,value in worst)+'。此排序仅描述本批预声明分布，不代表未测试环境风险顺序。')
    anchor=data.get('calibration_anchor_audit',{}).get('by_variant',{}).get('operator_noise')
    phase=data.get('phase_prior_annotation_audit',{}).get('by_variant',{}).get('operator_noise')
    if anchor and phase:
        notes.append(f"实际输入标定审计按唯一安装/标注组去重：底面anchor {anchor['n_records']}个，绝对MAE/P95/最大{number(anchor['mae_mm'])}/{number(anchor['p95_abs_mm'])}/{number(anchor['max_abs_mm'])}mm，越预算{anchor['over_allowed_limit']}个；非零相位先验{phase['n_error_values']}个标注量，绝对MAE/最大{number(phase['mae_mm'])}/{number(phase['max_abs_mm'])}mm，越预算{phase['over_allowed_limit']}个。其他几何量以协议及独立记录为准。")
    association=data.get('paired_comparison',{}).get('anchor_sensitivity',{}).get('by_domain',{})
    if association:
        notes.append('anchor与液深估计变化Δ=operator−ideal的安装等权关联：'+'；'.join(
            f"{key}: r={number(item['installation_balanced']['correlation'],4)}, slope={number(item['installation_balanced']['slope_mm_per_mm'],4)}mm/mm"
            for key,item in association.items())+'。其他尺寸/标注扰动同时存在，仅作描述性关联，不给显著性或单因素因果结论。')
    notes += [
        f"稳态计算P95 {number(primary['latency']['p95_ms'],1)}ms、最大{number(primary['latency']['max_ms'],1)}ms，严格超过500ms共{primary['latency']['over_500']}帧；排除初始化、渲染、磁盘、网络和界面，不是端到端硬实时保证。",
        'RGB低照/饱和审计不是lux或实测红外响应；透明相位仍为独立解析代理。同安装关联、部分输入复用和独立先验假设均需保留，不能直接声明现场工业精度。',
    ]
    return notes


def self_test():
    stat=dict(n=1,liquid=dict(mae_mm=2.,coverage=1.),highest=dict(coverage=0.),liquid_pass=1.,functional_pass=1.,
              actual_liquid_height_range_m=[.1,9.],latency=dict(p95_ms=10.,max_ms=20.,over_500=0))
    fixture=dict(evaluation_revision=REVISION,summary=stat,by_calibration_variant={'ideal':{'summary':stat}},
                 evaluation_provenance=dict(reused_input_images=1,new_input_images=0,new_measurement_runs=2,reused_measurement_runs=0),
                 depth_sampling_audit=dict(overall_counts=[1]+[0]*8,quotas_met=False,camera_bottom_actual_range_m=[.9,10.]),
                 qualification={'all':dict(new_observed_attempts=0,new_qualified=0,new_skipped=0,new_candidate_qualification_fraction=None)})
    text='\n'.join(derive(fixture))
    assert '固定标定协议' in text and '100mm' not in text and '浅液位加权' not in text
    json.dumps(derive(fixture),ensure_ascii=False,allow_nan=False)
    print('CONCLUSIONS_SELF_TEST_OK')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('LIQUID_EVAL_ROOT',str(Path(__file__).resolve().parent))))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    root=args.root.resolve();path=root/'performance.json';data=json.loads(path.read_text(encoding='utf-8'))
    notes=derive(data);data['failure_analysis']=notes
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    report=root/data.get('report_file','正式ToF_V14_0p1-9m_固定10mm标定性能报告_2026-10-21.rst')
    if report.parent.resolve()!=root:raise ValueError('Report must remain in output root')
    base=report.read_text(encoding='utf-8').split(MARKER)[0]
    report.write_text(base+MARKER+'\n本轮结果解释\n========================\n\n'+'\n\n'.join(notes)+'\n',encoding='utf-8')
    print('CONCLUSIONS_RECORDED',len(notes))


if __name__=='__main__':main()
