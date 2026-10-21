"""Summarize the frozen V14 holdout/replay; stdlib only, no model imports.

Run after capture and measurement have completed. Input eligibility is mandatory
and precedes measurement; this script never selects frames by their errors.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


COMPOSITES = (3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
NAMES = {
    'ordinary': '普通漫反射',
    'low_light_low': '低照·低', 'low_light_medium': '低照·中',
    'low_light_strong': '低照·强',
    'glare_low': '眩光·低', 'glare_medium': '眩光·中',
    'glare_strong': '眩光·强',
    'uneven_surface': '不平整液面', 'floating_objects': '漂浮物',
    'loss_low': '液面缺失0–1/6', 'loss_medium': '液面缺失1/6–1/3',
    'loss_high': '液面缺失1/3–1/2',
    'transparent': '透明均匀液体', 'translucent': '半透明均匀液体',
}
for _bits in COMPOSITES:
    NAMES[f'compound_{_bits:02d}'] = '+'.join(
        name for bit, name in ((1, '低照'), (2, '眩光'), (4, '波面'), (8, '漂浮物'))
        if _bits & bit)
ABS_BINS = (('≤5mm', .005), ('5–20mm', .020), ('20–50mm', .050),
            ('50–100mm', .100), ('100–500mm', .500), ('>500mm', math.inf))
REL_BINS = (('≤1%', .01), ('1–2%', .02), ('2–5%', .05),
            ('5–10%', .10), ('10–20%', .20), ('>20%', math.inf))
SCENE_FAMILIES = {
    'ordinary': ('普通漫反射', ('ordinary',)),
    'transparent': ('透明均匀液体', ('transparent',)),
    'translucent': ('半透明均匀液体', ('translucent',)),
    'low_light': ('低照', ('low_light_low', 'low_light_medium', 'low_light_strong')),
    'glare': ('眩光', ('glare_low', 'glare_medium', 'glare_strong')),
    'uneven_surface': ('不平整液面', ('uneven_surface',)),
    'floating_objects': ('漂浮物', ('floating_objects',)),
    'partial_depth_loss': ('部分深度缺失', ('loss_low', 'loss_medium', 'loss_high')),
}
FIXED_REVISION = 'range_0p1_9m_fixed_10mm'
DEPTH_BANDS = [(.1, 1.)] + [(float(index), float(index + 1)) for index in range(1, 9)]
DEPTH_BAND_QUOTAS = [12] + [11] * 8


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values, p):
    """Linear interpolation, matching NumPy's default percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def wilson(successes, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    phat = successes / total
    den = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / den
    half = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / den
    return [max(0., center - half), min(1., center + half)]


def histogram(values, bins):
    result = {label: {'n': 0, 'fraction': None} for label, _ in bins}
    for value in values:
        for label, upper in bins:
            if value <= upper:
                result[label]['n'] += 1
                break
    for item in result.values():
        item['fraction'] = item['n'] / len(values) if values else None
    return result


def intensity_stats(candidates, field):
    values = [finite(item.get(field)) for item in candidates]
    present = [value for value in values if value is not None]
    bounds = (.15, .4, .7, .8, .9, 1.) if field == 'darkness' else (.1, .35, .65, .7, .8, .9, 1.)
    fixed_bins = []
    lower = 0.
    for index, upper in enumerate(bounds):
        label = f'[{lower:g},{upper:g}]' if index == 0 else f'({lower:g},{upper:g}]'
        fixed_bins.append((label, upper))
        lower = upper
    fixed_bins.append(('>1', math.inf))
    return dict(n=len(candidates), observed=len(present), missing=len(candidates) - len(present),
                min=min(present) if present else None, max=max(present) if present else None,
                p05=quantile(present, .05), p25=quantile(present, .25), p50=quantile(present, .5),
                p75=quantile(present, .75), p95=quantile(present, .95),
                bins=histogram(present, fixed_bins),
                bin_fraction_denominator='observed_control_values_not_missing_candidates')


def prediction(row, output):
    value = row.get(output)
    if isinstance(value, dict):
        for key in ('height_m', 'value_m', 'value', 'liquid_height_m', 'highest_height_m'):
            if key in value:
                return finite(value[key])
        return None
    return finite(value)


def truth(row, output):
    data = row.get('truth', {})
    for key in (f'{output}_height_m', output, f'{output}_m'):
        if key in data:
            value = finite(data[key])
            if value is not None:
                return value
    raise ValueError(f'Missing finite {output} truth: {row.get("frame")}')


def eligible(row):
    data = row.get('qualification', {})
    if type(data.get('eligible')) is not bool:
        raise ValueError(f'Missing explicit Boolean qualification.eligible: {row.get("frame")}')
    if not data['eligible'] and row.get('inference_executed') is True:
        raise ValueError(f'Ineligible input was inferred: {row.get("frame")}')
    return data['eligible']


def tags(row, predicted=False):
    if predicted:
        values = (row.get('classification') or {}).get('tags', [])
    else:
        values = row.get('target_tags', [])
    if isinstance(values, str):
        values = [values]
    return set(values or [])


def tolerance(row, output):
    actual = abs(truth(row, output))
    if output == 'highest' or 'uneven_surface' in tags(row) or int(row.get('bits', 0)) & 4:
        return max(.050, .05 * actual)
    return max(.020, .02 * actual)


def passed(row, output):
    value = prediction(row, output)
    return value is not None and abs(value - truth(row, output)) <= tolerance(row, output)


def needs_highest(row):
    return bool(tags(row) & {'uneven_surface', 'floating_objects'} or int(row.get('bits', 0)) & 12)


def functional(row):
    return passed(row, 'liquid') and (not needs_highest(row) or passed(row, 'highest'))


def output_stats(rows, output):
    errors, relative = [], []
    zero_truth = 0
    for row in rows:
        actual = truth(row, output)
        value = prediction(row, output)
        if value is None:
            continue
        error = value - actual
        errors.append(error)
        if actual == 0:
            zero_truth += 1
        else:
            relative.append(abs(error) / abs(actual))
    absolute = [abs(value) for value in errors]
    return dict(numeric=len(errors), missing=len(rows) - len(errors),
                coverage=len(errors) / len(rows) if rows else None,
                mae_mm=sum(absolute) * 1000 / len(errors) if errors else None,
                p95_mm=quantile(absolute, .95) * 1000 if errors else None,
                max_mm=max(absolute) * 1000 if errors else None,
                bias_mm=sum(errors) * 1000 / len(errors) if errors else None,
                absolute_bins=histogram(absolute, ABS_BINS),
                relative_bins=histogram(relative, REL_BINS), relative_valid=len(relative),
                relative_undefined=zero_truth,
                absrel=sum(relative) / len(relative) if relative else None)


def timing(rows, key):
    values = [finite(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return dict(n=len(values), missing=len(rows) - len(values),
                p50_ms=quantile(values, .5), p95_ms=quantile(values, .95),
                max_ms=max(values) if values else None,
                over_500=sum(value > 500 for value in values),
                under_500_fraction=sum(value < 500 for value in values) / len(rows) if rows else None)


def installation_key(row):
    for value in (row.get('installation_id'), row.get('group'), row.get('controls', {}).get('installation_id')):
        if value is not None:
            return str(value)
    return None


def summarize(rows):
    input_rows = len(rows)
    rows = [row for row in rows if eligible(row)]
    n = len(rows)
    counts = dict(liquid_pass=sum(passed(row, 'liquid') for row in rows),
                  highest_pass=sum(passed(row, 'highest') for row in rows),
                  functional_pass=sum(functional(row) for row in rows),
                  strict_two_height=sum(passed(row, 'liquid') and passed(row, 'highest') for row in rows))
    installations = [installation_key(row) for row in rows]
    known_installations = Counter(key for key in installations if key is not None)
    descriptive = any(row.get('calibration_variant') is not None for row in rows) or any(
        count > 1 for count in known_installations.values())
    result = dict(n=n, input_rows=input_rows, ineligible_rows=input_rows - n, pass_counts=counts,
                  wilson95={key: [None, None] if descriptive else wilson(value, n) for key, value in counts.items()},
                  uncertainty_method=('descriptive_reused_installations_no_frame_independence_CI' if descriptive
                                      else 'frame_Wilson95_descriptive_shared_simulator_caveat'),
                  n_installations=len(known_installations), installation_unknown=sum(key is None for key in installations),
                  max_frames_per_installation=max(known_installations.values(), default=0),
                  liquid=output_stats(rows, 'liquid'), highest=output_stats(rows, 'highest'),
                  latency=timing(rows, 'wall_ms'),
                  initialization=timing([dict(row, init_ms=row.get('init_ms', row.get('initialization_ms')))
                                         for row in rows], 'init_ms'),
                  exact_tags=sum(tags(row) == tags(row, True) for row in rows) / n if n else None,
                  wrong_tags_but_functional=sum(tags(row) != tags(row, True) and functional(row) for row in rows),
                  functional_and_under_500=sum(functional(row) and finite(row.get('wall_ms')) is not None
                                              and finite(row['wall_ms']) < 500 for row in rows) / n if n else None,
                  failed_functional_frames=n - counts['functional_pass'],
                  exception_frames=sum(bool(row.get('error')) for row in rows))
    result.update({key: value / n if n else None for key, value in counts.items()})
    result['relative_bins'] = result['liquid']['relative_bins']
    actual_heights = [truth(row, 'liquid') for row in rows]
    result['actual_liquid_height_range_m'] = [min(actual_heights), max(actual_heights)] if actual_heights else [None, None]
    return result


def capture_disposition(root, old_root, rows, diverse=False):
    cohorts = {}
    intensity_observations = {}
    candidate_parameters = []
    row_lookup = {(row['cohort'], str(row['frame'])): row for row in rows}
    specifications = (('single', root, range(14)),
                      ('composite', root if diverse else old_root, range(14, 25) if diverse else COMPOSITES))
    for cohort, folder, indices in specifications:
        counts = Counter()
        reasons = Counter()
        files = []
        for index in indices:
            path = folder / f'capture-{index:02d}.json'
            state = json.loads(path.read_text(encoding='utf-8'))
            files.append(str(path))
            counts['attempts'] += int(state['attempts'])
            counts['capture_qualified'] += len(state.get('qualified', []))
            counts['capture_skipped'] += len(state.get('skipped', []))
            counts['pending_or_inflight'] += len(state.get('inflight', []))
            for skipped in state.get('skipped', []):
                reasons.update(skipped.get('qualification', {}).get('reasons', []))
            category = state.get('category') if cohort == 'single' or diverse else f'compound_{index:02d}'
            candidates = []
            for skipped in state.get('skipped', []):
                candidates.append(dict(controls=skipped.get('request', {}), status='rejected'))
            for captured in state.get('qualified', []):
                measured = row_lookup.get((cohort, str(captured['frame'])))
                candidates.append(dict(controls=measured.get('controls', {}) if measured else {},
                                       status=('eligible' if eligible(measured) else 'rejected') if measured else 'unresolved'))
            candidate_parameters.extend(dict(item, category=category, cohort=cohort) for item in candidates)
            active_fields = [field for bit, field in ((1, 'darkness'), (2, 'glare'))
                             if int(state.get('bits', index if cohort == 'composite' else 0)) & bit]
            if active_fields:
                intensity_observations[category] = {
                    field: {status: intensity_stats([item['controls'] for item in candidates
                                                    if status == 'all_candidates' or item['status'] == status], field)
                            for status in ('all_candidates', 'eligible', 'rejected', 'unresolved')}
                    for field in active_fields}
        subset = [row for row in rows if row['cohort'] == cohort]
        qualified = sum(eligible(row) for row in subset)
        deferred_reasons = Counter()
        for row in subset:
            if not eligible(row):
                deferred_reasons.update(row.get('qualification', {}).get('reasons', []))
        cohorts[cohort] = dict(counts, measured_rows=len(subset), measured_qualified=qualified,
                               measured_ineligible=len(subset) - qualified,
                               skipped_reasons=dict(reasons), deferred_phase_or_input_reasons=dict(deferred_reasons),
                               eligible_candidate_fraction=qualified / counts['attempts'] if counts['attempts'] else None,
                               input_skipped=counts['capture_skipped'] + len(subset) - qualified,
                               capture_files=files,
                               disposition_balances=(counts['attempts'] == counts['capture_qualified'] + counts['capture_skipped']
                                                     + counts['pending_or_inflight']))
    all_counts = {}
    for key in ('attempts', 'capture_qualified', 'capture_skipped', 'pending_or_inflight',
                'measured_rows', 'measured_qualified', 'measured_ineligible', 'input_skipped'):
        all_counts[key] = sum(item[key] for item in cohorts.values())
    all_counts['eligible_candidate_fraction'] = all_counts['measured_qualified'] / all_counts['attempts'] if all_counts['attempts'] else None
    all_counts['skipped_reasons'] = dict(sum((Counter(item['skipped_reasons']) for item in cohorts.values()), Counter()))
    return dict(by_cohort=cohorts, all=all_counts, intensity_observations=intensity_observations,
                parameter_observations=parameter_observations(candidate_parameters),
                count_unit='unique_input_images_not_calibration_variant_runs',
                reasons_can_overlap=True, qualification_uses_simulator_visibility=True,
                deployed_observability_oracle=False)


def parameter_observations(candidates):
    def subset_stats(subset):
        result = dict(n=len(subset), numeric={}, categorical={})
        controls = [item['controls'] for item in subset]
        keys = ('bottom_m', 'capacity_m', 'radius_m', 'aspect', 'horizontal_fov_deg', 'height_m',
                'sensor_severity', 'injected_common_bias_m', 'spatial_rms_m', 'darkness', 'glare',
                'wave_m', 'float_count', 'loss', 'camera_offset_m', 'height_to_bottom_ratio')
        for key in keys:
            values = []
            for control in controls:
                value = control.get(key)
                if key == 'camera_offset_m':
                    xy = control.get('camera_xy_m')
                    value = math.hypot(*xy) if isinstance(xy, (list, tuple)) and len(xy) == 2 else None
                elif key == 'height_to_bottom_ratio':
                    b, h = finite(control.get('bottom_m')), finite(control.get('height_m'))
                    value = h / b if b is not None and b > 0 and h is not None else None
                value = finite(value)
                if value is not None:
                    values.append(value)
            result['numeric'][key] = dict(observed=len(values), missing=len(controls) - len(values),
                                          min=min(values) if values else None, max=max(values) if values else None,
                                          p05=quantile(values, .05), p25=quantile(values, .25),
                                          p50=quantile(values, .5), p75=quantile(values, .75), p95=quantile(values, .95))
        for key in ('shape', 'noise_family', 'camera_mount', 'material', 'repeat'):
            counts = Counter(str(control[key]) for control in controls if control.get(key) is not None)
            result['categorical'][key] = dict(counts=counts, observed=sum(counts.values()),
                                              missing=len(controls) - sum(counts.values()))
        ids = {str(control['installation_id']) for control in controls if control.get('installation_id') is not None}
        result['n_unique_installations'] = len(ids)
        return result
    return {status: subset_stats([item for item in candidates if status == 'all_candidates' or item['status'] == status])
            for status in ('all_candidates', 'eligible', 'rejected', 'unresolved')}


def condition_groups(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not eligible(row):
            continue
        q = row['qualification']
        for kind, field in (('loss', 'liquid_invalid_fraction'), ('phase_loss', 'phase_invalid_fraction')):
            value = finite(q.get(field))
            if value is not None:
                name = '[0,1/6]' if value <= 1/6 else '(1/6,1/3]' if value <= 1/3 else '(1/3,1/2]' if value <= .5 else '>1/2'
                grouped[kind][name].append(row)
        controls = row.get('controls', {})
        for kind, active, edges in (('darkness', 'low_light', (.4, .7)), ('glare', 'glare', (.35, .65))):
            value = finite(controls.get(kind))
            if value is not None and active in tags(row):
                name = 'low' if value < edges[0] else 'medium' if value < edges[1] else 'strong'
                grouped[kind][name].append(row)
    return {kind: {name: summarize(subset) for name, subset in groups.items()} for kind, groups in grouped.items()}


def diagnostics(rows):
    rows = [row for row in rows if eligible(row)]
    result = {}
    for tag in ('low_light', 'glare', 'uneven_surface', 'floating_objects'):
        tp = sum(tag in tags(row) and tag in tags(row, True) for row in rows)
        fp = sum(tag not in tags(row) and tag in tags(row, True) for row in rows)
        fn = sum(tag in tags(row) and tag not in tags(row, True) for row in rows)
        result[tag] = dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp) if tp + fp else None,
                           recall=tp / (tp + fn) if tp + fn else None)
    return result


def tail_record(row, output):
    predicted = prediction(row, output)
    actual = truth(row, output)
    error = abs(predicted - actual) if predicted is not None else None
    return dict(frame=row.get('frame'), cohort=row['cohort'], category=row['category'], output=output,
                true_m=actual, predicted_m=predicted, absolute_error_mm=error * 1000 if error is not None else None,
                relative_error=error / abs(actual) if error is not None and actual != 0 else None,
                tolerance_mm=tolerance(row, output) * 1000, passed=passed(row, output),
                wall_ms=finite(row.get('wall_ms')), target_tags=sorted(tags(row)),
                predicted_tags=sorted(tags(row, True)), qualification=row['qualification'], error=row.get('error'))


def build(rows, protocol, disposition):
    for row in rows:
        eligible(row)
        if row.get('cohort') not in ('single', 'composite') or not row.get('category'):
            raise ValueError(f'Missing valid cohort/category: {row.get("frame")}')
        if row.get('sensor', 'tof') != 'tof':
            raise ValueError('This report is ToF only')
    unique = [(row['cohort'], str(row.get('frame'))) for row in rows]
    if len(set(unique)) != len(unique):
        raise ValueError('Duplicate measurement frame in the same cohort')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['category']].append(row)
    category_meta = {}
    for name, subset in grouped.items():
        cohorts = set(row['cohort'] for row in subset)
        if len(cohorts) != 1:
            raise ValueError(f'Category crosses cohorts: {name}')
        intensity = name.rsplit('_', 1)[-1] if name.startswith(('low_light_', 'glare_', 'loss_')) else None
        kind = ('compound' if name.startswith('compound_') else 'low_light' if name.startswith('low_light_')
                else 'glare' if name.startswith('glare_') else 'loss' if name.startswith('loss_') else name)
        category_meta[name] = dict(cohort=next(iter(cohorts)), label=NAMES.get(name, name),
                                   kind=kind, intensity=intensity)
    accepted = [row for row in rows if eligible(row)]
    largest = {}
    for output in ('liquid', 'highest'):
        present = [row for row in accepted if prediction(row, output) is not None]
        present.sort(key=lambda row: abs(prediction(row, output) - truth(row, output)), reverse=True)
        largest[output] = [tail_record(row, output) for row in present[:15]]
    category_summaries = {name: summarize(subset) for name, subset in grouped.items()}
    for name, stats in category_summaries.items():
        stats['observed_intensity'] = disposition.get('intensity_observations', {}).get(name, {})
    scene_families = {
        name: summarize([row for row in rows if row['cohort'] == 'single' and row['category'] in categories])
        for name, (label, categories) in SCENE_FAMILIES.items()}
    family_meta = {name: dict(label=label, cohort='single', categories=list(categories),
                              aggregation='pooled_frame_records_not_average_of_subgroup_metrics')
                   for name, (label, categories) in SCENE_FAMILIES.items()}
    return dict(version='formal_tof_v14_20261021', sensor='tof',
                generated_at=datetime.now(timezone.utc).isoformat(), protocol=protocol,
                summary=summarize(rows),
                by_cohort={cohort: summarize([row for row in rows if row['cohort'] == cohort])
                           for cohort in ('single', 'composite')},
                by_category=category_summaries,
                by_scene_family=scene_families, scene_family_meta=family_meta,
                scene_family_names={name: item[0] for name, item in SCENE_FAMILIES.items()},
                category_names=NAMES, category_labels=NAMES, category_meta=category_meta, qualification=disposition,
                intensity_observations=disposition.get('intensity_observations', {}),
                actual_condition_bins=condition_groups(rows), largest_errors=largest,
                classification_diagnostics=diagnostics(rows),
                failed_frames=[dict(frame=row.get('frame'), cohort=row['cohort'], category=row['category'],
                                    liquid=tail_record(row, 'liquid'), highest=tail_record(row, 'highest'))
                               for row in accepted if not functional(row)],
                over_500ms_frames=[dict(frame=row.get('frame'), cohort=row['cohort'], category=row['category'],
                                        wall_ms=finite(row.get('wall_ms')))
                                   for row in accepted if finite(row.get('wall_ms')) is not None and row['wall_ms'] > 500],
                source_digest={key: protocol.get(key) for key in ('git_commit', 'source_sha256', 'model_sha256')},
                exact_tags_are_hard_gate=False, no_error_based_frame_removal=True,
                field_metrology_qualified=False,
                timing_excludes=['initialization', 'Blender_render', 'disk_load', 'network', 'browser'],
                scope_notes=[
                    '新增单场景为独立采样；复合场景为既有1103个独立留出帧在当前冻结正式代码上的重放，不是又新增1103个独立样本。',
                    '普通漫反射复合范围只包括低照、眩光、波面、漂浮物及可见液面0–50%缺失；透明/半透明采用均匀液体、平整无遮挡液面的独立五帧相位先验链路。',
                    '透明/半透明使用模拟原始AMCW相关量及空槽+四帧已知液深先验；普通几何链路使用几何/深度回波代理，二者不是同一传感器数字孪生。',
                    '物理材料类型confirmed_material及透明链路的平整无遮挡确认是安装先验；本实验从预设场景传入，现场应由操作人员或独立信息确认。视觉头仅识别低照、眩光、波面、漂浮物四种非互斥现象，不在本实验中自动识别普通/透明/半透明材料或验证其链路选择准确率。',
                    '空槽几何、相机内参/位姿、容器米制尺寸及五个标定液深均为理想仿真先验；本批未加入现场标定误差、先验标签误差或相机移动后配准误差，不能把毫米级结果解释为真实部署包含标定误差后的精度。',
                    '透明/半透明每组安装中的折射率、反射振幅和散射等参数在五帧先验与测量帧之间保持一致；当前测试液位处于约0.06–0.86容器容量，而先验覆盖0–0.95容量，主要检验插值，不验证空槽测量、超先验外推或光学参数随时间漂移。',
                    'AMCW参数是可控物理代理而非经实测标定的工业ToF相机参数档案。3频率×4相位×48×64相关量与480×270 RGB/深度渲染代理是分开的输入生成过程，不代表一个完整联合成像的工业传感器数字孪生。',
                    '合格输入筛选依赖模拟可见性真值，只定义实验适用域，不是部署时可获得的真值门控；不按测量误差筛帧。',
                    '当前数据是固定顶部相机、单一圆形容器几何和同一物理代理域，不能推断真实工业总体或完整0–9m均匀量程表现。',
                    '分类组合完全正确不作为硬门槛；数值合格不证明分类问题已解决，模型置信度也不等于计量置信度。',
                    '本批是当前代码的单场景评估与同版本复合集重放；不把旧V1等不同原始相位参数域、协议或统计口径的历史结果作为直接优劣比较，比较改进须使用相同输入和协议的配对测试。',
                    'Wilson95%区间仅描述有限帧的二项比例，未校正共享仿真机制等相关性；100/100不是总体100%的证明。'])


def image_key(row):
    return row['cohort'], row['category'], str(row['frame'])


def paired_summary(pairs):
    complete = [pair for pair in pairs if 'ideal' in pair and 'operator_noise' in pair]
    qualified = [pair for pair in complete if eligible(pair['ideal'])]
    transitions = Counter()
    for pair in qualified:
        a, b = functional(pair['ideal']), functional(pair['operator_noise'])
        transitions['both_pass' if a and b else 'ideal_only_pass' if a else 'operator_only_pass' if b else 'both_fail'] += 1
    result = dict(n_images=len(pairs), n_complete_pairs=len(complete), n_qualified_pairs=len(qualified),
                  n_installations=len({installation_key(pair['ideal']) for pair in qualified
                                       if installation_key(pair['ideal']) is not None}),
                  functional_transitions={key: transitions[key] for key in ('both_pass', 'ideal_only_pass', 'operator_only_pass', 'both_fail')},
                  delta_functional_pass_pp=(transitions['operator_only_pass'] - transitions['ideal_only_pass']) * 100 / len(qualified) if qualified else None,
                  interpretation='operator_minus_ideal_same_image_descriptive_not_independent_frame_CI')
    for output in ('liquid', 'highest'):
        ideal_errors, operator_errors, deltas = [], [], []
        missing = Counter()
        for pair in qualified:
            a, b = prediction(pair['ideal'], output), prediction(pair['operator_noise'], output)
            if a is None or b is None:
                missing['both_missing' if a is None and b is None else 'ideal_only_missing' if a is None else 'operator_only_missing'] += 1
                continue
            actual = truth(pair['ideal'], output)
            ae, be = abs(a - actual) * 1000, abs(b - actual) * 1000
            ideal_errors.append(ae)
            operator_errors.append(be)
            deltas.append(be - ae)
        result[output] = dict(paired_numeric=len(deltas),
                              ideal_mae_mm=sum(ideal_errors) / len(deltas) if deltas else None,
                              operator_mae_mm=sum(operator_errors) / len(deltas) if deltas else None,
                              mean_delta_abs_error_mm=sum(deltas) / len(deltas) if deltas else None,
                              p50_delta_abs_error_mm=quantile(deltas, .5), p95_delta_abs_error_mm=quantile(deltas, .95),
                              worse=sum(value > 1e-9 for value in deltas), better=sum(value < -1e-9 for value in deltas),
                              same=sum(abs(value) <= 1e-9 for value in deltas),
                              missing_transitions={key: missing[key] for key in ('both_missing', 'ideal_only_missing', 'operator_only_missing')})
    return result


def error_audit_stats(records):
    """Each record is one physical calibration quantity, never one wet frame."""
    errors = [record['error_mm'] for record in records]
    absolute = [abs(value) for value in errors]
    with_limit = [record for record in records if finite(record.get('limit_mm')) is not None]
    ratios = [abs(record['error_mm']) / record['limit_mm'] for record in with_limit if record['limit_mm'] > 0]
    return dict(n_records=len(records), bias_mm=sum(errors) / len(errors) if errors else None,
                mae_mm=sum(absolute) / len(absolute) if absolute else None,
                p95_abs_mm=quantile(absolute, .95), max_abs_mm=max(absolute) if absolute else None,
                min_signed_mm=min(errors) if errors else None, max_signed_mm=max(errors) if errors else None,
                p50_signed_mm=quantile(errors, .5),
                absolute_bins=histogram([value / 1000 for value in absolute], ABS_BINS),
                n_limit_known=len(with_limit),
                over_allowed_limit=sum(abs(record['error_mm']) > record['limit_mm'] + 1e-6 for record in with_limit),
                max_abs_to_limit=max(ratios) if ratios else None,
                annotation_attempts_max=max((record.get('annotation_attempt', 1) for record in records), default=None))


def calibration_audits(rows, protocol):
    anchors, phase_sets = {}, {}
    expected = defaultdict(set)
    fractions = protocol.get('phase_parameters', {}).get('priors_fraction', [0., .1, .35, .65, .95])[1:]
    for row in rows:
        variant = row['calibration_variant']
        installation = installation_key(row)
        if installation is None:
            if row.get('calibration_audit') or row.get('phase_prior_annotation'):
                raise ValueError('Cannot deduplicate a calibration audit without installation ID')
            continue
        expected[variant].add(installation)
        audit = row.get('calibration_audit')
        if audit is not None:
            error = finite(audit.get('anchor_error_mm'))
            if error is None:
                raise ValueError(f'Invalid anchor error: {installation}/{variant}')
            record = dict(installation_id=installation, variant=variant, error_mm=error,
                          limit_mm=finite(audit.get('anchor_error_limit_mm')),
                          metric_anchor_m=finite(audit.get('metric_anchor_m')),
                          annotation_attempt=audit.get('attempt', 1), source=audit.get('source'))
            key = installation, variant
            if key in anchors and record != anchors[key]:
                raise ValueError(f'Nonidentical repeated installation anchor audit: {key}')
            anchors[key] = record
        annotation = row.get('phase_prior_annotation')
        if annotation is None:
            continue
        errors = annotation.get('errors_m', [])
        if len(errors) != 4 or any(finite(value) is None for value in errors):
            raise ValueError(f'Invalid four-phase-prior errors: {installation}/{variant}')
        material = row.get('controls', {}).get('material', row['category'])
        capacity = finite(row.get('controls', {}).get('capacity_m'))
        prior_records = []
        for index, error in enumerate(errors):
            known = capacity * fractions[index] if capacity is not None and len(fractions) == 4 else None
            declared_limit = protocol.get('operator_calibration', {}).get('error_limit_m')
            fixed_limit = finite(declared_limit)
            limit = fixed_limit * 1000 if fixed_limit is not None else (
                max(.010, .01 * abs(known)) * 1000 if known is not None else None)
            prior_records.append(dict(installation_id=installation, variant=variant, material=material,
                                      prior_index=index + 1, error_mm=float(error) * 1000, limit_mm=limit,
                                      annotation_attempt=annotation.get('annotation_attempt', 1)))
        key = installation, variant, material
        if key in phase_sets and prior_records != phase_sets[key]:
            raise ValueError(f'Nonidentical repeated phase-prior annotation: {key}')
        phase_sets[key] = prior_records
    anchor_result = dict(count_unit='unique_installation_variant', by_variant={}, records=list(anchors.values()))
    phase_result = dict(count_unit='unique_installation_variant_material_annotation',
                        by_variant={}, by_material={}, prior_index={},
                        note='Four nonzero annotation quantities per set; fixed physical-empty label is excluded; not frame weighted.')
    for variant in ('ideal', 'operator_noise'):
        subset = [record for (installation, v), record in anchors.items() if v == variant]
        stats = error_audit_stats(subset)
        stats.update(n_installations=len(subset), n_expected_installations=len(expected[variant]),
                     missing_installations=len(expected[variant]) - len(subset))
        anchor_result['by_variant'][variant] = stats
    def phase_stats(keys):
        records = [record for key in keys for record in phase_sets[key]]
        result = error_audit_stats(records)
        result.update(n_installations=len({key[0] for key in keys}), n_annotation_sets=len(keys), n_error_values=len(records))
        return result
    for variant in ('ideal', 'operator_noise'):
        phase_result['by_variant'][variant] = phase_stats([key for key in phase_sets if key[1] == variant])
    for material in sorted({key[2] for key in phase_sets}):
        phase_result['by_material'][material] = {variant: phase_stats([key for key in phase_sets if key[1:] == (variant, material)])
                                                 for variant in ('ideal', 'operator_noise')}
    for index in range(1, 5):
        phase_result['prior_index'][str(index)] = {
            variant: error_audit_stats([record for key, records in phase_sets.items() if key[1] == variant
                                       for record in records if record['prior_index'] == index])
            for variant in ('ideal', 'operator_noise')}
    return anchor_result, phase_result


def descriptive_regression(points):
    """No significance inference: inputs may share simulated installations."""
    n = len(points)
    if not n:
        return dict(n=0, correlation=None, slope_mm_per_mm=None, intercept_mm=None, r_squared=None,
                    mean_anchor_error_mm=None, mean_estimate_change_mm=None, mae_estimate_change_mm=None)
    xmean = sum(point[0] for point in points) / n
    ymean = sum(point[1] for point in points) / n
    xx = sum((point[0] - xmean) ** 2 for point in points)
    yy = sum((point[1] - ymean) ** 2 for point in points)
    xy = sum((point[0] - xmean) * (point[1] - ymean) for point in points)
    correlation = max(-1., min(1., xy / math.sqrt(xx * yy))) if xx > 0 and yy > 0 else None
    slope = xy / xx if xx > 0 else None
    return dict(n=n, correlation=correlation, slope_mm_per_mm=slope,
                intercept_mm=ymean - slope * xmean if slope is not None else None,
                r_squared=correlation ** 2 if correlation is not None else None,
                mean_anchor_error_mm=xmean, mean_estimate_change_mm=ymean,
                mae_estimate_change_mm=sum(abs(point[1]) for point in points) / n)


def anchor_sensitivity_summary(pairs):
    qualified = [pair for pair in pairs if 'ideal' in pair and 'operator_noise' in pair and eligible(pair['ideal'])]
    groups = defaultdict(list)
    missing_anchor_or_id = missing_numeric = 0
    for pair in qualified:
        noisy, ideal = pair['operator_noise'], pair['ideal']
        a, b = prediction(noisy, 'liquid'), prediction(ideal, 'liquid')
        if a is None or b is None:
            missing_numeric += 1
            continue
        anchor_error = finite((noisy.get('calibration_audit') or {}).get('anchor_error_mm'))
        installation = installation_key(noisy)
        if anchor_error is None or installation is None:
            missing_anchor_or_id += 1
            continue
        groups[installation].append((anchor_error, (a - b) * 1000))
    all_points = [point for points in groups.values() for point in points]
    balanced_points = [(sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))
                       for points in groups.values()]
    return dict(n_qualified_pairs=len(qualified), n_numeric_with_anchor=len(all_points), n_installations=len(groups),
                missing_anchor_or_id=missing_anchor_or_id, missing_numeric=missing_numeric,
                frame_weighted=descriptive_regression(all_points),
                installation_balanced=descriptive_regression(balanced_points))


def build_diverse(rows, protocol, disposition, control=None):
    variants = {'ideal': [], 'operator_noise': []}
    pairs = {}
    for row in rows:
        variant = row.get('calibration_variant')
        if variant not in variants:
            raise ValueError(f'Unknown/missing calibration_variant: {variant}')
        variants[variant].append(row)
        pair = pairs.setdefault(image_key(row), {})
        if variant in pair:
            raise ValueError(f'Duplicate image/variant: {image_key(row)}, {variant}')
        pair[variant] = row
    for key, pair in pairs.items():
        if len(pair) != 2:
            continue
        a, b = pair['ideal'], pair['operator_noise']
        if eligible(a) != eligible(b):
            raise ValueError(f'Calibration variants disagree on raw input eligibility: {key}')
        if a.get('truth') != b.get('truth') or a.get('controls') != b.get('controls'):
            raise ValueError(f'Calibration pair does not have identical image controls/truth: {key}')
    data = build(variants['operator_noise'], protocol, disposition)
    variant_reports = {'operator_noise': data, 'ideal': build(variants['ideal'], protocol, disposition)}
    data['by_calibration_variant'] = {
        key: {field: value[field] for field in ('summary', 'by_cohort', 'by_scene_family', 'by_category',
                                               'largest_errors', 'failed_frames', 'classification_diagnostics')}
        for key, value in variant_reports.items()}
    data['paired_comparison'] = dict(
        summary=paired_summary(list(pairs.values())),
        by_category={category: paired_summary([pair for key, pair in pairs.items() if key[1] == category])
                     for category in NAMES},
        by_scene_family={name: paired_summary([pair for key, pair in pairs.items()
                                               if key[0] == 'single' and key[1] in categories])
                         for name, (label, categories) in SCENE_FAMILIES.items()})
    data['paired_comparison']['anchor_sensitivity'] = dict(
        predictor='operator_noise calibration_audit.anchor_error_mm',
        outcome='signed liquid estimate change: operator_noise minus ideal, in mm; NOT absolute-error change',
        interpretation='descriptive association only; other geometry/prior perturbations co-vary, no p-values, frame-independence CI or causal attribution',
        domain_names=dict(ordinary='普通几何单场景（含各单因素）', refraction='透明与半透明相位链路', composite='普通材料复合场景'),
        summary=anchor_sensitivity_summary(list(pairs.values())),
        by_domain={
            'ordinary': anchor_sensitivity_summary([pair for key, pair in pairs.items() if key[0] == 'single' and key[1] not in ('transparent', 'translucent')]),
            'refraction': anchor_sensitivity_summary([pair for key, pair in pairs.items() if key[1] in ('transparent', 'translucent')]),
            'composite': anchor_sensitivity_summary([pair for key, pair in pairs.items() if key[0] == 'composite'])},
        by_category={category: anchor_sensitivity_summary([pair for key, pair in pairs.items() if key[1] == category])
                     for category in NAMES})
    data['evaluation_domain'] = 'diverse_parameterized_sensitivity'
    data['primary_calibration_variant'] = 'operator_noise'
    data['measurement_runs'] = len(rows)
    data['unique_input_images'] = len(pairs)
    data['parameter_observations'] = disposition.get('parameter_observations', {})
    for field, control_key in (('by_noise_family', 'noise_family'), ('by_geometry_shape', 'shape')):
        data[field] = {
            variant: {group: summarize([row for row in subset if str(row.get('controls', {}).get(control_key, 'unknown')) == group])
                      for group in sorted({str(row.get('controls', {}).get(control_key, 'unknown')) for row in subset})}
            for variant, subset in variants.items()}
    data['noise_family_names'] = {str(index): name for index, name in enumerate(
        protocol.get('sensor_sensitivity', {}).get('families', ['base iid', 'plus spatial', 'plus frame common drift', 'plus both']))}
    data['geometry_shape_names'] = dict(cylindrical='圆柱', elliptical='椭圆', rectangular_tank='矩形槽', unknown='未记录')
    data['calibration_anchor_audit'], data['phase_prior_annotation_audit'] = calibration_audits(rows, protocol)
    data['capture_domain_audit'] = protocol.get('capture_domain_audit')
    data['capture_domain_audit_definitions'] = dict(
        pixel_mask='offline semantic==2: visible liquid only; excludes floating-object pixels',
        luminance_scale='OpenCV BGR2GRAY divided by255, display grayscale0..1; not radiometric luminance or lux',
        luminance_mean='mean grayscale of visible liquid pixels per frame',
        luminance_p95='95th percentile of visible liquid grayscale per frame',
        saturated_any_channel_fraction='fraction visible liquid pixels with any RGB/BGR channel>=250 on0..255 scale',
        underexposed_fraction='fraction visible liquid pixels with grayscale<0.04 on0..1 scale',
        liquid_visible_pixels='number visible liquid pixels in rendered frame',
        summary='min/median/p95/max across per-frame scalar values, not pooled-pixel distribution',
        eligibility='capture True/False status before inference; any deferred phase qualification is reported separately')
    data['calibration_parameter_protocol'] = {key: value for key, value in protocol.items()
                                               if any(word in key.lower() for word in ('calibrat', 'operator', 'perturb', 'noise', 'sensitivity'))}
    data['proportions_are_descriptive'] = True
    data['qualification']['comparable_to_narrow_qualification'] = False
    data['qualification']['additional_diverse_conditions'] = [
        '生成器来源标签origin==0的真实液面返回至少30点并满足空间跨度；不使用测量误差定义这些点。',
        '有漂浮物时最高实例须有至少3个真实返回，而非任意一个较低漂浮物可见即可。',
        '透明相位使用实际64×48中央像素裁剪和对应内参，空/湿共同可见支撑至少30点，另检查解析相位ROI资格。',
        '这些测试前声明的真值可见性/来源条件不作为模型输入，也不等价于现场可以直接获得的判断。']
    data['scope_notes'] = [
        '本报告主体为多样化新独立仿真输入的operator_noise版本；标定扰动是预先设定的参数化敏感性试验，不是真实工业相机准确率或现场误差分布的标定结果。',
        '主体标定误差按用户允许预算定义：对各被标定米制量D设绝对上限T=max(10mm,1%|D|)，使用零均值截断高斯，截断前尺度σ=T/3，最终误差限于±T；底面距离、独立几何尺寸及四张非零已知液深先验分别按其自身D计算，最终精确定义与参数记录以protocol.json为准，不沿用旧的小σ预算。',
        '首张物理空槽的液深标签固定为0，不再单独扰动其零液深；零基准误差由底面anchor承载。像素点击扰动0.3px与重力扰动0.001另列，不将它们包含在米制截断高斯T中；重力扰动的具体参数含义和单位以协议为准。',
        '这是允许标定预算下的敏感性评估：被标定量D可能是相机到底面距离或几何尺寸，并非当前浅液深H。其允许误差可能大于最终浅液深容差，例如D=10m时T=100mm、σ约33.3mm，而H=0.1m的液深容差仅20mm；不能将这种预算冲突自动归因于回波算法，也不能按测试误差删样或偷偷缩小标定扰动。',
        '14个单场景子类与11个复合组合均在本批重新采样；同一安装可能跨液位/场景复用，同一图像分别使用ideal与operator_noise两种先验运行。资格与样本数只按唯一图像计一次，不能将两次标定运行算成两张独立图像。',
        '主体保留operator_noise的所有合格输入输出，包括异常、缺值与超差；逐图与ideal配对比较。两组共享同一RGB/深度/相位输入，只改变安装标定先验，不把当前测量真值交给逆解器。',
        '样本复用安装和共同仿真机制形成相关性。本报告只给描述性帧比例，不使用逐帧独立Wilson区间宣称总体置信度；每类100帧不等于100个独立安装。',
        '容器、相机和噪声参数按预声明协议变化；采用生成器已实现的圆柱、椭圆和矩形几何，圆柱aspect=1，半径范围固定而不为构图缩成细管；不能仅凭更大的样本数宣称真实工业多样性。最终实际取值、几何与拒绝分布均另列。',
        '低照可见外观与主动ToF回波退化解耦；眩光采用局部饱和区域；噪声为测量帧公共漂移及空间相关随机扰动，按相机到液面距离而非到底面距离设定尺度，不是安装持久偏置，也不冒充已实测的商业ToF硬件参数。',
        '普通漫反射复合只限低照、眩光、波面、漂浮物与实际可见液面0–50%缺失；透明/半透明仍限制均匀液体、平整无遮挡和五帧相位先验，不测试跨物理边界的透明波面或漂浮物组合。',
        '材料类型及透明链路平整无遮挡确认是独立安装先验；视觉头只预测四类现象，不以本报告声称自动材料类型识别准确率。',
        'AMCW相关量仍为参数化物理代理，但48×64使用与内参匹配的实际中央像素裁剪而非缩放，空/湿共同可见支撑及解析相位ROI均单独检查。逆解器不直接获得真实光学参数；同安装物性在标定/测试之间固定，先验扰动不等同实测光学漂移。',
        '本批在基础有效点/跨度/缺失资格之外，预先要求来源origin==0的真实液面返回至少30点及相应跨度；有漂浮物时最高实例至少3个真实返回。来源/实例真值只用于测量前资格，不输入估计器；不按拟合残差或测量误差删样。',
        '多样域采用了更强的可观测性先验资格，不能将其资格通过比例直接与旧窄域比较，也不能把合格子集误差称为所有生成帧表现。这些离线真值条件不是现场可部署的可观测性判定器。',
        '窄域理想标定结果只作为不同测试域的历史对照，不能将域间MAE变化直接归因于算法优劣；所有量程声明以本批实际液深范围为准，不作完整0–9m均匀认证。',
        '分类组合不作为硬验收门槛；数值误差、输出覆盖和延时是主指标。本批未依据评估结果修改正式测量算法。']
    if control is not None:
        data['narrow_domain_control'] = {key: control.get(key) for key in
                                         ('version', 'summary', 'by_cohort', 'by_scene_family', 'by_category',
                                          'qualification', 'protocol', 'source_digest')}
        data['narrow_domain_control']['comparison_type'] = 'different_domain_unpaired_descriptive_control_not_algorithm_gain'
    return data


def depth_band_index(height):
    value = finite(height)
    if value is None or value < .1 - 1e-9 or value > 9. + 1e-9:
        return None
    for index, (lower, upper) in enumerate(DEPTH_BANDS):
        if value >= lower - 1e-9 and (value < upper if index < 8 else value <= upper + 1e-9):
            return index
    return None


def fixed_range_audit(rows, states):
    bands = [dict(index=index, lower_m=lower, upper_m=upper, upper_inclusive=index == 8,
                  label=f'[{lower:g},{upper:g}{"]" if index == 8 else ")"}', target_per_category=DEPTH_BAND_QUOTAS[index])
             for index, (lower, upper) in enumerate(DEPTH_BANDS)]
    selected = {(state['cohort'], str(item['frame'])): item for state in states for item in state.get('qualified', [])}
    by_category = {}
    out_of_range, declared_mismatches = [], []
    source_counts = Counter()
    source_identities = set()
    for name in NAMES:
        subset = [row for row in rows if row['category'] == name]
        counts = [0] * 9
        sources = Counter()
        for row in subset:
            band = depth_band_index(truth(row, 'liquid'))
            if band is None:
                out_of_range.append(dict(frame=row['frame'], actual_liquid_height_m=truth(row, 'liquid')))
            else:
                counts[band] += 1
            item = selected.get((row['cohort'], str(row['frame'])))
            if item is None:
                raise ValueError(f'Measured image absent from selected qualified manifest: {row["frame"]}')
            declared = item.get('depth_band')
            if declared is not None and declared != band:
                declared_mismatches.append(dict(frame=row['frame'], declared=declared, actual=band))
            source = item.get('source', 'unknown')
            sources[source] += 1
            source_counts[source] += 1
            original = str(item.get('source_frame') or row['frame'])
            source_key = source, original
            if source_key in source_identities:
                raise ValueError(f'Same original image selected more than once: {source_key}')
            source_identities.add(source_key)
        by_category[name] = dict(n=len(subset), counts=counts, target_counts=DEPTH_BAND_QUOTAS,
                                 quotas_met=all(count >= quota for count, quota in zip(counts, DEPTH_BAND_QUOTAS)),
                                 source_counts=dict(sources),
                                 n_installations=len({installation_key(row) for row in subset if installation_key(row) is not None}))
    overall_counts = [sum(item['counts'][index] for item in by_category.values()) for index in range(9)]
    camera_values = [finite(row.get('controls', {}).get('bottom_m')) for row in rows]
    camera_present = [value for value in camera_values if value is not None]
    camera_outside = [dict(frame=row['frame'], bottom_m=value) for row, value in zip(rows, camera_values)
                      if value is None or value < .9 - 1e-9 or value > 10. + 1e-9]
    return dict(measurement_range_m=[.1, 9.], camera_bottom_protocol_range_m=[.9, 10.],
                camera_bottom_actual_range_m=[min(camera_present), max(camera_present)] if camera_present else [None, None],
                camera_outside_range=camera_outside, bands=bands, by_category=by_category,
                overall_counts=overall_counts, target_overall_counts=[25 * value for value in DEPTH_BAND_QUOTAS],
                n_selected_images=len(rows), source_counts=dict(source_counts), out_of_range=out_of_range,
                declared_band_mismatches=declared_mismatches,
                quotas_met=all(item['quotas_met'] for item in by_category.values()),
                note='Counts only, not depth-stratified performance. Stratum-selected reuse is not a new random capture cohort.')


def selected_source_counts(states):
    result = Counter()
    for state in states:
        for item in state.get('qualified', []):
            source = item.get('source')
            result['reused_selected' if source == 'reused_input' else 'new_qualified' if source == 'new_input' else 'unknown_source'] += 1
        for item in state.get('skipped', []):
            source = item.get('source')
            if source not in ('reused_input','new_input'):
                # Recover older preflight records using the frozen input naming
                # contract. Never silently assume all missing-source skips new.
                name = Path(str(item.get('frame',''))).name
                source = 'reused_input' if name.startswith('reused_') else 'new_input' if name.startswith('new_') else None
                if source:result['skip_source_inferred_from_filename'] += 1
            result['reused_skipped' if source == 'reused_input' else 'new_skipped' if source == 'new_input' else 'unknown_skipped_source'] += 1
        result['pending_or_inflight'] += len(state.get('inflight', []))
        result['raw_state_attempts'] += int(state.get('attempts',0))
    keys=('reused_selected','new_qualified','unknown_source','reused_skipped','new_skipped','unknown_skipped_source',
          'skip_source_inferred_from_filename','pending_or_inflight','raw_state_attempts')
    return {key:result[key] for key in keys}


def fixed_capture_disposition(root, primary_rows, protocol):
    states = [json.loads((root / f'capture-{index:02d}.json').read_text(encoding='utf-8')) for index in range(25)]
    data = capture_disposition(root, root, primary_rows, diverse=True)
    for cohort in ('single', 'composite'):
        subset = [state for state in states if state['cohort'] == cohort]
        sources = selected_source_counts(subset)
        reused, fresh_qualified, unknown = sources['reused_selected'], sources['new_qualified'], sources['unknown_source']
        new_skipped, reused_skipped = sources['new_skipped'], sources['reused_skipped']
        inflight = sources['pending_or_inflight']
        new_observed_attempts = fresh_qualified + new_skipped + inflight
        item = data['by_cohort'][cohort]
        item.update(sources, reused_selected=reused, new_qualified=fresh_qualified, unknown_source=unknown,
                    new_skipped=new_skipped, new_observed_attempts=new_observed_attempts,
                    raw_state_attempts=sum(int(state.get('attempts', 0)) for state in subset),
                    attempts=reused + reused_skipped + unknown + sources['unknown_skipped_source'] + new_observed_attempts,
                    eligible_candidate_fraction=None,
                    new_candidate_qualification_fraction=fresh_qualified / new_observed_attempts if new_observed_attempts and not sources['unknown_skipped_source'] else None,
                    disposition_balances=True,
                    count_note='attempts is considered reuse including preflight exclusions plus observed fresh candidates; not an unbiased random-candidate denominator')
    for key in ('attempts', 'reused_selected', 'new_qualified', 'unknown_source', 'new_skipped',
                'reused_skipped','unknown_skipped_source','skip_source_inferred_from_filename',
                'new_observed_attempts', 'raw_state_attempts'):
        data['all'][key] = sum(item[key] for item in data['by_cohort'].values())
    data['all']['eligible_candidate_fraction'] = None
    new_total = data['all']['new_observed_attempts']
    data['all']['new_candidate_qualification_fraction'] = data['all']['new_qualified'] / new_total if new_total and not data['all']['unknown_skipped_source'] else None
    data['selected_reuse_is_already_qualified'] = True
    data['selection_provenance'] = protocol.get('selection_provenance', {})
    return data, states


def apply_fixed_range_protocol(data, rows, states, protocol, allow_incomplete=False):
    calibration = protocol.get('operator_calibration', {})
    limit, sigma = finite(calibration.get('error_limit_m')), finite(calibration.get('sigma_m'))
    if protocol.get('evaluation_revision') != FIXED_REVISION or limit != .01 or sigma is None or abs(sigma - .01 / 3) > 1e-12:
        raise ValueError('Fixed-range revision requires protocol error_limit_m=.01 and sigma_m=.01/3')
    primary = [row for row in rows if row['calibration_variant'] == 'operator_noise']
    audit = fixed_range_audit(primary, states)
    if not allow_incomplete:
        if audit['out_of_range'] or audit['camera_outside_range'] or audit['declared_band_mismatches'] or not audit['quotas_met']:
            raise ValueError('Range/band quotas or declared actual-truth bands invalid; no test-error filtering is permitted')
        if len(primary) != 2500 or any(item['n'] != 100 for item in audit['by_category'].values()):
            raise ValueError('The predeclared balanced primary manifest must select exactly100 images per25 categories')
        if set(audit['source_counts']) - {'reused_input', 'new_input'}:
            raise ValueError('Unknown selected-input provenance')
    if any(row.get('measurement_reused') is True for row in rows):
        raise ValueError('This revision reruns both calibration variants, not old numeric results')
    if not allow_incomplete and data.get('qualification',{}).get('all',{}).get('unknown_skipped_source',0):
        raise ValueError('Skipped-input source is unresolved; cannot report fresh-candidate denominator')
    diagnostic_failures = [dict(frame=row['frame'],variant=row['calibration_variant'],error=row['diagnostic_error'])
                           for row in rows if row.get('diagnostic_error')]
    data['diagnostic_failures'] = diagnostic_failures
    if not allow_incomplete and diagnostic_failures:
        raise ValueError('Fresh-parity diagnostics failed; primary numbers retained but this evaluation cannot be published as complete')
    anchor_records = data.get('calibration_anchor_audit', {}).get('records', [])
    limit_mismatches = [record for record in anchor_records
                        if finite(record.get('limit_mm')) is not None and abs(record['limit_mm'] - 10.) > 1e-6]
    anchor_exceeded = sum(abs(record['error_mm']) > 10. + 1e-6 for record in anchor_records)
    phase_exceeded = sum(value.get('over_allowed_limit', 0) for value in
                         data.get('phase_prior_annotation_audit', {}).get('by_variant', {}).values())
    data['fixed_calibration_budget_audit'] = dict(limit_mm=10., anchor_declared_limit_mismatches=limit_mismatches,
                                                anchor_exceeded=anchor_exceeded, phase_exceeded=phase_exceeded)
    if not allow_incomplete and (limit_mismatches or anchor_exceeded or phase_exceeded):
        raise ValueError('Recorded input calibration violates fixed10mm protocol; repair/remeasure priors, do not delete by output error')
    data['evaluation_revision'] = FIXED_REVISION
    data['evaluation_domain'] = 'diverse_parameterized_sensitivity'
    data['measurement_range_m'] = [.1, 9.]
    data['depth_sampling_audit'] = audit
    data['selection_provenance'] = protocol.get('selection_provenance', {})
    data['evaluation_provenance'] = dict(
        reused_input_images=audit['source_counts'].get('reused_input', 0),
        new_input_images=audit['source_counts'].get('new_input', 0), reused_measurement_runs=0,
        new_measurement_runs=len(rows), paired_same_input=True,
        notes=['部分旧输入按预声明范围、深度带配额与先验顺序选入，不依据历史预测误差选取。',
               '全部ideal/operator_noise均按固定±10mm新标定重测，未复用旧测量数字。',
               '旧库未选入的多余合格图保持原样；未选入不等于资格失败。'])
    data['scope_notes'] = [note for note in data['scope_notes'] if
                           'max(10mm,1%' not in note and 'D=10m' not in note]
    data['scope_notes'][:0] = [
        '本次评价版本range_0p1_9m_fixed_10mm：测量液深目标范围0.1–9m，按九个深度带选样；每个小类首带12张，其余八带各11张，共25类×100张。按范围分带均衡不等于真实工业部署频率。',
        '所有米制标定误差均采用固定绝对上限±10mm，零均值截断高斯的截断前尺度σ=10/3mm，不再按被标定量百分比放大。底面anchor、独立几何尺寸及四张非零已知液深先验使用该预算；物理空槽标签仍为0，点击0.3px和重力0.001为另列假设，精确定义以protocol为准。',
        '相机到底面协议范围仍为0.9–10m，实际覆盖由输入审计报告；相机到液面的工作距离为B−H。高液深与远距浅液位不是同一种传感器难度，不把总体误差变化直接归因算法优化。',
        '部分历史输入按范围与深度配额复用，其余补采；所有ideal/operator结果重新计算，不能将选入的旧输入称为新增独立采集，也不能将旧配额外合格图算作本轮资格失败。']
    data['scope_notes'] = [note for note in data['scope_notes'] if '14个单场景子类与11个复合组合均在本批重新采样' not in note]
    if data.get('narrow_domain_control'):
        data['narrow_domain_control']['comparison_type'] = 'historical_different_range_sampling_calibration_budget_unpaired_not_algorithm_gain'
    data['incomplete_diagnostic'] = allow_incomplete
    return data


def num(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def pct(value):
    return '—' if value is None else f'{value * 100:.2f}%'


def table(headers, rows):
    lines = ['.. list-table::', '   :header-rows: 1', '']
    for row in [headers] + list(rows):
        lines.append('   * - ' + str(row[0]).replace('\n', ' '))
        lines.extend('     - ' + str(value).replace('\n', ' ') for value in row[1:])
    return '\n'.join(lines) + '\n\n'


def heading(title, char='-'):
    return title + '\n' + char * max(16, len(title) * 2) + '\n\n'


def stats_tables(named_stats):
    items = list(named_stats)
    text = table(['类型', '合格帧', '功能通过', '液深覆盖', '最高覆盖', '双高度通过'],
                 [(name, item['n'], pct(item['functional_pass']), pct(item['liquid']['coverage']),
                   pct(item['highest']['coverage']), pct(item['strict_two_height'])) for name, item in items])
    text += table(['类型', '液深MAE/P95/最大 mm', '最高MAE/P95/最大 mm', '延时P50/P95/最大 ms', '>500ms'],
                  [(name, '/'.join(num(item['liquid'][key]) for key in ('mae_mm', 'p95_mm', 'max_mm')),
                    '/'.join(num(item['highest'][key]) for key in ('mae_mm', 'p95_mm', 'max_mm')),
                    '/'.join(num(item['latency'][key], 1) for key in ('p50_ms', 'p95_ms', 'max_ms')),
                    item['latency']['over_500']) for name, item in items])
    return text


def report(data):
    diverse = data.get('evaluation_domain') == 'diverse_parameterized_sensitivity'
    fixed_range = data.get('evaluation_revision') == FIXED_REVISION
    title = ('正式ToF V14：0.1–9m与固定±10mm标定性能报告' if fixed_range else
             '正式ToF V14：多样域与标定扰动配对性能报告' if diverse
             else '正式ToF V14：单场景与复合场景性能补充报告')
    text = heading(title, '=')
    text += '日期：2026-10-21。版本：formal_tof_v14_20261021。仅补充冻结代码的测试，不修改测量算法。\n\n'
    text += '.. contents:: 目录\n   :depth: 2\n\n'
    text += heading('一、测试性质与边界')
    for note in data['scope_notes']:
        text += note + '\n\n'
    if fixed_range:
        text += heading('均衡测量域与输入复用', '~')
        sampling = data['depth_sampling_audit']
        provenance = data['evaluation_provenance']
        text += f"本轮复用{provenance['reused_input_images']}张旧输入，新采{provenance['new_input_images']}张输入；全部{provenance['new_measurement_runs']}次测量重新运行，旧数字复用0次。配额外旧合格图未删除，也不算本轮资格失败。\n\n"
        text += table(['实际液深带m', '每类目标数', '总目标数', '实际总数'],
                      [(band['label'], band['target_per_category'], 25 * band['target_per_category'], count)
                       for band, count in zip(sampling['bands'], sampling['overall_counts'])])
        text += table(['类别', '九带实际数量（依上表顺序）', '复用/新采'],
                      [(NAMES[name], '/'.join(map(str, item['counts'])),
                        f"{item['source_counts'].get('reused_input',0)}/{item['source_counts'].get('new_input',0)}")
                       for name, item in sampling['by_category'].items()])
        text += '以上是采样分布审计，不是恢复按液深分层的大量误差表。总体性能以这组预声明均衡样本为分母，不代表任意部署场景的发生频率。\n\n'
    if diverse:
        text += heading('实际参数分布与窄域对照', '~')
        observations = data.get('parameter_observations', {})
        if observations:
            all_parameters, selected_parameters = observations['all_candidates'], observations['eligible']
            text += f"候选参数按唯一图像统计，合格图像涉及{selected_parameters['n_unique_installations']}个有ID安装；同安装多帧并非独立重复安装。下表为实际请求参数，不是实测硬件规格。\n\n"
            text += table(['参数', '全部候选最小/最大', '合格最小/最大', '合格P25/P50/P75'],
                          [(key, num(item['min'], 4) + '/' + num(item['max'], 4),
                            num(selected_parameters['numeric'][key]['min'], 4) + '/' + num(selected_parameters['numeric'][key]['max'], 4),
                            '/'.join(num(selected_parameters['numeric'][key][point], 4) for point in ('p25', 'p50', 'p75')))
                           for key, item in all_parameters['numeric'].items()])
            for key, item in all_parameters['categorical'].items():
                text += f"{key}候选/合格频数：" + json.dumps(item['counts'], ensure_ascii=False) + ' / ' + json.dumps(selected_parameters['categorical'][key]['counts'], ensure_ascii=False) + '。\n\n'
        control = data.get('narrow_domain_control')
        if control and control.get('summary'):
            text += '历史协议与本次样本不是同一输入分布的算法对照，液深范围、采样配额、传感器代理与标定预算均可能不同；不能把差异直接解释为算法提升或退化。\n\n'
            text += stats_tables([('历史协议结果（非配对）', control['summary']),
                                  ('多样域ideal', data['by_calibration_variant']['ideal']['summary']),
                                  ('多样域operator_noise（主体）', data['summary'])])
        if data.get('calibration_parameter_protocol'):
            text += '标定先验与噪声的预声明敏感性协议（按原始字段保存，不作为实测商业相机规格）：\n\n::\n\n'
            text += '\n'.join('   ' + line for line in json.dumps(data['calibration_parameter_protocol'], ensure_ascii=False, indent=2).splitlines()) + '\n\n'
        text += heading('几何、噪声与标定的简要实测审计', '~')
        for field, label, names in (('by_noise_family', '噪声族', data['noise_family_names']),
                                     ('by_geometry_shape', '几何形状', data['geometry_shape_names'])):
            groups = data[field]
            keys = sorted(set(groups['ideal']) | set(groups['operator_noise']))
            text += label + '分组（理想/扰动均为同一多样域输入）：\n\n'
            text += table(['分组', '主体帧数', '液深理想MAE/P95mm', '液深扰动MAE/P95mm',
                           '最高理想/扰动MAEmm', '功能理想/扰动'],
                          [(names.get(key, key), groups['operator_noise'].get(key, {}).get('n', 0),
                            '/'.join(num(groups['ideal'].get(key, {}).get('liquid', {}).get(metric)) for metric in ('mae_mm', 'p95_mm')),
                            '/'.join(num(groups['operator_noise'].get(key, {}).get('liquid', {}).get(metric)) for metric in ('mae_mm', 'p95_mm')),
                            '/'.join(num(groups[variant].get(key, {}).get('highest', {}).get('mae_mm')) for variant in ('ideal', 'operator_noise')),
                            '/'.join(pct(groups[variant].get(key, {}).get('functional_pass')) for variant in ('ideal', 'operator_noise')))
                           for key in keys])
        text += '底面anchor误差按唯一安装及标定版本计一次，不按该安装测量帧数加权；真实返回的测量误差与输入anchor误差分开。\n\n'
        text += table(['标定', '唯一安装/缺失审计', '偏差mm', '绝对MAE/P95/最大mm', '最大|误差|/T', '越预算数'],
                      [(variant, f"{item['n_installations']}/{item['missing_installations']}", num(item['bias_mm']),
                        '/'.join(num(item[key]) for key in ('mae_mm', 'p95_abs_mm', 'max_abs_mm')),
                        num(item['max_abs_to_limit'], 4), item['over_allowed_limit'])
                       for variant, item in data['calibration_anchor_audit']['by_variant'].items()])
        text += '四个非零相位先验的标注误差按安装/版本/物料的首份标注去重；透明与半透明各自的标注组保持分开，固定空槽0不纳入此表。\n\n'
        phase_rows = []
        for material, variants in data['phase_prior_annotation_audit']['by_material'].items():
            for variant, item in variants.items():
                phase_rows.append((material + '/' + variant,
                                   f"{item['n_installations']}/{item['n_annotation_sets']}/{item['n_error_values']}",
                                   num(item['bias_mm']), '/'.join(num(item[key]) for key in ('mae_mm', 'p95_abs_mm', 'max_abs_mm')),
                                   item['annotation_attempts_max']))
        text += table(['物料/标定', '安装/标注组/量数', '偏差mm', '绝对MAE/P95/最大mm', '最大重标次数'], phase_rows)
        capture_audit = data.get('capture_domain_audit')
        entries = capture_audit if isinstance(capture_audit, list) else (
            [dict(item, category=item.get('category', key)) for key, item in capture_audit.items() if isinstance(item, dict)]
            if isinstance(capture_audit, dict) else [])
        if entries:
            text += '以下来自采集输入审计，不以模型预测分档。灰度为可见液面像素BGR2GRAY/255，非lux；饱和为任一通道≥250/255，暗像素为灰度<0.04。各列是逐帧统计量的中位数，完整候选/合格的min/median/P95/max保存在JSON。捕获阶段True/False与延期相位资格应区分。\n\n'
            def audit_value(summary, key, statistic='median'):
                return (summary.get(key) or {}).get(statistic)
            text += table(['类别', '采集合格N', '液面灰度均值/帧内P95的中位数', '饱和/暗像素比例中位数', '可见液面像素中位数'],
                          [(NAMES.get(entry.get('category'), entry.get('category')), entry.get('summary', {}).get('True', {}).get('n', 0),
                            '/'.join(num(audit_value(entry.get('summary', {}).get('True', {}), key), 4) for key in ('luminance_mean', 'luminance_p95')),
                            '/'.join(pct(audit_value(entry.get('summary', {}).get('True', {}), key)) for key in ('saturated_any_channel_fraction', 'underexposed_fraction')),
                            num(audit_value(entry.get('summary', {}).get('True', {}), 'liquid_visible_pixels'), 0)) for entry in entries])
            metadata_failures = sum(len(entry.get('metadata_failures', [])) for entry in entries)
            text += f'请求与实际形状/半径/aspect/视场/相机偏移元数据不一致记录：{metadata_failures}条。该审计验证输入构造，不证明工业相机仿真精度。\n\n'
    phase_ranges = data['protocol'].get('phase_parameter_ranges', data['protocol'].get('phase_parameters'))
    if phase_ranges:
        text += '本批透明/半透明相位生成参数（与固定演示配置的测试域可能不同，不据此直接作同域回退比较）：\n\n::\n\n'
        text += '\n'.join('   ' + line for line in json.dumps(phase_ranges, ensure_ascii=False, indent=2).splitlines()) + '\n\n'
        text += '其中index为液体折射率；surface_ratio为相对于底部参考的液面反射相位振幅比例；extinction以m⁻¹计，scatter为无量纲散射强度。物理参数仅用于生成数据，不直接传给逆解器；逆解器获得五帧原始相关量及其已知液深。相位原始相关量形状与RGB分辨率不同，不能把RGB像素数当作相位有效采样点数。\n\n'
    text += ('每帧液深与最高高度均相对容器底面。液深真值来自生成器96×96表面采样并按其有效液面域选择后求平均；' if diverse else
             '每帧液深与最高高度均相对容器底面。液深真值为生成器在椭圆液面区域96×96采样所得表面高度的平均值；')
    text += '最高高度为表面采样最高点与漂浮物世界坐标网格顶点最高点的较大值。两者不能混用，也不是把请求液位直接当作波面测量真值。\n\n'
    text += '评估驱动的只读核查未发现将当前测试液深或未知真实光学参数显式传给逆解器：当前液深仅用于前向生成，逆解器仅获得五帧已知标定深度及观测，truth.json在推理后读取用于统计。此结论针对已检查的驱动接口，不代替对所有模型内部行为的形式化证明；候选资格依赖真值的限制仍须单列。\n\n'
    text += table(['组别', '实际液深最小/最大 m', '样本性质'],
                  [(name, '/'.join(num(value, 4) for value in data['by_cohort'][key]['actual_liquid_height_range_m']), kind)
                   for key, name, kind in (('single', '单场景', '新独立采样'),
                                           ('composite', '复合场景', '本批新采样，同安装相关' if diverse else '既有独立留出集重放'))])
    text += heading('二、输入资格与分母')
    text += '资格须在分类/测量之前满足。无资格帧不运行推理，不进入合格输入通过率分母，但仍统计全部候选及原因。合格帧无数值输出、异常或超差均保留；无输出按通过失败。误差分布只以该输出有数值的帧为分母。\n\n'
    if diverse:
        text += f"唯一输入图像{data['unique_input_images']}张，标定版本运行{data['measurement_runs']}次。资格、候选/合格计数均按图像去重；以下数值精度总表只统计operator_noise一次，不把两组标定叠加。\n\n"
        if fixed_range:
            text += '复用输入是历史已合格、经本次范围与配额选入的条件子集，因此不把复用加新采的混合池当作随机总体报告统一资格通过比例。下表“候选”仅指已选复用输入加本轮实际新候选；另列新采资格比例，旧库未选项保留在selection_provenance。\n\n'
            text += table(['组别', '选入复用', '新候选', '新合格', '新排除', '新采资格比例'],
                          [(label, item['reused_selected'], item['new_observed_attempts'], item['new_qualified'],
                            item['new_skipped'], pct(item['new_candidate_qualification_fraction']))
                           for label, item in [('单场景', data['qualification']['by_cohort']['single']),
                                              ('复合场景', data['qualification']['by_cohort']['composite']),
                                              ('合并', data['qualification']['all'])]])
            sources=data['qualification']['all']
            text += f"复用输入在本轮preflight另排除{sources.get('reused_skipped',0)}张，不计入新采候选/新采排除分母。缺source但按固定文件前缀恢复{sources.get('skip_source_inferred_from_filename',0)}条，未决来源{sources.get('unknown_skipped_source',0)}条；未决来源时不发布新采资格比例。\n\n"
        for condition in data['qualification'].get('additional_diverse_conditions', []):
            text += condition + '\n\n'
    text += table(['组别', '候选', '捕获暂收', '捕获排除', '最终合格', '资格通过比例'],
                  [(name, item['attempts'], item['capture_qualified'], item['capture_skipped'], item['measured_qualified'],
                    pct(item['eligible_candidate_fraction']))
                   for name, item in [('单场景', data['qualification']['by_cohort']['single']),
                                      ('复合场景', data['qualification']['by_cohort']['composite']),
                                      ('合并', data['qualification']['all'])]])
    for cohort, item in data['qualification']['by_cohort'].items():
        text += ('单场景' if cohort == 'single' else '复合场景') + '捕获排除原因（可重叠）：' + json.dumps(item['skipped_reasons'], ensure_ascii=False) + '。\n\n'
        if item['measured_ineligible']:
            text += f"相位等延期资格复核另排除{item['measured_ineligible']}帧：" + json.dumps(item['deferred_phase_or_input_reasons'], ensure_ascii=False) + '。\n\n'
    text += heading('三、统一评价规则与总表')
    text += '普通液深容差max(20mm,2%真实液深)；存在真实波面的液深容差max(50mm,5%真实液深)；最高高度容差max(50mm,5%真实最高高度)。使用真实现象定义评价容差，绝不根据分类预测改变容差。功能通过要求液深通过，存在波面或漂浮物时还要求最高高度通过；另列所有帧双高度同时通过。分类标签不一致但测量合格仍计通过。\n\n'
    text += heading('八类单场景总表', '~')
    text += '低照、眩光和部分缺失各自汇合程度子组的逐帧记录后重算全部指标，包括分位数；不平均子组MAE或P95。不按实际液深分层。\n\n'
    text += stats_tables([(data['scene_family_meta'][name]['label'], item)
                          for name, item in data['by_scene_family'].items()])
    text += heading('单场景、复合场景与总体汇总', '~')
    text += stats_tables([('单场景', data['by_cohort']['single']), ('复合场景', data['by_cohort']['composite']), ('合并', data['summary'])])
    text += table(['组别', '液深偏差mm', '最高偏差mm', '通过比例说明' if diverse else '功能通过Wilson95%', '合格且<500ms'],
                  [(name, num(item['liquid']['bias_mm']), num(item['highest']['bias_mm']),
                    '描述性比例，不给帧独立CI' if diverse else '–'.join(pct(value) for value in item['wilson95']['functional_pass']),
                    pct(item['functional_and_under_500']))
                   for name, item in [('单场景', data['by_cohort']['single']), ('复合场景', data['by_cohort']['composite']), ('合并', data['summary'])]])
    if diverse:
        text += heading('同一图像的理想/扰动标定配对', '~')
        text += '两个版本使用同一批观测，operator_noise为预设标定先验扰动。配对误差增量=扰动绝对误差−理想绝对误差；仅双侧有数值的配对计算误差增量，缺值另列，未从通过率分母删除。所有比例只作描述，不把同安装复用帧当作独立样本给出置信区间。\n\n'
        if fixed_range:
            text += '本次所有米制标定量均采用固定T=10mm、σ=10/3mm的零均值截断高斯，误差限制±10mm，不随距离放大。评价仍用真实液深的既定容差，ideal为同图对照；输出不裁剪到0.1–9m、不归零、不按误差筛帧。精确参数以protocol.json为准。\n\n'
        else:
            text += '标定误差预算与输出容差不是同一指标，输入误差分布以protocol.json为准；ideal为同图对照，不通过裁剪、归零或筛除失败帧掩盖误差。\n\n'
        text += stats_tables([(variant, value['summary']) for variant, value in data['by_calibration_variant'].items()])
        pairs = [('合并', data['paired_comparison']['summary'])] + [
            (NAMES.get(name, name), item) for name, item in data['paired_comparison']['by_category'].items()]
        text += table(['类型', '合格配对/安装数', '液深平均/P95误差增量mm', '最高平均/P95增量mm', '功能通过变化百分点'],
                      [(name, f"{item['n_qualified_pairs']}/{item['n_installations']}",
                        num(item['liquid']['mean_delta_abs_error_mm']) + '/' + num(item['liquid']['p95_delta_abs_error_mm']),
                        num(item['highest']['mean_delta_abs_error_mm']) + '/' + num(item['highest']['p95_delta_abs_error_mm']),
                        num(item['delta_functional_pass_pp'])) for name, item in pairs])
        text += table(['类型', '双通过', '仅理想通过', '仅扰动通过', '双失败'],
                      [(name,) + tuple(item['functional_transitions'][key] for key in
                                        ('both_pass', 'ideal_only_pass', 'operator_only_pass', 'both_fail'))
                       for name, item in pairs])
        sensitivity = data['paired_comparison']['anchor_sensitivity']
        text += '底面anchor误差与液深估计变化的关联：先对同安装多帧的Δ=扰动估计−理想估计求均值，再按安装等权计算描述性相关和回归。这里的Δ不是绝对误差增量；没有给独立帧显著性或置信区间，其他尺寸/先验扰动同时存在，相关性不能单独证明因果归因。\n\n'
        text += table(['链路域', '有效配对/安装', '平均估计变化mm', '相关系数r', '回归斜率mm/mm', '截距mm'],
                      [(sensitivity['domain_names'][key], f"{item['n_numeric_with_anchor']}/{item['n_installations']}",
                        num(item['installation_balanced']['mean_estimate_change_mm']),
                        num(item['installation_balanced']['correlation'], 4),
                        num(item['installation_balanced']['slope_mm_per_mm'], 4),
                        num(item['installation_balanced']['intercept_mm']))
                       for key, item in sensitivity['by_domain'].items()])
        text += '这项同图配对用于描述标定变化的影响；窄域与多样域之间不是同图输入，二者差异仍混有测试构造、资格和物理代理变化，不能用同一相关系数解释全部误差差异。\n\n'
    text += heading('四、单场景与程度分层')
    single = [(NAMES.get(name, name), item) for name, item in data['by_category'].items() if not name.startswith('compound_')]
    text += stats_tables(single)
    text += ('低照参数范围为低[0.15,0.4)、中[0.4,0.7)、强[0.7,0.98]；' if diverse else
             '低照参数范围为低[0.15,0.4)、中[0.4,0.7)、强[0.7,0.96]；')
    text += '眩光参数为低[0.1,0.35)、中[0.35,0.65)、强[0.65,1]。这是生成器控制参数，不是经照度计标定的lux，也不是通用工业强度标准。缺失三档按实际可见液面缺失比例分组，而非请求丢失参数。分档边界相邻处只归一档。\n\n'
    text += heading('实际施加强度与资格筛选偏差', '~')
    text += '以下同时展示全部候选与最终合格帧实际施加的强度，不只列名义采样区间。资格检查可能使强眩光等组的合格帧集中在名义区间的较低端；长时间采样得到100个合格帧并不证明整个强度区间均适用，也不能把这些合格帧的测量通过率解释成所有候选都成功。强度直方图以实际可取得控制值的候选为分母，缺失控制值另列。\n\n'
    for category, controls in data.get('intensity_observations', {}).items():
        for field, groups in controls.items():
            label = NAMES.get(category, category) + ' / ' + ('低照参数' if field == 'darkness' else '眩光参数')
            text += label + '\n\n'
            text += table(['样本组', '候选数/控制缺失数', '最小/最大', 'P25/P50/P75'],
                          [(name, f"{groups[key]['n']}/{groups[key]['missing']}",
                            num(groups[key]['min'], 4) + '/' + num(groups[key]['max'], 4),
                            '/'.join(num(groups[key][point], 4) for point in ('p25', 'p50', 'p75')))
                           for key, name in (('all_candidates', '全部候选'), ('eligible', '最终合格'), ('rejected', '输入排除'))])
            text += table(['施加强度区间', '全部候选数/比例', '最终合格数/比例'],
                          [(key, f"{groups['all_candidates']['bins'][key]['n']}/{pct(groups['all_candidates']['bins'][key]['fraction'])}",
                            f"{groups['eligible']['bins'][key]['n']}/{pct(groups['eligible']['bins'][key]['fraction'])}")
                           for key in groups['all_candidates']['bins']])
    text += '“单场景”指独立设置一个主动现象控制项，并不保证传感器回波没有天然缺失；输入仍须通过同一资格检查。完整数据另按实际缺失比例聚合，不能把下述跨场景分档当成新的独立样本。\n\n'
    for kind, label in (('loss', '全部普通几何帧：实际液面缺失分档'),
                        ('phase_loss', '透明/半透明帧：相位有效样本缺失分档')):
        if data['actual_condition_bins'].get(kind):
            text += heading(label, '~')
            text += stats_tables(data['actual_condition_bins'][kind].items())
    text += heading('五、逐复合组合统计')
    text += stats_tables([(NAMES.get(name, name), item) for name, item in data['by_category'].items() if name.startswith('compound_')])
    text += heading('六、误差分布')
    text += '绝对误差使用互斥区间；相对误差=绝对高度误差/真实相对底面高度。真值为零不使用epsilon替代，单独记为相对误差无定义。下表不把未输出帧藏进误差小的区间。\n\n'
    for cohort, label in (('single', '单场景'), ('composite', '复合场景'), (None, '合并')):
        item = data['summary'] if cohort is None else data['by_cohort'][cohort]
        text += heading(label, '~')
        text += table(['绝对误差区间', '液深帧数/比例', '最高帧数/比例'],
                      [(key, f"{item['liquid']['absolute_bins'][key]['n']}/{pct(item['liquid']['absolute_bins'][key]['fraction'])}",
                        f"{item['highest']['absolute_bins'][key]['n']}/{pct(item['highest']['absolute_bins'][key]['fraction'])}")
                       for key, _ in ABS_BINS])
        text += table(['相对误差区间', '液深帧数/比例', '最高帧数/比例'],
                      [(key, f"{item['liquid']['relative_bins'][key]['n']}/{pct(item['liquid']['relative_bins'][key]['fraction'])}",
                        f"{item['highest']['relative_bins'][key]['n']}/{pct(item['highest']['relative_bins'][key]['fraction'])}")
                       for key, _ in REL_BINS])
        text += f"液深无输出{item['liquid']['missing']}帧，相对误差无定义{item['liquid']['relative_undefined']}帧；最高无输出{item['highest']['missing']}帧，相对误差无定义{item['highest']['relative_undefined']}帧。\n\n"
    text += ('每个场景/程度的完整误差区间计数、偏差与描述性比例写入performance.json的by_category，不按实际液深另行分层。\n\n' if diverse else
             '每个场景/程度的完整误差区间计数、偏差和置信区间均同时写入performance.json的by_category，不按实际液深另行分层。\n\n')
    text += heading('七、尾部误差与选路诊断')
    text += f"本次合并功能未通过{data['summary']['failed_functional_frames']}帧，异常{data['summary']['exception_frames']}帧；标签组合不同但功能合格{data['summary']['wrong_tags_but_functional']}帧。以下仅列本次测量的最大误差，不引用旧版本已修复错误作为当前错误。\n\n"
    for output, label in (('liquid', '液深'), ('highest', '最高高度')):
        text += label + '最大误差前5帧：\n\n'
        text += table(['帧', '类型', '真实/预测m', '绝对误差mm', '容差mm', '通过'],
                      [(item['frame'], NAMES.get(item['category'], item['category']),
                        num(item['true_m'], 4) + '/' + num(item['predicted_m'], 4),
                        num(item['absolute_error_mm']), num(item['tolerance_mm']), '是' if item['passed'] else '否')
                       for item in data['largest_errors'][output][:5]])
    text += table(['标签', '精确率', '召回率', '误报/漏报'],
                  [(name, pct(item['precision']), pct(item['recall']), f"{item['fp']}/{item['fn']}")
                   for name, item in data['classification_diagnostics'].items()])
    text += '标签诊断不作为硬验收门槛；未进行强制真值标签的反事实重跑，不能据此声称每次模块开关均最优。\n\n'
    text += heading('八、延时、局限与复现')
    text += '延时包含前置资格检查、分类、完整测量与CUDA同步，排除初始化、渲染、磁盘读取、网络及浏览器。初始化单独记录；当前是在服务器GPU环境下的计算耗时，不等同终端端到端延时。最大耗时与超500ms帧均保留，P95达标不代表逐帧硬实时。\n\n'
    text += '计时采用逐帧串行与预热后的执行；空槽几何构建/读取先于测量计时，单列初始化也不是首次安装与全部标定流程的总耗时。不同批次机器状态或顺序可能影响尾延时，不能在算法未改变时仅凭重放耗时波动宣称速度改进。\n\n'
    text += ('每类100个图像不是100个独立安装，参数化仿真和先验扰动不是真实工业总体的抽样；真实传感器噪声及实际标定误差仍须现场验证。' if diverse else
             '每类100帧或以上是当前测试规模而非权威认证；同域仿真、圆形容器与理想准确先验限制外推，必须经真实传感器和现场独立标定验证才能评价工业精度。')
    text += '透明/半透明实验不证明复杂多层、强散射、漂浮物遮挡或非平整液面均受支持。\n\n'
    text += '冻结版本与指纹：\n\n::\n\n'
    text += '\n'.join('   ' + line for line in json.dumps(data['source_digest'], ensure_ascii=False, indent=2).splitlines()) + '\n\n'
    text += '逐帧依据为measurements.jsonl，候选资格依据为capture-*.json，完整汇总为performance.json。'
    text += ('本批使用按预声明配额选入的历史输入与新增输入，所有标定测量均重跑；按唯一图像配对计数，历史结果仅另列协议对照。\n' if fixed_range else
             '本批单场景与复合场景均重新采样，两种标定运行按唯一图像配对而不重复计数；历史报告另列对照。\n' if diverse else
             '新增单场景与既有复合集重放的来源分别列示，不重复计算为新独立样本。\n')
    return text


def self_test():
    def row(frame, h, pred, high, eligible_value=True, bits=0):
        return dict(frame=frame, cohort='single', category='ordinary', sensor='tof', bits=bits,
                    truth=dict(liquid_height_m=h, highest_height_m=h), liquid=pred, highest=high,
                    qualification=dict(eligible=eligible_value, liquid_invalid_fraction=.1),
                    inference_executed=eligible_value, wall_ms=100., init_ms=900.,
                    target_tags=[], classification=dict(tags=[]), controls={})
    rows = [row('a', 1., 1.01, 1.03), row('b', 1., None, None), row('c', 0., .001, .001),
            row('skip', 1., None, None, False)]
    stat = summarize(rows)
    assert stat['n'] == 3 and stat['liquid']['numeric'] == 2
    assert stat['pass_counts']['functional_pass'] == 2
    assert stat['liquid']['relative_undefined'] == 1 and stat['liquid']['relative_valid'] == 1
    assert abs(stat['liquid']['mae_mm'] - 5.5) < 1e-8
    assert sum(x['n'] for x in stat['liquid']['absolute_bins'].values()) == 2
    assert 0 < wilson(100, 100)[0] < 1 and wilson(0, 0) == [None, None]
    assert quantile([0, 10], .95) == 9.5
    intensity = intensity_stats([{'glare': .65}, {'glare': .7}, {'glare': .8}, {}], 'glare')
    assert intensity['n'] == 4 and intensity['missing'] == 1 and intensity['observed'] == 3
    assert intensity['min'] == .65 and intensity['max'] == .8
    assert sum(item['n'] for item in intensity['bins'].values()) == 3
    malformed = dict(rows[0], qualification={})
    try:
        summarize([malformed])
        raise AssertionError('missing eligibility silently accepted')
    except ValueError:
        pass
    waves = row('wave', 1., 1.03, 1.06, bits=4)
    assert passed(waves, 'liquid') and not functional(waves)
    assert not passed(dict(waves, bits=0), 'liquid')
    result = build(rows, {}, {})
    assert sum(item['n'] for item in result['by_scene_family'].values()) == result['by_cohort']['single']['n']
    assert result['by_scene_family']['ordinary']['liquid']['mae_mm'] == result['by_cohort']['single']['liquid']['mae_mm']
    json.dumps(result, allow_nan=False)
    fixture_disposition = dict(attempts=4, capture_qualified=4, capture_skipped=0,
                               measured_qualified=3, eligible_candidate_fraction=.75,
                               measured_ineligible=1, skipped_reasons={},
                               deferred_phase_or_input_reasons={'fixture': 1})
    result['qualification'] = dict(by_cohort=dict(single=fixture_disposition, composite=fixture_disposition),
                                   all=fixture_disposition)
    rendered = report(result)
    assert '目录' in rendered and '0.00%' not in rendered.split('日期')[0]
    paired_rows = []
    for record in rows:
        ideal = dict(record, calibration_variant='ideal', installation_id=7)
        operator = dict(record, calibration_variant='operator_noise', installation_id=7)
        if record['frame'] == 'a':
            operator['liquid'] = 1.025
        paired_rows.extend((ideal, operator))
    diverse = build_diverse(paired_rows, {}, result['qualification'], result)
    assert diverse['unique_input_images'] == 4 and diverse['measurement_runs'] == 8
    assert diverse['summary']['n'] == 3 and diverse['summary']['n_installations'] == 1
    assert diverse['summary']['wilson95']['functional_pass'] == [None, None]
    assert diverse['paired_comparison']['summary']['functional_transitions']['ideal_only_pass'] == 1
    assert abs(diverse['paired_comparison']['summary']['liquid']['mean_delta_abs_error_mm'] - 7.5) < 1e-8
    assert diverse['paired_comparison']['summary']['liquid']['missing_transitions']['both_missing'] == 1
    assert diverse['qualification']['all']['measured_qualified'] == 3
    assert diverse['by_noise_family']['operator_noise']['unknown']['n'] == 3
    json.dumps(diverse, allow_nan=False)
    diverse_text = report(diverse)
    assert '标定扰动' in diverse_text and '既有独立留出集重放' not in diverse_text
    wrong_pair = [dict(record) for record in paired_rows]
    wrong_pair[1] = dict(wrong_pair[1], controls={'changed': True})
    try:
        build_diverse(wrong_pair, {}, {})
        raise AssertionError('nonidentical paired input accepted')
    except ValueError:
        pass
    calibration_rows = []
    for variant in ('ideal', 'operator_noise'):
        record = dict(rows[0], installation_id=7, calibration_variant=variant,
                      controls=dict(material='homogeneous_transparent', capacity_m=1.),
                      calibration_audit=dict(anchor_error_mm=0. if variant == 'ideal' else 4.,
                                             anchor_error_limit_mm=10., metric_anchor_m=1., attempt=1),
                      phase_prior_annotation=dict(errors_m=[0.] * 4 if variant == 'ideal' else [.001, -.002, .003, -.004],
                                                  annotation_attempt=1))
        calibration_rows.extend([record, dict(record)])
    anchors, annotation = calibration_audits(calibration_rows, {})
    assert anchors['by_variant']['operator_noise']['n_records'] == 1
    assert anchors['by_variant']['operator_noise']['mae_mm'] == 4.
    assert anchors['by_variant']['operator_noise']['max_abs_to_limit'] == .4
    assert annotation['by_variant']['operator_noise']['n_annotation_sets'] == 1
    assert annotation['by_variant']['operator_noise']['n_error_values'] == 4
    assert annotation['by_variant']['operator_noise']['mae_mm'] == 2.5
    assert annotation['prior_index']['4']['operator_noise']['max_abs_mm'] == 4.
    regression = descriptive_regression([(-1., 2.), (0., 3.), (1., 4.)])
    assert abs(regression['correlation'] - 1.) < 1e-12
    assert regression['slope_mm_per_mm'] == 1. and regression['intercept_mm'] == 3.
    assert descriptive_regression([(1., 2.), (1., 3.)])['slope_mm_per_mm'] is None
    diverse['calibration_anchor_audit'], diverse['phase_prior_annotation_audit'] = anchors, annotation
    metric = dict(min=.1, median=.2, p95=.3, max=.4)
    diverse['capture_domain_audit'] = [dict(category='glare_strong', metadata_failures=[],
                                         summary={'True': dict(n=1, luminance_mean=metric, luminance_p95=metric,
                                                               saturated_any_channel_fraction=metric,
                                                               underexposed_fraction=metric, liquid_visible_pixels=metric)})]
    assert '底面anchor' in report(diverse) and '灰度' in report(diverse)
    assert depth_band_index(.1) == 0 and depth_band_index(1.) == 1 and depth_band_index(9.) == 8
    assert depth_band_index(.099) is None and depth_band_index(9.001) is None
    selected_rows, states = [], []
    for category, name in enumerate(NAMES):
        cohort = 'composite' if name.startswith('compound_') else 'single'
        state = dict(category=name, cohort=cohort, qualified=[])
        local_index = 0
        for band, ((lower, upper), quota) in enumerate(zip(DEPTH_BANDS, DEPTH_BAND_QUOTAS)):
            for repeat in range(quota):
                height = .1 if band == 0 and repeat == 0 else 9. if band == 8 and repeat == quota - 1 else (lower + upper) / 2
                frame = f'quota-{category}-{local_index}'
                source = 'reused_input' if local_index % 2 else 'new_input'
                state['qualified'].append(dict(frame=frame, source=source, source_frame=None, depth_band=band))
                selected_rows.append(dict(frame=frame, category=name, cohort=cohort,
                                           calibration_variant='operator_noise', controls=dict(bottom_m=10.),
                                           truth=dict(liquid_height_m=height)))
                local_index += 1
        states.append(state)
    sampling = fixed_range_audit(selected_rows, states)
    assert sampling['quotas_met'] and sampling['overall_counts'] == [300] + [275] * 8
    assert sampling['source_counts'] == dict(new_input=1250, reused_input=1250)
    fixed_protocol = dict(evaluation_revision=FIXED_REVISION,
                          operator_calibration=dict(error_limit_m=.01, sigma_m=.01 / 3))
    fixed_data = apply_fixed_range_protocol(dict(scope_notes=['T=max(10mm,1%|D|)', '保留审计']),
                                            selected_rows + [dict(row, calibration_variant='ideal') for row in selected_rows],
                                            states, fixed_protocol)
    assert fixed_data['evaluation_provenance']['new_measurement_runs'] == 5000
    assert all('max(10mm,1%' not in note and 'D=10m' not in note for note in fixed_data['scope_notes'])
    high_prior_rows = [dict(record, controls=dict(material='homogeneous_transparent', capacity_m=10.))
                       for record in calibration_rows]
    _, fixed_prior = calibration_audits(high_prior_rows, fixed_protocol)
    assert fixed_prior['by_variant']['operator_noise']['max_abs_to_limit'] == .4
    sources=selected_source_counts([dict(attempts=10,qualified=[dict(source='reused_input')]*2+[dict(source='new_input')]*5,
                                       skipped=[dict(source='reused_input'),dict(source='new_input'),dict(frame='/data/new_legacy')])])
    assert sources['reused_selected']==2 and sources['new_qualified']==5
    assert sources['reused_skipped']==1 and sources['new_skipped']==2 and sources['skip_source_inferred_from_filename']==1
    assert sources['new_qualified']/(sources['new_qualified']+sources['new_skipped'])==5/7
    fixed_report = json.loads(json.dumps(diverse))
    fixed_report.update(evaluation_revision=FIXED_REVISION, depth_sampling_audit=sampling,
                        evaluation_provenance=fixed_data['evaluation_provenance'], scope_notes=fixed_data['scope_notes'])
    for item in [fixed_report['qualification']['all']] + list(fixed_report['qualification']['by_cohort'].values()):
        item.update(reused_selected=1,new_observed_attempts=2,new_qualified=2,new_skipped=0,new_candidate_qualification_fraction=1.)
    fixed_text = report(fixed_report)
    assert '固定±10mm' in fixed_text and 'D=10m时T=100mm' not in fixed_text
    print('SELF_TEST_OK: missing output, zero truth, eligibility, tolerance, histogram, pooled percentiles, intensity, paired variants, unique qualification and descriptive clustered proportions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get('LIQUID_EVAL_ROOT', str(Path(__file__).resolve().parent))))
    parser.add_argument('--old-root', type=Path)
    parser.add_argument('--diverse', action='store_true', help='25 new capture states, two calibration variants per input image')
    parser.add_argument('--control-root', type=Path, help='Historical evaluation root containing performance.json, unpaired protocol comparison only')
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.root.resolve()
    old_root = args.old_root or root.parent / 'formal-v14-compounds-20261021'
    source = root / 'measurements.jsonl'
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    fixed_range = protocol.get('evaluation_revision') == FIXED_REVISION
    if fixed_range:
        args.diverse = True
    primary_rows = [row for row in rows if row.get('calibration_variant') == 'operator_noise'] if args.diverse else rows
    if not args.allow_incomplete:
        if args.diverse:
            paired_keys = defaultdict(set)
            for row in rows:
                paired_keys[image_key(row)].add(row.get('calibration_variant'))
            incomplete = [key for key, variants in paired_keys.items() if variants != {'ideal', 'operator_noise'}]
            if incomplete:
                raise ValueError(f'Incomplete calibration pairs: {len(incomplete)} input images')
        else:
            for cohort, key in (('single', 'generated_single_target'), ('composite', 'replay_composite_target')):
                expected = protocol.get(key)
                actual = sum(row.get('cohort') == cohort for row in rows)
                if expected is not None and actual < expected:
                    raise ValueError(f'Incomplete {cohort}: {actual} rows, minimum {expected}; do not publish prematurely')
        for category in NAMES:
            qualified = sum(row.get('category') == category and eligible(row) for row in primary_rows)
            if qualified < 100:
                raise ValueError(f'Incomplete category {category}: only {qualified} qualified rows, minimum 100')
    if fixed_range:
        disposition, selection_states = fixed_capture_disposition(root, primary_rows, protocol)
    else:
        disposition = capture_disposition(root, old_root, primary_rows, diverse=args.diverse)
    if args.diverse:
        control = json.loads((args.control_root / 'performance.json').read_text(encoding='utf-8')) if args.control_root else None
        data = build_diverse(rows, protocol, disposition, control)
        if fixed_range:
            data = apply_fixed_range_protocol(data, rows, selection_states, protocol, args.allow_incomplete)
    else:
        data = build(rows, protocol, disposition)
    data['input_files_sha256'] = {source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
                                  'protocol.json': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest()}
    if args.diverse and args.control_root:
        data['input_files_sha256']['narrow_control_performance.json'] = hashlib.sha256((args.control_root / 'performance.json').read_bytes()).hexdigest()
    report_path = root / ('正式ToF_V14_0p1-9m_固定10mm标定性能报告_2026-10-21.rst' if fixed_range else
                          '正式ToF_V14_多样域与标定扰动配对性能报告_2026-10-21.rst' if args.diverse
                          else '正式ToF_V14_单场景与复合场景性能报告_2026-10-21.rst')
    data['report_file'] = report_path.name
    if (root / 'selection-manifest.json').exists():
        data['input_files_sha256']['selection-manifest.json'] = hashlib.sha256((root / 'selection-manifest.json').read_bytes()).hexdigest()
    (root / 'performance.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    report_path.write_text(report(data), encoding='utf-8')
    print(json.dumps(dict(summary=data['summary'], report=str(report_path)), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
