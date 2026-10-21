"""Independent post-hoc integrity audit; never edits measurements or gates.

Reads measurements.jsonl/protocol.json and writes final-results-audit.json.
Calibration errors are deduplicated by physical annotation, not wet frames.
All model failures remain in counts and error/pass denominators.
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

VARIANTS = ('ideal', 'operator_noise')
SINGLES = ('ordinary', 'low_light_low', 'low_light_medium', 'low_light_strong',
           'glare_low', 'glare_medium', 'glare_strong', 'uneven_surface',
           'floating_objects', 'loss_low', 'loss_medium', 'loss_high',
           'transparent', 'translucent')
CATEGORIES = SINGLES + tuple(f'compound_{bits:02d}' for bits in (3,5,6,7,9,10,11,12,13,14,15))
BANDS = ((.1,1.),) + tuple((float(i),float(i+1)) for i in range(1,9))
QUOTAS = (12,) + (11,) * 8
GEOMETRY_ANNOTATIONS = ('bottom_distance','wall_height','check_span','rim_width','rim_length')
PHASE_ANNOTATIONS = tuple(f'known_liquid_height_{i}' for i in range(1,5))
METRIC_LIMIT_MM = 10.
METRIC_SIGMA_MM = 10. / 3


def finite(value):
    if isinstance(value,bool):
        return None
    try:
        value=float(value)
    except (TypeError,ValueError):
        return None
    return value if math.isfinite(value) else None


def percentile(values, fraction):
    if not values:
        return None
    ordered=sorted(values);position=(len(ordered)-1)*fraction
    lo,hi=math.floor(position),math.ceil(position)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def error_statistics(errors):
    n=len(errors)
    mean=sum(errors)/n if n else None
    absolute=[abs(value) for value in errors]
    return dict(n=n,mean_error_mm=mean,
                std_error_mm=math.sqrt(sum((value-mean)**2 for value in errors)/n) if n else None,
                std_definition='population standard deviation; parent scope specifies unique annotation quantities or measurement rows',
                min_error_mm=min(errors) if n else None,max_error_mm=max(errors) if n else None,
                mae_mm=sum(absolute)/n if n else None,
                p95_abs_error_mm=percentile(absolute,.95),
                max_abs_error_mm=max(absolute) if n else None,
                over_10mm=sum(value>METRIC_LIMIT_MM+1e-6 for value in absolute))


def key_for(row):
    return row.get('cohort'),row.get('category'),str(row.get('frame'))


def installation(row):
    value=row.get('installation_id',row.get('group',row.get('controls',{}).get('installation_id')))
    return None if value is None else str(value)


def actual_height(row):
    return finite(row.get('truth',{}).get('liquid_height_m'))


def band_of(value):
    if value is None or not .1<=value<=9.:
        return None
    return min(8,int(value))


def is_phase(row):
    return row.get('category') in ('transparent','translucent')


def output(row, name):
    return finite(row.get(name))


def liquid_error_summary(rows):
    errors=[];missing=0;passed=0;invalid_truth=0
    for row in rows:
        truth=actual_height(row);prediction=output(row,'liquid')
        if truth is None:
            invalid_truth+=1
            continue
        if prediction is None:
            missing+=1
            continue
        difference=(prediction-truth)*1000
        errors.append(difference)
        passed+=abs(difference)<=max(20.,20.*abs(truth))
    stats=error_statistics(errors)
    stats.update(rows=len(rows),numeric=len(errors),missing=missing,invalid_truth=invalid_truth,
                 coverage=len(errors)/len(rows) if rows else None,
                 tolerance_pass_count=passed,tolerance_pass_fraction=passed/len(rows) if rows else None,
                 tolerance='max(20mm,2% true liquid height); phase is planar')
    return stats


def annotation_audit(rows):
    unique={};conflicts=[];bad_records=[];raw_error_values=[]
    expected_geometry=set();expected_phase=set()
    for row in rows:
        variant=row.get('calibration_variant');gid=installation(row)
        if variant not in VARIANTS or gid is None:
            continue
        if row.get('qualification',{}).get('eligible') is True:
            expected_geometry.update((gid,variant,'geometry','',name) for name in GEOMETRY_ANNOTATIONS)
            if is_phase(row):
                material=row.get('controls',{}).get('material',row['category'])
                expected_phase.update((gid,variant,'phase',material,name) for name in PHASE_ANNOTATIONS)
        for scope,audit_key in (('geometry','calibration_audit'),('phase','phase_prior_annotation')):
            audit=row.get(audit_key) or {}
            material=row.get('controls',{}).get('material',row.get('category')) if scope=='phase' else ''
            for name,record in (audit.get('metric_annotations') or {}).items():
                ref=finite(record.get('reference_m'));annotated=finite(record.get('annotated_m'))
                if ref is None or annotated is None:
                    bad_records.append(dict(frame=row.get('frame'),scope=scope,annotation=name,reason='missing finite reference/annotation'))
                    continue
                error=(annotated-ref)*1000
                raw_error_values.append(error)
                limit=finite(record.get('error_limit_mm'));sigma=finite(record.get('sigma_mm'))
                declared_error=finite(record.get('error_mm'))
                declared_error_m=finite(record.get('error_m'))
                problems=[]
                if limit is None or abs(limit-10.)>1e-6:problems.append('declared_limit_not_10mm')
                if sigma is None or abs(sigma-METRIC_SIGMA_MM)>1e-6:problems.append('declared_sigma_not_10_div_3mm')
                if declared_error is None or abs(declared_error-error)>1e-6:problems.append('error_mm_disagrees_with_annotation')
                if declared_error_m is None or abs(declared_error_m*1000-error)>1e-6:problems.append('error_m_disagrees_with_annotation')
                if variant=='ideal' and abs(error)>1e-6:problems.append('nonzero_ideal_annotation_error')
                if scope=='phase' and name=='known_liquid_height_0':problems.append('empty_zero_must_not_be_perturbed')
                item=dict(installation_id=gid,variant=variant,scope=scope,material=material,annotation=name,
                          reference_m=ref,annotated_m=annotated,error_mm=error,limit_mm=limit,sigma_mm=sigma)
                key=gid,variant,scope,material,name
                if key in unique and unique[key]!=item:
                    conflicts.append(dict(key=list(key),first=unique[key],conflicting=item))
                else:
                    unique[key]=item
                if problems:
                    bad_records.append(dict(frame=row.get('frame'),key=list(key),reasons=problems))
    grouped={}
    for scope in ('geometry','phase'):
        by_variant={}
        for variant in VARIANTS:
            selected=[record for key,record in unique.items() if key[1]==variant and key[2]==scope]
            stats=error_statistics([record['error_mm'] for record in selected])
            stats.update(n_installations=len({record['installation_id'] for record in selected}),
                         count_unit='unique installation/variant/annotation' if scope=='geometry'
                                    else 'unique installation/variant/material/annotation')
            by_variant[variant]=stats
        names=sorted({key[-1] for key in unique if key[2]==scope})
        by_name={name:{variant:error_statistics([record['error_mm'] for key,record in unique.items()
                                                if key[1]==variant and key[2]==scope and key[-1]==name])
                       for variant in VARIANTS} for name in names}
        grouped[scope]=dict(by_variant=by_variant,by_annotation=by_name)
    phase_materials=sorted({key[3] for key in unique if key[2]=='phase'})
    grouped['phase']['by_material']={
        material:{variant:error_statistics([record['error_mm'] for key,record in unique.items()
                                           if key[1]==variant and key[2]=='phase' and key[3]==material])
                  for variant in VARIANTS} for material in phase_materials}
    observed=set(unique)
    grouped.update(unique_annotation_count=len(unique),conflicting_duplicate_count=len(conflicts),
                   conflicting_duplicate_examples=conflicts[:20],metadata_problem_occurrences=len(bad_records),
                   metadata_problem_examples=bad_records[:20],
                   expected_geometry_annotations=len(expected_geometry),expected_phase_annotations=len(expected_phase),
                   missing_geometry_annotations=[list(key) for key in sorted(expected_geometry-observed)],
                   missing_phase_annotations=[list(key) for key in sorted(expected_phase-observed)],
                   all_unique_error_statistics=error_statistics([record['error_mm'] for record in unique.values()]),
                   max_abs_error_including_conflicting_duplicates_mm=max(map(abs,raw_error_values),default=None))
    return grouped,unique


def extrapolation_audit(rows,unique_annotations,protocol):
    fractions=protocol.get('phase_parameters',{}).get('priors_fraction',[0,.1,.35,.65,.95])
    last_fraction=max(map(float,fractions))
    materials={}
    flag_mismatches=[]
    for category in ('transparent','translucent'):
        by_variant={}
        for variant in VARIANTS:
            subset=[row for row in rows if row.get('category')==category and row.get('calibration_variant')==variant
                    and row.get('qualification',{}).get('eligible') is True]
            groups=defaultdict(list)
            for row in subset:
                truth=actual_height(row);cap=finite(row.get('controls',{}).get('capacity_m'))
                if truth is None or cap is None:
                    groups['unknown_true_prior'].append(row)
                    continue
                above=truth>last_fraction*cap
                groups['above_true_prior' if above else 'within_true_prior'].append(row)
                declared=row.get('phase_test_above_highest_prior')
                if type(declared) is bool and declared!=above:
                    flag_mismatches.append(dict(frame=row.get('frame'),variant=variant,declared=declared,recomputed=above))
                material=row.get('controls',{}).get('material',category)
                key=installation(row),variant,'phase',material,'known_liquid_height_4'
                annotation=unique_annotations.get(key)
                if annotation is None:
                    groups['unknown_annotated_prior'].append(row)
                else:
                    groups['above_annotated_prior' if truth>annotation['annotated_m'] else 'within_annotated_prior'].append(row)
            by_variant[variant]=dict(total_rows=len(subset),
                **{key:liquid_error_summary(groups[key]) for key in
                   ('above_true_prior','within_true_prior','unknown_true_prior',
                    'above_annotated_prior','within_annotated_prior','unknown_annotated_prior')})
        materials[category]=dict(by_variant=by_variant)
    return dict(reference='true prior capture depth=max(prior fractions)*capacity; annotation threshold separately audited',
                count_unit='measurement rows within each variant; not two independent images',
                by_material=materials,declared_flag_mismatches=flag_mismatches)


def audit_rows(rows,protocol):
    pairs=defaultdict(dict);duplicate_rows=[];unexpected_variants=[];qualification_mismatches=[]
    inconsistent_pairs=[];wrong_cohorts=[];camera_outside=[]
    for row in rows:
        variant=row.get('calibration_variant')
        if variant not in VARIANTS:
            unexpected_variants.append(dict(frame=row.get('frame'),variant=variant));continue
        key=key_for(row)
        expected_cohort='single' if row.get('category') in SINGLES else 'composite'
        if row.get('cohort')!=expected_cohort:wrong_cohorts.append(dict(frame=row.get('frame'),cohort=row.get('cohort')))
        bottom=finite(row.get('controls',{}).get('bottom_m'))
        if bottom is None or not .9<=bottom<=10.:camera_outside.append(dict(frame=row.get('frame'),bottom_m=bottom))
        if variant in pairs[key]:
            duplicate_rows.append(dict(key=list(key),variant=variant))
        pairs[key][variant]=row
    by_category={}
    outside=[];missing_truth=[];declared_band_mismatches=[];ineligible_inference=[]
    for name in CATEGORIES:
        selected=[(key,pair) for key,pair in pairs.items() if key[1]==name]
        counts=[0]*9;incomplete=0;qualified=0
        for key,pair in selected:
            if set(pair)!=set(VARIANTS):incomplete+=1
            representative=pair.get('operator_noise',next(iter(pair.values())))
            actual=actual_height(representative);band=band_of(actual)
            if actual is None:missing_truth.append(list(key))
            elif band is None:outside.append(dict(key=list(key),truth_height_m=actual))
            else:counts[band]+=1
            if representative.get('depth_band') is not None and representative['depth_band']!=band:
                declared_band_mismatches.append(list(key))
            for record in pair.values():
                if record.get('qualification',{}).get('eligible') is False and record.get('inference_executed') is True:
                    ineligible_inference.append(list(key))
            eligibility=[record.get('qualification',{}).get('eligible') for record in pair.values()]
            if all(value is True for value in eligibility):qualified+=1
            if len(pair)==2:
                a,b=pair['ideal'],pair['operator_noise']
                if a.get('qualification',{}).get('eligible')!=b.get('qualification',{}).get('eligible'):
                    qualification_mismatches.append(list(key))
                if a.get('truth')!=b.get('truth') or a.get('controls')!=b.get('controls') or installation(a)!=installation(b):
                    inconsistent_pairs.append(list(key))
        by_category[name]=dict(unique_images=len(selected),complete_pairs=len(selected)-incomplete,
                               qualified_images=qualified,depth_band_counts=counts,target_counts=list(QUOTAS),
                               exact100=len(selected)==100,quotas_exact=counts==list(QUOTAS))
    variant_counts={}
    for variant in VARIANTS:
        selected=[row for row in rows if row.get('calibration_variant')==variant]
        eligible=[row for row in selected if row.get('qualification',{}).get('eligible') is True]
        geometry=[row for row in eligible if not is_phase(row)]
        variant_counts[variant]=dict(rows=len(selected),qualified_rows=len(eligible),
            numeric_liquid=sum(output(row,'liquid') is not None for row in eligible),
            qualified_missing_liquid=sum(output(row,'liquid') is None for row in eligible),
            qualified_missing_highest_raw=sum(output(row,'highest') is None for row in eligible),
            highest_not_applicable_phase=sum(is_phase(row) for row in eligible),
            geometry_highest_rows=len(geometry),
            qualified_missing_highest_geometry=sum(output(row,'highest') is None for row in geometry),
            primary_exception_rows=sum(bool(row.get('error')) for row in eligible),
            primary_exception_with_numeric_liquid=sum(bool(row.get('error')) and output(row,'liquid') is not None for row in eligible),
            diagnostic_error_rows=sum(bool(row.get('diagnostic_error')) for row in selected),
            pre_measurement_failure_rows=sum(bool(row.get('error')) and row.get('inference_executed') is not True for row in eligible),
            missing_wall_time=sum(finite(row.get('wall_ms')) is None for row in eligible))
    annotations,unique=annotation_audit(rows)
    integrity=dict(expected_unique_images=2500,unique_images=len(pairs),measurement_rows=len(rows),
                   complete_pairs=sum(set(pair)==set(VARIANTS) for pair in pairs.values()),
                   duplicate_rows=duplicate_rows,unexpected_variants=unexpected_variants,
                   unknown_categories=sorted({key[1] for key in pairs if key[1] not in CATEGORIES},key=str),
                   qualification_mismatches=qualification_mismatches,inconsistent_pair_inputs=inconsistent_pairs,
                   out_of_range=outside,missing_truth=missing_truth,declared_band_mismatches=declared_band_mismatches,
                   ineligible_inputs_inferred=ineligible_inference,
                   wrong_cohorts=wrong_cohorts,camera_bottom_outside_protocol=camera_outside,
                   per_category=by_category,overall_band_counts=[sum(value['depth_band_counts'][i] for value in by_category.values()) for i in range(9)])
    structural=(len(pairs)==2500 and len(rows)==5000 and all(set(pair)==set(VARIANTS) for pair in pairs.values())
                and all(value['exact100'] and value['quotas_exact'] and value['qualified_images']==100 for value in by_category.values())
                and not any((duplicate_rows,unexpected_variants,qualification_mismatches,inconsistent_pairs,
                             outside,missing_truth,declared_band_mismatches,ineligible_inference,integrity['unknown_categories'],wrong_cohorts,camera_outside)))
    annotation_valid=(not annotations['conflicting_duplicate_count'] and not annotations['metadata_problem_occurrences']
                      and not annotations['missing_geometry_annotations'] and not annotations['missing_phase_annotations']
                      and not annotations['all_unique_error_statistics']['over_10mm'])
    diagnostics_valid=not any(value['diagnostic_error_rows'] for value in variant_counts.values())
    policy=protocol.get('operator_calibration',{})
    sigma=finite(policy.get('sigma_m'))
    protocol_valid=(protocol.get('evaluation_revision')=='range_0p1_9m_fixed_10mm'
                    and protocol.get('measurement_range_m')==[.1,9.]
                    and finite(policy.get('error_limit_m'))==.01
                    and sigma is not None and abs(sigma-.01/3)<1e-12)
    return dict(evaluation_revision=protocol.get('evaluation_revision'),
                generated_at=datetime.now(timezone.utc).isoformat(),audit_integrity_passed=bool(structural and annotation_valid and diagnostics_valid and protocol_valid),
                integrity_components=dict(structural=structural,annotation_budget=annotation_valid,diagnostics=diagnostics_valid,protocol=protocol_valid),
                meaning='Integrity/budget audit only; ordinary measurement errors and no-output failures do not invalidate this audit if fully retained.',
                sampling_and_pairing=integrity,measurement_outcomes=variant_counts,
                calibration_annotations=annotations,phase_extrapolation=extrapolation_audit(rows,unique,protocol),
                protocol=dict(measurement_range_m=protocol.get('measurement_range_m'),
                              operator_calibration=protocol.get('operator_calibration'),
                              phase_parameters=protocol.get('phase_parameters')))


def self_test():
    rows=[]
    def annotation(reference,error):
        return dict(reference_m=reference,annotated_m=reference+error,error_m=error,error_mm=error*1000,
                    error_limit_mm=10.,sigma_mm=10/3,noise_applied=error!=0)
    for variant in VARIANTS:
        for index,h in enumerate((.5,.99)):
            error=0. if variant=='ideal' else .003
            row=dict(frame=f'phase-{index}',category='transparent',cohort='single',calibration_variant=variant,
                     installation_id=7,controls=dict(material='homogeneous_transparent',capacity_m=1.),
                     truth=dict(liquid_height_m=h),liquid=h+.002,highest=None,depth_band=0,
                     qualification=dict(eligible=True),inference_executed=True,wall_ms=50.,
                     phase_test_above_highest_prior=h>.95,
                     calibration_audit=dict(metric_annotations={name:annotation(1.,error) for name in GEOMETRY_ANNOTATIONS}))
            if index==0:
                row['phase_prior_annotation']=dict(metric_annotations={name:annotation(fraction,error)
                    for name,fraction in zip(PHASE_ANNOTATIONS,(.1,.35,.65,.95))})
            rows.append(row)
    result=audit_rows(rows,dict(phase_parameters=dict(priors_fraction=[0,.1,.35,.65,.95])))
    assert result['sampling_and_pairing']['unique_images']==2
    assert result['sampling_and_pairing']['complete_pairs']==2
    assert result['measurement_outcomes']['operator_noise']['qualified_missing_highest_raw']==2
    assert result['measurement_outcomes']['operator_noise']['qualified_missing_highest_geometry']==0
    geometry=result['calibration_annotations']['geometry']['by_variant']['operator_noise']
    assert geometry['n']==5 and abs(geometry['mean_error_mm']-3.)<1e-9 and geometry['std_error_mm']<1e-9
    assert result['calibration_annotations']['phase']['by_variant']['operator_noise']['n']==4
    above=result['phase_extrapolation']['by_material']['transparent']['by_variant']['operator_noise']['above_true_prior']
    assert above['rows']==1 and abs(above['mae_mm']-2.)<1e-9
    rows[-1]=dict(rows[-1],liquid=None,error='retained initialization failure',inference_executed=False)
    result=audit_rows(rows,{})
    assert result['measurement_outcomes']['operator_noise']['qualified_missing_liquid']==1
    assert result['phase_extrapolation']['by_material']['transparent']['by_variant']['operator_noise']['above_true_prior']['missing']==1
    assert error_statistics([11.])['over_10mm']==1 and band_of(.1)==0 and band_of(9.)==8
    json.dumps(result,allow_nan=False)
    full=[]
    for category in CATEGORIES:
        local=0
        for band,((low,high),quota) in enumerate(zip(BANDS,QUOTAS)):
            for repeat in range(quota):
                h=(low+high)/2
                for variant in VARIANTS:
                    error=0. if variant=='ideal' else .003
                    material='homogeneous_'+category if category in ('transparent','translucent') else 'ordinary_diffuse'
                    record=dict(frame=f'{category}-{local}',category=category,
                                cohort='single' if category in SINGLES else 'composite',calibration_variant=variant,
                                installation_id=7,controls=dict(material=material,capacity_m=9.2,bottom_m=10.),
                                truth=dict(liquid_height_m=h),liquid=h+.002,
                                highest=None if category in ('transparent','translucent') else h,
                                depth_band=band,qualification=dict(eligible=True),inference_executed=True,wall_ms=50.,
                                calibration_audit=dict(metric_annotations={name:annotation(value,error) for name,value in
                                    zip(GEOMETRY_ANNOTATIONS,(10.,9.2,.3,.6,.6))}))
                    if category in ('transparent','translucent'):
                        record['phase_prior_annotation']=dict(metric_annotations={name:annotation(9.2*fraction,error)
                            for name,fraction in zip(PHASE_ANNOTATIONS,(.1,.35,.65,.95))})
                    full.append(record)
                local+=1
    protocol=dict(evaluation_revision='range_0p1_9m_fixed_10mm',measurement_range_m=[.1,9.],
                  operator_calibration=dict(error_limit_m=.01,sigma_m=.01/3))
    full_result=audit_rows(full,protocol)
    assert full_result['audit_integrity_passed']
    assert full_result['sampling_and_pairing']['overall_band_counts']==[300]+[275]*8
    assert full_result['calibration_annotations']['unique_annotation_count']==26
    full[1]=dict(full[1],liquid=None,error='retained model failure')
    failed_result=audit_rows(full,protocol)
    assert failed_result['audit_integrity_passed']
    assert failed_result['measurement_outcomes']['operator_noise']['qualified_missing_liquid']==1
    full[1]=dict(full[1],diagnostic_error=dict(reason='must_block_complete'))
    assert not audit_rows(full,protocol)['audit_integrity_passed']
    print('FINAL_RESULTS_AUDIT_SELF_TEST_OK')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('LIQUID_EVAL_ROOT',str(Path(__file__).resolve().parent))))
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--strict',action='store_true',help='Exit1 on integrity/budget/diagnostic failure, not ordinary numerical underperformance')
    args=parser.parse_args()
    if args.self_test:self_test();return
    root=args.root.resolve()
    source=root/'measurements.jsonl';protocol_path=root/'protocol.json'
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    result=audit_rows(rows,protocol)
    result['input_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (source,protocol_path)}
    result['completion_marker_present']=(root/'measurements.complete').exists()
    output_path=root/'final-results-audit.json'
    output_path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(path=str(output_path),audit_integrity_passed=result['audit_integrity_passed'],
                          pairs=result['sampling_and_pairing']['complete_pairs']),ensure_ascii=False))
    if args.strict and not result['audit_integrity_passed']:raise SystemExit(1)


if __name__=='__main__':main()
