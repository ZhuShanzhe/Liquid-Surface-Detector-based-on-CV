"""Error histograms with explicit all-frame denominators and no-solution bins."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ABS_EDGES=(0,.005,.01,.02,.05,.1,.2,float('inf'))
REL_EDGES=(0,.005,.01,.02,.03,.05,.1,float('inf'))
def band(h,nominal=None):
    if nominal==0 or h==0:return 'empty'
    if h<.1:return '0-0.1'
    if h<1:return '0.1-1'
    return f'{int(h)}-{int(h)+1}'
def summarize(rows,method):
    n=len(rows);available=[r for r in rows if method in r['methods']]
    q=[r for r in available if r['methods'][method]['numeric']]
    ps=[r['methods'][method] for r in q];e=np.array([p['absolute_error_m'] for p in ps])
    rel=np.array([p['abs_rel'] for p in ps if p['abs_rel'] is not None]);signed=[p['signed_error_m'] for p in ps]
    hist=lambda x,edges:[int(v) for v in np.histogram(x,edges)[0]]
    good=sum(p['within_tolerance'] for p in ps)
    return dict(frames=n,applicable_frames=len(available),numeric_frames=len(q),clusters=len(set(r['cluster'] for r in rows)),
        numeric_coverage=len(q)/n if n else None,missing_frames=n-len(q),mae_mm=float(e.mean()*1000) if len(e) else None,
        median_mm=float(np.median(e)*1000) if len(e) else None,p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,
        max_mm=float(e.max()*1000) if len(e) else None,signed_bias_mm=float(np.mean(signed)*1000) if ps else None,
        abs_rel_percent=float(rel.mean()*100) if len(rel) else None,relative_evaluable_frames=len(rel),
        numeric_pass_percent=100*good/len(q) if q else None,all_frame_pass_percent=100*good/n if n else None,
        absolute_bins_counts=hist(e,ABS_EDGES),absolute_bins_percent_all=[100*v/n for v in hist(e,ABS_EDGES)],
        relative_bins_counts=hist(rel,REL_EDGES),relative_bins_percent_numeric=[100*v/len(rel) for v in hist(rel,REL_EDGES)] if len(rel) else None,
        no_solution_reasons=dict(Counter(reason for r in available if not r['methods'][method]['numeric'] for reason in r['methods'][method]['reasons'])),
        mean_mask_iou=float(np.mean([r['mask_iou'] for r in rows])) if rows else None,
        mean_mask_precision=float(np.mean([r['predicted_mask_precision'] for r in rows])) if rows else None,
        mean_raw_valid_fraction=float(np.mean([r['raw_surface_valid_fraction'] for r in rows])) if rows else None,
        mean_extrema_error_mm={k:float(np.mean([p['extrema_error_m'][k] for p in ps])*1000) if ps else None for k in ('min_m','max_m','p05_m','p95_m')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=json.loads((a.output/'test-raw.json').read_text())['records'];groups=defaultdict(list)
    for r in records:
        scene=r['scenario']+('/'+r['frame']['variant'] if r['scenario']=='depth_failure' else '')
        b=band(r['true_height_m'],r['nominal_height_m'])
        for key in (scene,scene+'/'+b,scene+'/'+r['sensor'],scene+'/'+b+'/'+r['sensor']):groups[key].append(r)
    result={key:{m:summarize(rows,m) for m in sorted(set(m for r in rows for m in r['methods']))} for key,rows in groups.items()}
    output=dict(measurand='area_mean_actual_liquid_height_above_interior_bottom_m',synthetic_only=True,
        test_frames=len(records),absolute_bins_mm=['[0,5)','[5,10)','[10,20)','[20,50)','[50,100)','[100,200)','>=200'],
        relative_bins_percent=['[0,.5)','[.5,1)','[1,2)','[2,3)','[3,5)','[5,10)','>=10'],
        note='Diagnostic numeric coverage, not production accepted coverage; no zero snap or clipping; relative error undefined at empty',
        groups=result)
    (a.output/'analysis.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    lines=['# 其他场景分液深误差区间测试','',f'共 {len(records)} 个测试帧条件。测量量为相对容器内底面的实际液深。',
        '','普通、低照、眩光、漂浮物及波动各 396 帧；失效另含原始、额外75%、90%、97%连续空洞。',
        '每个场景每液深格为9组独立几何配置，每组4次深度噪声重复；重复帧不能当作独立容器。',
        '理想底面、位姿、内参和容器轮廓；无新增液深先验，无训练，无产品默认变更。',
        'refined 为冻结的现有传感器域估计；波动使用冻结分区插值。未做时序恢复或实机资格验证。',
        '','## 场景汇总','', '| 场景 | 方法 | 有值率 % | MAE mm | P95 mm | 有值通过率 % | 全帧通过率 % |',
        '|---|---|---:|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.2f}'
    scenes=['ordinary','low_light','glare','floating_objects','uneven_surface']+['depth_failure/'+v for v in ('base','drop75','drop90','drop97')]
    for scene in scenes:
        for m,s in result[scene].items():
            lines.append('| '+ ' | '.join([scene,m,fmt(s['numeric_coverage']*100),fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['numeric_pass_percent']),fmt(s['all_frame_pass_percent'])])+' |')
    for scene in scenes:
        lines+=['',f'## {scene} 分液深结果','',
            '| 液深 m | 全帧数 | 有值率 % | MAE mm | P95 mm | AbsRel % | 全帧通过率 % |',
            '|---|---:|---:|---:|---:|---:|---:|']
        bands=['empty','0-0.1','0.1-1']+[f'{i}-{i+1}' for i in range(1,9)]
        for b in bands:
            s=result.get(scene+'/'+b,{}).get('refined')
            if not s:continue
            lines.append('| '+' | '.join([b,str(s['frames']),fmt(s['numeric_coverage']*100),fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['abs_rel_percent']),fmt(s['all_frame_pass_percent'])])+' |')
        lines+=['','各误差区间占全部测试帧的百分比；未输出单列，不计零误差。',
            '', '| 液深 m | <5 mm | 5–10 mm | 10–20 mm | 20–50 mm | 50–100 mm | 100–200 mm | ≥200 mm | 未输出 % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for b in bands:
            s=result.get(scene+'/'+b,{}).get('refined')
            if s:lines.append('| '+' | '.join([b]+[fmt(x) for x in s['absolute_bins_percent_all']]+[fmt(100*(1-s['numeric_coverage']))])+' |')
        lines+=['','相对误差仅对非空槽且有值帧统计，分母为本格可计算相对误差的帧数。',
            '','| 液深 m | <0.5% | 0.5–1% | 1–2% | 2–3% | 3–5% | 5–10% | ≥10% |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for b in bands[1:]:
            s=result.get(scene+'/'+b,{}).get('refined')
            if s:lines.append('| '+' | '.join([b]+[fmt(x) for x in (s['relative_bins_percent_numeric'] or [None]*7)])+' |')
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(frames=len(records),groups=len(result)),indent=2))
if __name__=='__main__':main()
