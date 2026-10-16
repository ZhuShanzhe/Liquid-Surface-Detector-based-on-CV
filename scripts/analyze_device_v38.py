"""Post-test cause diagnostics without changing inference or selecting thresholds."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from liquid_depth.device_profiles_v38 import metric_summary
from report_device_v38 import OUT,MAIN,NAMES,CAM,table,fmt

def main():
    records=json.loads((OUT/'letterbox/test-raw.json').read_text())['records'];results={}
    for profile in CAM:
      for scenario in NAMES:
        if scenario=='translucent_mild':continue
        for regime in ('nominal','adverse'):
          method=MAIN.get(scenario,'native_refined')
          rows=[r for r in records if (r['profile'],r['scenario'],r['regime'],r['variant'],r['population'])==(profile,scenario,regime,'base','uniform')]
          methods={m:metric_summary([r for r in rows if r['method']==m]) for m in {r['method'] for r in rows}}
          selected=[r for r in rows if r['method']==method]
          diagnostic=dict(methods=methods,mean_mask_iou=float(np.mean([r['mask_iou'] for r in selected])),
              mean_raw_valid_pct=100*float(np.mean([r['raw_valid_fraction'] for r in selected])),
              no_output_reasons=dict(Counter(x for r in selected if r['predicted_height_m'] is None for x in r['reasons'])))
          if scenario=='uneven_surface':
              diagnostic['extrema_mae_mm']={k:(float(np.mean([r['extrema_error_m'][k] for r in selected if r['predicted_height_m'] is not None])*1000) if any(r['predicted_height_m'] is not None for r in selected) else None)
                  for k in ('min_m','max_m','p05_m','p95_m')}
          results['|'.join([profile,scenario,regime])]=diagnostic
    far={};fusion={};failure={}
    for profile in CAM:
      for regime in ('nominal','adverse'):
        rr=[r for r in records if r['profile']==profile and r['scenario']=='ordinary' and r['regime']==regime]
        for pop in ('uniform','empty','near_zero_enrichment'):
            key='|'.join([profile,regime,pop]);rows=[r for r in rr if r['method']=='ordinary_profile' and r['population']==pop and r['camera_to_surface_m']>=5.5 and r['nominal_height_m']<=1]
            far[key]=metric_summary(rows)
        for method in ['ordinary_profile','ordinary_guarded4']:
            fusion['|'.join([profile,regime,method])]=metric_summary([r for r in rr if r['method']==method and r['repeat']==3 and r['population']=='uniform'])
        for variant in ['base','drop75','drop90','drop97']:
            rr=[r for r in records if (r['profile'],r['scenario'],r['regime'],r['variant'],r['method'],r['population'])==(profile,'depth_failure',regime,variant,'native_refined','uniform')]
            failure['|'.join([profile,regime,variant])]=dict(metric_summary(rr),actual_mean_missing_pct=100*(1-float(np.mean([r['raw_valid_fraction'] for r in rr]))))
    obj=dict(scene_diagnostics=results,far_shallow=far,fusion_same_endpoints=fusion,depth_failure=failure)
    (OUT/'cause-diagnostics.json').write_text(json.dumps(obj,indent=2,allow_nan=False))
    lines=['# 相机档案试验诊断表\n','## 选点与传感器有效回波\n',
       '以下仅统计0–9 m等量主采样。“GT掩码诊断”提供离线对照：它仍使用含噪深度，只替换液面位置选择，不能作为部署成绩。V8列使用网络预测深度，拟合列使用原生传感器深度。\n']
    rows=[]
    for key,d in results.items():
        profile,sc,regime=key.split('|')
        if regime!='nominal':continue
        m=d['methods'];main=m[MAIN.get(sc,'native_refined')]
        rows.append([CAM[profile],NAMES[sc],fmt(d['mean_mask_iou']),fmt(d['mean_raw_valid_pct']),fmt(main['mae_mm']),
             fmt(m['v8_neural']['mae_mm']),fmt(m['native_refined']['mae_mm']),fmt(m['oracle_mask_diagnostic']['mae_mm']),main['no_output']])
    lines.append(table(['相机','场景','掩码IoU','有效回波%','选定链路MAE mm','V8 MAE mm','原生拟合MAE mm','GT掩码MAE mm','未输出'],rows)+'\n')
    lines+=['## 远距离浅液位\n','此处相机到液面≥5.5 m，真实液深≤1 m。空槽、厘米级加密和均匀主采样分别统计，避免相对误差在零附近的分母效应掩盖绝对误差。\n']
    rows=[]
    for key,s in far.items():
        profile,regime,pop=key.split('|')
        rows.append([CAM[profile],regime,pop,s['frames'],s['clusters'],fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['absrel_positive_pct']),fmt(s['pass_pct'])])
    lines.append(table(['相机','噪声','采样组','帧','独立配置','MAE mm','P95 mm','相对误差%','通过%'],rows)+'\n')
    lines+=['## 相同末帧的四帧融合对照\n','只比较第4帧及其因果历史窗口，四帧方法的样本分母与单帧一致。当前ToF分支沿用单帧，因而结果应一致。\n']
    rows=[]
    for key,s in fusion.items():
        profile,regime,method=key.split('|');rows.append([CAM[profile],regime,method,s['frames'],fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['absrel_positive_pct']),fmt(s['pass_pct'])])
    lines.append(table(['相机','噪声','方法','帧','MAE mm','P95 mm','相对误差%','通过%'],rows)+'\n')
    lines+=['## 大面积失效的实际失效率\n','drop75/90/97表示在原有场景失效之上追加对应比例的空间遮挡，不能直接当成最终总失效率。实际缺失比例和输出误差如下；极严重失效仅保留诊断。\n']
    rows=[]
    for key,s in failure.items():
        profile,regime,variant=key.split('|')
        rows.append([CAM[profile],regime,variant,fmt(s['actual_mean_missing_pct']),fmt(s['mae_mm']),fmt(s['p95_mm']),fmt(s['numeric_output_pct']),fmt(s['pass_pct'])])
    lines.append(table(['相机','噪声','追加失效','实际缺失%','MAE mm','P95 mm','输出%','通过%'],rows)+'\n')
    (OUT/'cause-diagnostics.md').write_text('\n\n'.join(lines),encoding='utf-8')
    print('DIAGNOSTICS',len(results),len(far),flush=True)
if __name__=='__main__':main()
