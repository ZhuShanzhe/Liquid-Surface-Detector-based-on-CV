"""Attach the independent audit to the report without changing metric tables."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
path=root/'performance.json';data=json.loads(path.read_text());audit_path=root/'final-results-audit.json';audit=json.loads(audit_path.read_text())
assert audit['audit_integrity_passed'] and audit['completion_marker_present']
for name,digest in audit['input_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
data['final_results_audit']=audit
data['input_files_sha256']['final-results-audit.json']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
data['report_augmentation_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
notes=[
 '本轮主体为按预声明液深分带选择的旧输入与新增输入，共同在固定±10mm标定协议下重新测量；不是全部新增独立图像。参数化扰动不代表实测工业传感器误差分布。',
 '新增安装长宽比0.75–1.35、偏心±0.15倍半径，且主动从完整槽口可见的可实现范围抽样；旧输入维持原几何。协议总包络0.65–1.45、±0.22倍半径为新旧并集，不能误述为新增样本的分布。统一资格允许可见底面或部分槽口加独立尺寸，区别于新增安装构造约束。',
 '高液深端受相机到底面不超过10m和构图约束，新增容器偏向细长几何；实际容器尺寸分布需结合输入审计阅读，不能把这批9m液深结果推广为所有宽大型深槽表现。',
 '透明/半透明五帧先验采集高度为容量C的0、10%、35%、65%、95%。新增8–9m液深的容量为H+0.08至0.12m，因此这些高液深测试超过名义最高先验约0.286–0.374m；属于先验外推。独立审计另列插值/外推及带误差标签上界，不把这部分差异全部归因于测距量程。',
 '固定±10mm是标注误差的绝对截断上限，截断前σ约3.33mm，不是输出MAE、RMS或最坏情况下的液深保证误差；输入合法性重标可能使最终接受分布略变，实际去重分布另列。',
 '计时只含暖态基础资格检查、分类和测量，不含离线真实回波来源审计、初始化、曝光采集、磁盘、网络、渲染和界面。输出覆盖率是证据充分合格输入上的条件比例，不是任意实地视频流的总体覆盖率。',
]
existing=[n for n in data['scope_notes'] if not n.startswith('本报告主体为多样化新独立仿真输入') and n not in notes]
existing=[n.replace('半径范围固定而不为构图缩成细管','旧容器不缩放，新增半径从可实现构图范围抽样').replace('不作完整0–9m均匀认证','已覆盖0.1–9m分带样本，不是连续逐点或真机认证') for n in existing]
data['scope_notes']=notes+existing
path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
report=root/data['report_file'];assert report.parent==root
marker='\n.. BEGIN_INDEPENDENT_FINAL_AUDIT\n'
base=report.read_text().split(marker)[0]
lines=['独立后验审计补充','==========================','']+notes+['']
for domain in ('geometry','phase'):
    stat=audit['calibration_annotations'][domain]['by_variant']['operator_noise']
    lines.append(f"{domain}唯一标注量n={stat['n']}，平均误差{stat['mean_error_mm']:.4f}mm，标准差{stat['std_error_mm']:.4f}mm，最大绝对误差{stat['max_abs_error_mm']:.4f}mm，超过10mm={stat['over_10mm']}。")
lines.append('透明/半透明的插值与外推完整计数、误差和通过率参见同名PDF审计表及final-results-audit.json。')
report.write_text(base+marker+'\n'+'\n\n'.join(lines)+'\n')
print('AUDIT_ATTACHED_TO_REPORT')
