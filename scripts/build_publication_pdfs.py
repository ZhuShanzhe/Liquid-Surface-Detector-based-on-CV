"""Build two audited Chinese PDFs and a machine-readable current scorecard."""
import gzip
import argparse
import hashlib
import html
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,LongTable,TableStyle,PageBreak,KeepTogether

HERE=Path(__file__).resolve().parent
_args=argparse.ArgumentParser(description=__doc__)
_args.add_argument('--input-root',type=Path,default=HERE/'inputs')
_args.add_argument('--output-dir',type=Path,default=HERE/'output')
_options=_args.parse_args()
INPUT=_options.input_root.resolve()
OUT=_options.output_dir.resolve();OUT.mkdir(exist_ok=True,parents=True)
E=INPUT/'evidence'
BASE=E/'determined-all-scenes-v1-20261016'
SPECIAL=E/'special-scene-strata-v1-20261016/evaluation-meaningful-balanced'
NAMES={'ordinary_diffuse':'普通漫反射','partial_depth_loss':'部分深度失效',
 'homogeneous_transparent':'均匀透明','homogeneous_translucent':'轻度均匀半透明',
 'floating_objects':'漂浮物','low_light':'低照','glare':'眩光','uneven_surface':'不平整液面',
 'all':'合并','tof':'ToF','structured_light':'结构光','active_stereo':'主动双目',
 'stereo_IR_image_proxy':'左右目/IR图像代理','mild':'低','moderate':'中','severe':'强',
 'low_extinction':'低消光','medium_extinction':'中消光','dry':'空槽'}
SCENES=('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent',
 'partial_depth_loss','low_light','glare','uneven_surface','floating_objects')
DEPTHS=('空槽','0<H<0.01','0.01-0.1','0.1-1','1-2','2-3','3-4','4-5','5-6','6-7','7-8','8-9')

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def depth_bin(h):
    if h==0:return '空槽'
    if h<.01:return '0<H<0.01'
    if h<.1:return '0.01-0.1'
    if h<1:return '0.1-1'
    return f'{min(int(h),8)}-{min(int(h),8)+1}'
def tol(scene,h,peak=False):
    return max(.05,.05*h) if peak or scene=='uneven_surface' else max(.02,.02*h)
def f(x,d=2):return '-' if x is None else f'{x:.{d}f}'

def load_rows():
    rows=[]
    sources=[BASE/'depth-grid/records.jsonl',BASE/'refraction-frozen-records.json',
             BASE/'dedicated-floating-v8-records.json',SPECIAL/'records.jsonl']
    for file in sources:
        original=read(file) if file.suffix=='.json' else [json.loads(s) for s in file.read_text().splitlines()]
        for r in original:
            if file==sources[2] and r['truth']['liquid_height_m']==0:continue
            scene=r.get('scene',r.get('scenario'))
            if scene not in SCENES:continue
            h=r['truth']['liquid_height_m']
            level=r.get('level','all')
            if scene=='homogeneous_translucent':level='low_extinction' if r['config_id'].endswith('_1') else 'medium_extinction'
            loss=r.get('actual_loss_pct')
            loss_bin='all' if loss is None else f'{min(int(loss//10)*10,40)}-{min(int(loss//10)*10,40)+10}%'
            q={k:r.get(k) for k in ('sensor','camera_to_bottom_m','liquid_depth_m','highest_height_m','algorithm_ms','case','repeat','config_id','sample_id')}
            q.update(scene=scene,truth=r['truth'],depth_bin=depth_bin(h),level=level,loss_bin=loss_bin,
                cohort=('expanded_meaningful_4752' if file==sources[3] else 'legacy_determined_grid' if file==sources[0]
                        else 'frozen_refraction_8016' if file==sources[1] else 'floating_v8_wet_360'),
                provenance_file=str(file.relative_to(INPUT)))
            if q['case'] is None:q.pop('case')
            if r.get('input_evidence'):q['input_evidence']=r['input_evidence']
            rows.append(q)
    return rows,sources

def metric(rows,peak=False):
    key='highest_height_m' if peak else 'liquid_depth_m'
    tk='highest_height_m' if peak else 'liquid_height_m'
    n=len(rows);good=[r for r in rows if r.get(key) is not None and np.isfinite(r[key]) and tk in r['truth']]
    errors=np.array([abs(r[key]-r['truth'][tk]) for r in good])
    rel=np.array([abs(r[key]-r['truth'][tk])/r['truth'][tk] for r in good if r['truth'][tk]>0])
    ok=sum(abs(r[key]-r['truth'][tk])<=tol(r['scene'],r['truth'][tk],peak) for r in good)
    return dict(n=n,numeric=len(good),coverage=100*len(good)/n if n else None,
        passed=ok,pass_pct=100*ok/n if n else None,
        mae_mm=float(errors.mean()*1000) if len(errors) else None,
        p95_mm=float(np.percentile(errors,95)*1000) if len(errors) else None,
        max_mm=float(errors.max()*1000) if len(errors) else None,
        absrel_pct=float(rel.mean()*100) if len(rel) else None,relative_n=len(rel),
        absolute_bins_pct=(np.histogram(errors,[0,.005,.02,.05,.1,np.inf])[0]*100/n).tolist() if n else [],
        relative_bins_pct=(np.histogram(rel,[0,.01,.02,.05,.1,np.inf])[0]*100/len(rel)).tolist() if len(rel) else [],
        no_output=n-len(good))

def select(rows,scene,sensor='all',level='all',depth='all',loss='all',subset='all'):
    q=[r for r in rows if r['scene']==scene and (sensor=='all' or r['sensor']==sensor)
       and (level=='all' or r['level']==level) and (depth=='all' or r['depth_bin']==depth)
       and (loss=='all' or r['loss_bin']==loss)]
    if subset=='far_shallow':q=[r for r in q if r['camera_to_bottom_m']>=6 and 0<r['truth']['liquid_height_m']<=1]
    if subset=='far_dry':q=[r for r in q if r['camera_to_bottom_m']>=6 and r['truth']['liquid_height_m']==0]
    return q

def summarize(rows):
    groups={}
    for scene in SCENES:
        ss=[r for r in rows if r['scene']==scene]
        sensors=['all',*sorted({r['sensor'] for r in ss})]
        levels=['all',*sorted({r['level'] for r in ss}-{'all'})]
        losses=['all',*sorted({r['loss_bin'] for r in ss}-{'all'})]
        for sensor,level,loss,depth in itertools.product(sensors,levels,losses,['all',*DEPTHS]):
            q=select(rows,scene,sensor,level,depth,loss)
            if not q:continue
            k='|'.join((scene,sensor,level,loss,depth))
            groups[k]={'liquid':metric(q)}
            if scene in ('floating_objects','uneven_surface'):groups[k]['highest']=metric(q,True)
        for sensor in sensors:
            for part in ('far_shallow','far_dry'):
                q=select(rows,scene,sensor,subset=part)
                groups['|'.join((scene,sensor,'all','all',part))]={'liquid':metric(q)}
                if scene in ('floating_objects','uneven_surface'):groups['|'.join((scene,sensor,'all','all',part))]['highest']=metric(q,True)
    return groups

def clean(text):
    return str(text).replace('–','-').replace('—','-').replace('‑','-').replace('−','-').replace('`','')

def setup_fonts():
    pdfmetrics.registerFont(TTFont('CN',os.environ.get('PDF_CN_FONT','C:/Windows/Fonts/msyh.ttc'),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNB',os.environ.get('PDF_CN_BOLD_FONT','C:/Windows/Fonts/msyhbd.ttc'),subfontIndex=0))

STYLES={}
def p(text,style='body'):
    return Paragraph(html.escape(clean(text)).replace('\n','<br/>'),STYLES[style])
def heading(story,text):story.extend([p(text,'h1'),Spacer(1,7)])
def para(story,text):story.extend([p(text),Spacer(1,7)])
def page(story,title):
    if story:story.append(PageBreak())
    heading(story,title)
def table(headers,data,widths,size=9):
    items=[[p(x,'th') for x in headers]]+[[p(x,'cell') for x in row] for row in data]
    t=LongTable(items,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#203957')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F9')]),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#203957')),
        ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#D8E1EA')),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return t

def footer(canvas,doc):
    canvas.saveState();w,h=doc.pagesize
    canvas.setStrokeColor(colors.HexColor('#D5DFE9'));canvas.line(32,29,w-32,29)
    canvas.setFont('CN',8);canvas.setFillColor(colors.HexColor('#62748A'))
    canvas.drawString(32,17,'RGB-D 液深系统 | 2026-10-16 | 仿真研发证据，非工业计量认证')
    canvas.drawRightString(w-32,17,f'第 {doc.page} 页');canvas.restoreState()

def document(path,story,portrait=False,title=''):
    class OutlinedDocument(SimpleDocTemplate):
        def afterFlowable(self,flowable):
            if isinstance(flowable,Paragraph) and flowable.style.name=='h1':
                key='section_'+str(getattr(self,'_section_number',0))
                self._section_number=getattr(self,'_section_number',0)+1
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(flowable.getPlainText(),key,level=0,closed=False)
    d=OutlinedDocument(str(path),pagesize=A4 if portrait else landscape(A4),
        leftMargin=36,rightMargin=36,topMargin=34,bottomMargin=40,
        title=title,author='Liquid Surface Detector Project',subject='Verified simulation evidence and development record')
    d.build(story,onFirstPage=footer,onLaterPages=footer)

def metric_line(m):return [m['n'],f(m['mae_mm']),f(m['p95_mm']),f(m['max_mm']),f(m['pass_pct']),f(m['coverage']),f(m['absrel_pct'])]
MH=['帧数','MAE mm','P95 mm','最大 mm','通过 %','有值 %','AbsRel %']

def make_system(rows,groups):
    story=[]
    page(story,'RGB-D 液深系统 正式链路与全场景测试报告')
    para(story,'发布标识 formal-20261016。本报告统一整理当前八类确定场景、代码入口、最新测试证据、分层误差，以及远距浅液位的不足、历史修复与当前实际处理。此次只整理、计分、封装和发布，不训练或改变已确定的测量算法。相对旧 GitHub 主线，本次发布也包含此前尚未提交的项目研发代码。')
    para(story,'结论：当前存在可复现的全场景研究入口，但不能称全部场景、全部相机、全部量程均已工业达标。普通/部分失效的远距浅液位仍弱；低照和眩光的新有效输入测试改善了统计覆盖；透明/半透明成绩仅属于可见底纹的左右目/IR 图像代理，不是 ToF 折射成绩。')
    story.append(table(['场景','证据组',*MH],[[NAMES[s],{'ordinary_diffuse':'旧确定版网格','partial_depth_loss':'旧确定版0-50%网格','floating_objects':'V8有液专项','homogeneous_transparent':'既有全记录重计','homogeneous_translucent':'既有全记录重计'}.get(s,'新有效输入均衡集'),*metric_line(metric(select(rows,s)))] for s in SCENES],[85,120,38,55,55,55,55,55,65]))
    para(story,'上表不同证据组不相加为一个总准确率。低照/眩光容差 max(20 mm, 2%H)；不平整液深改用 max(50 mm, 5%H)；其他液深 max(20 mm, 2%H)。最高高度统一 max(50 mm, 5%实际最高高度)。H 均相对容器内底面，波面真值用物理面积均值，不使用相机深度作为相对误差分母。')

    page(story,'正式链路结构与代码边界')
    para(story,'同步采集/录制回放 → 输入单位与对齐校验 → 空槽/已知液深先验初始化 → 操作者选择场景 → 确定版测量 → 液深、可选最高高度、诊断及版本输出。数据采集、模型、几何、图形面板彼此分离；已有面板仍有自身旧产品运行端，本次没有宣称已经完成所有新算法到 GUI/USB 硬件的自动接入。')
    story.append(table(['模块','当前入口或位置','实际职责'],[
      ['统一测量','src/liquid_depth/determined_scene_meter.py','八类场景显式路由；不自动可靠分类；不将离线真值用于选点或方法选择。'],
      ['空槽与几何','empty_scene_routes_v44.py / empty_geometry_v41.py','空槽深度与少量独立米制信息构建容器底面、范围、相机-重力坐标；相机移动后重新标定。'],
      ['主液深','frozen_symmetric_geometry.py / symmetric_visual_meter.py','冻结对称候选，底面与液面同等比较；有支持时估计，没有候选时调用原正式链路回退。准确文件清单见代码索引。'],
      ['透明先验','supported_scene_meter.py / range_relative_v32.py / range_registration_v33.py','1 组空槽 + 4 组已知液深左右目图像；透明 baseline_v32，半透明 midpoint_dense。'],
      ['不平整最高','uneven_surface_meter.py / uneven_surface_repair.py','几何最高包络及既有 V11 修复；不等于所有隐藏波峰可见。'],
      ['可选视觉','floating_rank_application_v8.py / floating_rank_v8.py','仅漂浮物最高高度；液深不受影响。ToF/结构光走 V5 选定分支，主动双目走 V8 策略，失败按既有规则回退。'],
      ['历史与实验','scripts/ 与 docs/ 中版本化研究入口','V37/V39/V40/V72、照明 V2、V9 等保留以便复现，不因代码存在而视作当前默认启用。']],[100,270,330]))
    para(story,'发布代码的准确依赖列表由 runtime-modules.json 和新的发布清单给出。旧源码名不强行改名或移动，避免破坏导入与复现实验。新便携命令行包装仅解决路径与小模型资源定位，不改变推理数值。')

    page(story,'场景与双相机适配矩阵')
    story.append(table(['场景','ToF','结构光','测量条件与边界'],[
      ['普通漫反射','已接入并有仿真','已接入并有仿真','需要可靠表面回波和空槽标定；可见颜色不等于 IR 漫反射性质。'],
      ['部分失效','0-50%范围有仿真','0-50%范围有仿真','缺失率指可见液面内数值空洞率；受控删点不是独立容器。超过50%不作适配承诺。'],
      ['低照 / 眩光','新均衡分层仿真','新均衡分层仿真','低/中/强等额。RGB亮度和深度退化为联合参数代理，不是实测 lux/IR 功率档。'],
      ['不平整','新三档波幅仿真','新三档波幅仿真','平均液深和最高高度分别评分；本次新集不混入漂浮物。'],
      ['漂浮物','V8接口内选定V5分支','V8接口内选定V5分支','视觉只算最高高度；不宣称 ToF/结构光实际执行了主动双目 V8 学习排序头。'],
      ['均匀透明 / 半透明','未适配、无专线成绩','缺少专用正式结构光硬件评测','现有高通过率来自左右目/IR图像代理；可参考三角测量原理，不能移贴到 ToF 或真实相位结构光。'],
      ['多层透明 / 极端复合','当前正式版不承诺','当前正式版不承诺','历史 SeeGroup/多层投票等研究保留；95-100%失效及未知不可观测情况不在本轮继续优化。']],[115,150,170,265]))
    para(story,'相机建议：长距可优先实测长距 ToF；近中距主动结构光仍有价值。眩光不应仅凭本批代理模型直接定型，因为 ToF 同样可能受镜面、多径与饱和影响。Astra 2 官方范围为 0.6-8 m，不能被本项目 10 m 代理试验扩展成硬件规格；blaze 101 的 0.3-10 m 范围也不代表全程均满足其 0.5-5.5 m 下列示的典型精度。')
    para(story,'官方参考：docs.baslerweb.com/influencing-factors；docs.baslerweb.com/blaze-101；www.orbbec.com/products/structured-light-camera/astra-2/；www.orbbec.com/blog/structured-light-vs-itof-depth-cameras/。')

    page(story,'初始化 操作流程与部署状态')
    for title,body in [
      ('普通及非透明特殊场景','固定机位采集真实空槽深度，确认内参、米制单位与 RGB-D 对齐；操作员选容器轮廓并输入少量独立尺寸/底面垂距，完成底面、重力与范围核验；再给当前深度并显式选择场景。无需把 CAD 当作每次固定机位部署的刚需。'),
      ('透明与轻度半透明','同机位获取一组空槽和四组不同已知实际液深的左右目图像。先验液深必须在容器可达范围内、单位为米；代码不把位置百分比写死。先验需覆盖测量范围，保留独立液位作验收；底纹消失、相机/底纹变化或多界面混合时不能假定仍有效。'),
      ('漂浮物与最高高度','默认纯几何。只有操作者确认漂浮物场景并显式启用视觉后，才允许视觉改变最高高度；不得改写液深，不自动扩展到普通、低照或眩光。'),
      ('相机移动','偶发移动后重新采空槽并标定；持续移动的 CAD/实时位姿路线是既有产品研究功能，不属于本版固定机位仿真已验证承诺。'),
      ('发布运行','仓库根目录安装公开依赖，运行便携发布入口。无需USB可做回放。采集驱动、操作员面板、物理相机曝光/传输与现场控制仍需各自联调；算法调用低于0.5秒不等于整机端到端已认证。')]:
        para(story,title+'：'+body)
    para(story,'当前测量保留负值、容量外值、支持不足与候选竞争诊断，不强制归零、不以真实液深裁剪。10 m 是应用与测试布置约定，默认 10.5 m 原始返回档案与它分开；不是将所有接近10 m的数值直接清零。')

    page(story,'统计口径与证据组')
    story.append(table(['证据组','规模与来源','有效性与统计限制'],[
      ['普通/部分失效','99次安装，普通396帧，0-50%缺失2346个条件；既有确定版网格回放','使用原有全输入评分口径，不伪称已按本次新来源审核重采；不得将重复删点条件当作2346个独立容器。'],
      ['透明/半透明','48配置、8016组图像；2672透明+5344半透明；另288帧接口核验','采用冻结选定结果重计；不是8016次新配准，也不是 ToF 成绩。近零区间加密，不是连续0-9m均匀总体。'],
      ['漂浮物','99安装共396帧，正式有液专项360帧','保留36个干槽控制于来源归档；本报告有液专项与干槽不混作物体最高点识别成绩。'],
      ['低照/眩光/不平整','66物理安装、132相机标定，最终4752有效帧，三类各1584帧','每个相机、强度、实际液深单元24帧；物理实例6个，每个4次噪声重复。三档等额，按实例聚类而非帧数虚增独立性。']],[130,260,310]))
    para(story,'新集有效输入有两级审核：①仅用空槽标定与当前深度，要求64个容器内部回波、至少6/16空间区、两轴跨度均不低于25%；②仿真采集来源审核要求64个直接液面来源返回，并覆盖至少6/16可见液面区和两轴25%跨度；空槽改审底面来源。第二级只判断合成采集是否包含基本物理证据，不是实地已实现的在线观察能力。')
    para(story,'来源标签不进入测量算法；带大噪声或系统偏置的真实液面来源点仍保留，不按数值误差删帧。每个条件按固定种子取前4组 ToF/结构光共同合格输入，最多64次尝试，不降低强度、不换容器。共2566组尝试、2376组入选、190组排除，成对采样成功率92.60%。强眩光只有60.27%，需要与筛选后精度同时看。')
    para(story,'MAE/P95/最大误差对有数值帧计算；通过率与有值率的分母是对应明确范围的全部正式帧，未输出计失败，不计零误差。AbsRel只对H>0帧计算，空槽另记绝对误差。绝对误差档为 [0,5)、[5,20)、[20,50)、[50,100)、≥100 mm；相对档为 [0,1)、[1,2)、[2,5)、[5,10)、≥10%。')
    para(story,'新集95%区间按每个实际液深层内的物理安装实例重采1000次，配对相机和重复帧不拆开。零失败样本的退化区间不是现实零失败保证。旧数据和新有效输入数据的纳入方式不同，因此不将它们的成绩变化称为算法提升。')

    page(story,'ToF 结构光与补充主动双目性能')
    camera_rows=[]
    for s in SCENES:
        for sensor in sorted({r['sensor'] for r in select(rows,s)}):
            camera_rows.append([NAMES[s],NAMES[sensor],*metric_line(metric(select(rows,s,sensor)))])
    story.append(table(['场景','相机/代理',*MH],camera_rows,[95,130,38,52,52,52,52,52,60]))
    para(story,'透明类没有 ToF 数值，不补零，也不把左右目/IR图像代理伪装为专用结构光验证。普通/部分失效/漂浮物的源汇总包含主动双目；低照/眩光/不平整新集只含 ToF 与结构光，合并相机集合不同。逐相机表与附录使两条要求关注的链路可单独核验。')

    page(story,'低 中 强档位的合并与双相机结果')
    intensity=[]
    for s in ('low_light','glare','uneven_surface'):
        for level in ('mild','moderate','severe'):
            for cam in ('all','tof','structured_light'):
                intensity.append([NAMES[s],NAMES[level],NAMES[cam],*metric_line(metric(select(rows,s,cam,level)))])
    story.append(table(['场景','强度','相机',*MH],intensity,[85,35,90,35,50,50,50,50,50,65]))
    page(story,'部分失效缺失率与透明材料层级')
    lost=[]
    for loss in ('0-10%','10-20%','20-30%','30-40%','40-50%'):
        for cam in ('all','tof','structured_light'):
            lost.append([loss,NAMES[cam],*metric_line(metric(select(rows,'partial_depth_loss',cam,loss=loss)))])
    story.append(table(['实际数值空洞率','相机',*MH],lost,[110,120,35,50,50,50,50,50,65]))
    para(story,'区间左闭右开，最后一档含50%端点。最后一档同时含40%与50%受控条件，所以数量较多；不同档不是等量独立容器。')
    semi=[]
    for s,level in [('homogeneous_transparent','all'),('homogeneous_translucent','low_extinction'),('homogeneous_translucent','medium_extinction')]:
        semi.append([NAMES[s],NAMES.get(level,level),*metric_line(metric(select(rows,s,level=level)))])
    story.append(table(['液体类型','消光层级',*MH],semi,[110,120,35,50,50,50,50,50,65]))
    page(story,'各类总体误差区间占比')
    distribution=[]
    for s in SCENES:
        mm=metric(select(rows,s))
        distribution.append([NAMES[s],mm['n'],mm['relative_n'],' / '.join(f(v,2) for v in mm['absolute_bins_pct']),
                             ' / '.join(f(v,2) for v in mm['relative_bins_pct'])])
    story.append(table(['场景','绝对分母','相对分母','绝对五档占比 %','相对五档占比 %'],distribution,[110,65,65,250,250]))
    para(story,'绝对五档：0-5、5-20、20-50、50-100、≥100 mm；相对五档：0-1、1-2、2-5、5-10、≥10%。相对分母排除空槽；有值覆盖与通过率仍按各证据组完整正式分母计算。')

    page(story,'最高高度与延时单列')
    peakrows=[]
    for s in ('floating_objects','uneven_surface'):
        for camera in ('all',*sorted({r['sensor'] for r in select(rows,s)})):
            peakrows.append([NAMES[s],NAMES[camera],*metric_line(metric(select(rows,s,camera),True))])
    story.append(table(['场景','相机',*MH],peakrows,[100,125,38,52,52,52,52,52,60]))
    para(story,'不平整最新有效集的液深按旧20mm/2%重计通过94.38%，按新50mm/5%为96.91%；这部分通过率变化是评价要求变化，不是算法改进。最高波峰真值与液深均值不同，不能互相替代。V8漂浮物仍可能漏最高实例、选择较低实例或校正共同偏置失败。')
    para(story,'冻结版透明/半透明实际接口回放的算法P95约271/388 ms，不含五帧先验初始化。漂浮物专项预热P95约493 ms，冷启动/逐安装回放P95约639 ms，首次加载可更慢。新特殊场景在多进程竞争下的计时附于JSON。以上不是含曝光、USB、网络、GUI和排队的端到端延时承诺。')

    page(story,'远距浅液位 当前性能与主要不足')
    far=[]
    for s in SCENES:
        for sensor in ('all',*sorted({r['sensor'] for r in select(rows,s)})):
            q=select(rows,s,sensor,subset='far_shallow')
            if q:far.append([NAMES[s],NAMES[sensor],*metric_line(metric(q))])
    story.append(table(['场景','相机',*MH],far,[95,130,38,52,52,52,52,52,60]))
    para(story,'统一远浅定义：相机至内底面垂距 B≥6m 且 0<H≤1m。H是液深，B为到底面距离，D为到液面距离；平整近顶视时 H≈B-D。大距离相减会放大共同测距偏差、标定法向偏差与尺度误差。空槽H=0单列，不能用空槽好成绩代替真实浅液位。')
    for body in [
      '结构光/双目远端视差小，距离噪声和量化非线性被放大。当前小基线通用代理不是高分辨率商用设备实测档案，不能把本批严重浅液位误差完全归罪于某个实体品牌。',
      '液面、底面错误回波和近零候选可能重叠；同等候选比较纠正了旧的不对称覆盖规则，但评分和支持窗口仍可能把正确液面混入底面峰或选到偏移峰。',
      '真实液面回波数量多，不代表米制距离已准确。空间相关噪声、公共偏置和同一空槽参考误差不会按像素数的平方根消失；加密插值或超分辨率也不是新的独立测量。',
      '底面法向、ROI和操作员尺寸先验可能先出错；应与液面识别失败分开。新集 ToF 液深误差与底面距离标定误差相关约0.93，说明不能把毫米级误差都归给液面网络。',
      '最高物体候选漏召回、候选全被拒和已有候选但测量不准是三种问题。它们是最高高度分支的问题，不应混称为液面回波全部缺失。']:
        para(story,body)

    page(story,'远距浅液位 历史修复尝试与是否启用')
    text=(INPUT/'docs/far_shallow_history_and_current_v1_20261016.md').read_text(encoding='utf-8')
    section=text.split('## 已做过的修复与结果')[1].split('## 现在对远距浅液位的实际处理')[0]
    history=[]
    for line in section.splitlines():
        if line.startswith('|') and not line.startswith('|---'):
            parts=[x.strip() for x in line.strip('|').split('|')]
            if parts[0]!='尝试':history.append(parts)
    story.append(table(['历史方法','对应试验结论与限制','当前确定版状态'],history,[175,360,190]))
    para(story,'表中早期结果有不同噪声、分辨率、先验和容差，多数旧试验用5mm/2%。保留原口径，不把它们横向相减当作当前提升；精度数值仅说明对应试验有效或失败，具体原记录见仓库远距历史文档及引用源。')

    page(story,'远距浅液位 现在的实际处理和下一步约束')
    para(story,'当前普通、部分失效、低照、眩光、不平整和漂浮物液深共用空槽建模及冻结对称几何：收集容器ROI内有限正深度；按传感器噪声代理形成高度候选；底面/液面按同等规则评分；有候选时给稳健代表液深，无候选时调用既有 FormalSceneMeter 回退。支持不足和竞争原因保留，不因知道真值误差而关闭数值。')
    para(story,'当前没有默认启用 V37 四帧联合估计、V39 公共偏置、V40 视差/空槽空间去噪、V72 量化档案、照明 V2 历史修复或 V9 多帧记忆。代码保留不等于链路启用。透明/半透明则使用五组标定图像的相对响应与原生对称配准，当前测量帧间也没有长时记忆。')
    para(story,'后续只优化远距浅液位：先分清先验错误、真实表面有噪声、错误物理层与候选裁决错误；建立与确定版同输入的配对消融。先验证设备距离噪声与量化档案，再比较空槽参考差分、共同偏置估计及静态2-4帧融合。针对持续相关误差须有独立约束，不能只做平均。')
    para(story,'分层固定保留：空槽、0-0.1m、0.1-0.5m、0.5-1m；按 B、D、相机机制、缺失面积和独立空间支持分组。选型使用新的安装级验证集，最终用新容器、位姿和噪声的独立测试确认；全量程其他区间和其他场景只做防回退。几毫米差异单独解释，不用通过率跳变冒充显著性能变化。')
    story.append(table(['其他问题','本轮简记'],[
      ['小漂浮物/错误实例','V8仍有漏检和共同测距偏差；视觉仅最高高度，不扩大到其他场景。'],
      ['不平整隐藏峰值','稳健代表液深未必严格等于面积均值，最高包络不能恢复不可见极值。'],
      ['强眩光/暗光','有数值不等于信息充分；RGB提亮不能修复失真的IR回波；报告采集可用性。'],
      ['多层/强散射/复合','不归入已完成的均匀透明路线；多层投票与旧置信策略仅作历史。'],
      ['工业集成','GUI、真实相机噪声、固件、实时同步与整机延时仍待现场独立验收。']],[170,535]))

    page(story,'附录导航与复现证据')
    para(story,'以下每页按实际液深分层，既给合并结果，也给 ToF、结构光及按强度/缺失率交叉的结果。N=0表示该证据组没有该区间样本，不填零误差。AbsRel对空槽不定义。绝对/相对误差五档占比均按本页标题的范围计算；非零样本的相对误差百分比不与空槽混合。')
    para(story,'为可读性，PDF给出所有场景合并液深层、要求关注的双相机分层，以及三档光照/波幅与五档缺失率交叉；主动双目的完整交叉、更多子集和全部逐帧数据位于仓库 reports/evidence/。低照/眩光/波面新集按安装聚类的置信区间还保留在原 stratified-summary.json，发布清单提供来源哈希。')
    para(story,'代码回归：原确定版719项；扩样输入审核后726项；本次增加3项发布资源测试后729项通过，14条依赖警告。便携入口八类场景均与归档原入口数值一致，漂浮物视觉开关不改变液深，其他场景即使传开关也不执行视觉。发布验证记录随仓库存档。本报告不把运行成功或软件著作权准备工作当作精度认证。')
    para(story,'主要证据：determined-all-scenes-v1-20261016（普通/缺失/透明/漂浮物确定版）；special-scene-strata-v1-20261016/evaluation-meaningful-balanced（最新4752帧）；far_shallow_history_and_current_v1_20261016.md（修复历史）；special_scene_expanded_evaluation_20261016.md（最新采集审核）。完整来源SHA256随仓库保存。')

    tasks=[]
    for s in SCENES:tasks.append((s,'all','all','all','合并全部强度/条件'))
    for s in ('ordinary_diffuse','partial_depth_loss','floating_objects'):
        for cam in ('tof','structured_light'):tasks.append((s,cam,'all','all','合并条件'))
    for s in ('low_light','glare','uneven_surface'):
        for cam in ('tof','structured_light'):
            for level in ('mild','moderate','severe'):tasks.append((s,cam,level,'all',NAMES[level]+'强度'))
    for cam in ('tof','structured_light'):
        for loss in ('0-10%','10-20%','20-30%','30-40%','40-50%'):tasks.append(('partial_depth_loss',cam,'all',loss,'缺失率'+loss))
    tasks=[(*task,False) for task in tasks]
    for s in ('floating_objects','uneven_surface'):
        for cam in ('all','tof','structured_light'):
            tasks.append((s,cam,'all','all','合并条件',True))
    for cam in ('tof','structured_light'):
        for level in ('mild','moderate','severe'):
            tasks.append(('uneven_surface',cam,level,'all',NAMES[level]+'强度',True))
    for s,cam,level,loss,caption,peak in tasks:
        page(story,NAMES[s]+' '+NAMES[cam]+' '+caption+(' 最高高度' if peak else ' 液深')+'分层')
        para(story,'绝对五档：0-5 / 5-20 / 20-50 / 50-100 / ≥100 mm；相对五档：0-1 / 1-2 / 2-5 / 5-10 / ≥10%。占比单位%。'+('本页是新采集有效输入集。' if s in ('low_light','glare','uneven_surface') else '本页为既有确定版证据，保留原始纳入口径。'))
        data=[]
        for dep in DEPTHS:
            mm=metric(select(rows,s,cam,level,dep,loss),peak)
            data.append([dep,*metric_line(mm),'/'.join(f(v,1) for v in mm['absolute_bins_pct']) or '-',
                         '/'.join(f(v,1) for v in mm['relative_bins_pct']) or '-'])
        story.append(table(['实际液深\nm','帧数','MAE\nmm','P95\nmm','最大\nmm','通过\n%','有值\n%','AbsRel\n%','绝对误差五档占比','相对误差五档占比'],data,[62,30,45,45,48,43,43,43,190,190]))
        if s in ('floating_objects','uneven_surface'):
            para(story,('本页评价最高高度，相对误差分母与容差采用真实最高高度；行分组仍按实际平均液深，不按最高高度重分层。' if peak else '本页为液深分层；最高高度另有专页，不与液深混用。完整交叉还保留在发布JSON中。'))
    document(OUT/'2026-10-16_system_and_scenario_evaluation.pdf',story,title='RGB-D液深系统 正式链路与全场景测试报告')

def make_history():
    story=[]
    page(story,'RGB-D 液深系统 开发过程与重要进展')
    para(story,'截至2026年10月16日。按实际开发顺序概述重要工作、路线变更、失败回退与当前状态。日期以已有Git记录、阶段文档和旧PDF正文为准；未从零散聊天臆造精确试验日期。这里只整理证据，不在本轮开展新的算法训练。')
    milestones=[
      ('2026-08-25','项目迁移与工程底座','将同步 RGB-D 采集、离线帧契约、相机驱动、深度修复、平面几何及服务器环境分开。建立Git仓库、RTX5090/CUDA兼容环境、数据注册与外置存储；大量原始数据和运行环境不进Git。'),
      ('2026-08-26','公开基线复现与多任务起点','复现 TransCG DFNet、DREDS SwinDRNet、ClearGrasp；LIDF因旧检查点不可得不再阻塞。共同真实对象测试倾向选择 SwinDRNet，但对象表面深度修复不是液体最终液深。完成RGB-D多任务预训练，输出掩码、深度、法向和不确定度。'),
      ('早期产品化阶段','标定与操作员面板','设计固定机位少量操作员标定、空槽/底面几何、相机移动后的重新标定；连续移动另需CAD和实时位姿。建立简洁工业操作面板并与算法分离，但后续新确定版算法与旧GUI不是自动完全耦合。'),
      ('2026-08-28','仿真与混合训练','构建10类液体场景的Eevee/Cycles仿真；混合ClearGrasp、TODD、TransCG等公开对象先验。纯仿真模型偏差较大，混合和高保真微调改善域差异，但未达到普适工业精度。'),
      ('2026-08-29','顶部视角与少样本标定','根据真实部署以容器顶部为主，扩大容器类型，采用5,000帧顶部困难数据及独立近垂直测试；引入现场尺度/偏置标定，明确少样本拟合与独立验收应分开。'),
      ('2026-09-04','V6置信度与普通场景优化','扩至12,000帧，进行场景均衡、置信度监督、分层评估；普通场景尝试尺度-偏置头、量程均衡、尾部与分位数损失。旧报告曾给“达标”结论，但部分液位统计后来确认实际是相机轴向深度代理，资格结论随后撤回。'),
      ('2026-09-05起','透明多层与SeeGroup思想','尝试多层深度解构、液面层选择、候选投票和物理约束。液体界面候选有改进，完整多层恢复和弱界面身份仍不足；多层透明不等于已完成任意透明容器加液体的测量。'),
      ('9月中旬','深度失效 时序与可观测性','研究可靠点记忆、对齐、RGB补全、距离相关噪声门控、置信度与错误接受控制；观察覆盖率和精度之间的权衡。用户明确95-100%严重失效不再继续恢复优化。'),
      ('2026-09-14至09-16','适配状态与尾部审核','形成分场景适配汇总及固定5cm审核。旧平均值与最大误差不能互相替代；多液位额外标定的旧好成绩不能套到单次空槽冻结测评。保留V19/V20研究记录，不能用测试集反选模型。'),
      ('2026-10-11','统一为相对底面的真实液深','将 H、相机轴深Z、到底面B以及层厚区分。旧V6等轴向代理“达标”撤回；按实际H重新统计。透明方向限定均匀、底纹可见的透明/轻半透明，不再将强散射混入该路线优化。'),
      ('2026-10-11','透明先验与相对位移路线','分析先验失败、过度经验校正、外推斜率、匹配偏置与相位误差；保持五组预算，验证受容器容量约束的先验。采用一组空槽加四组已知液深、跨先验相对配准，不再只靠错误点云拟液面。'),
      ('2026-10-12','量程扩展与透明薄弱段','实际液深上限改9m、相机到底面≤10m。发现空槽、0-1m、2-3m薄弱，采用单调响应反解、逐参考校准与鲁棒融合；半透明原生对称配准综合效果优于所测通用超分组合，保留 baseline_v32 / midpoint_dense。小容器不再追加优化。'),
      ('2026-10-12','普通远浅与相机噪声研究','试验结构光量化/截断似然、2-4帧估计、空间相关噪声、主流相机档案、公共偏置与空槽空间去噪。独立随机噪声可被多帧降低，共享偏置与漂移仍难解决；历史收益未全部接入后来冻结主链路。'),
      ('2026-10-13至10-14','空槽建模与部分失效','复用空槽和小规模米制输入建立容器，限定0-50%液面数值丢失。放宽数值输出，逐步研究弱掩码、候选回波、分位数、弹性证据与量化似然。液深容差改为max(20mm,2%H)，旧结果按新口径重计。'),
      ('2026-10-14至10-15','不平整 低照与眩光','分别评价液深与最高高度，最高容差50mm/5%；修正不合格标定先验，记录远距浅液位统一待办。探索空槽位移/因果历史，发现底面覆盖规则与液面候选地位不对称。'),
      ('2026-10-15','对称几何与视觉边界','让底面和液面候选同等评分，封存对称几何作为旧链路确定版。训练回波/区域可靠性与小漂浮物识别，最终将视觉严格限制为漂浮物最高高度，普通和照明场景不启用。'),
      ('2026-10-16','视觉V4至V9对比并恢复V8','比较最高实例、物理建模、失配检查、保边分片、连续高度风险、硬支撑及多帧。V9在新专项均值、通过率与延时综合不如V8，未合并；保留V8选定策略，ToF/结构光仍走选定V5分支，液深不变。'),
      ('2026-10-16','统一确定版与全量回放','统一八类场景入口，核验719项测试；明确当前不启用历史V37/V39/V40/V72等修复。记录全部场景性能、远距浅液位长尾、失败原因及下一阶段只优化远浅的约束。'),
      ('2026-10-16','特殊场景扩样与有效输入审核','低照、眩光和不平整各扩至1584帧，ToF/结构光、三档强度、11个原设计液深层等额。按用户要求排除无基本物理回波支撑的采集并同条件补采；最终4752有效帧，记录190组成对输入排除及来源审计，726项回归通过。'),
      ('2026-10-16 本次交付','正式发布与双PDF整理','汇总当前链路、全场景/分层数据、远浅修复史与实际启用状态；整理README、便携资源与复现实验索引，发布到main。旧进程资料和分支先做可恢复备份，桌面目标文件夹仅保留本记录与正式测试报告两份PDF。')]
    for i,(date,title,body) in enumerate(milestones,1):
        story.append(KeepTogether([p(f'{i:02d}  {date}  {title}','h2'),p(body),Spacer(1,10)]))
    page(story,'关键决策 当前状态与尚未完成的部分')
    for body in [
      '评价量纠正是关键转折：像素深度、相机轴向液面位置、底面相对实际液深和最高高度不是同一个量。旧报告指标不能直接累加，也不能跨不同先验预算比较。',
      '从“通用网络恢复所有像素”转向按可观测信息选路线：普通可靠回波用几何；透明底纹可见用少帧图像响应；漂浮物视觉只用于最高高度。多层透明、极端缺失和持续相机移动不冒充完成适配。',
      '支持充分但测错必须留在正式测试；真实液面来源少、空间支撑不足的采集单列，而不是按误差大小筛掉。此次来源审核只适用于可知物理来源的仿真，不是已部署的在线可观测性证明。',
      '当前已形成服务器可运行、无USB可回放的确定研究版及数据审计。尚需真实相机噪声标定、目标容器采集、硬件/GUI联调、工业安全与计量验收；软件著作权准备并不自动给出精度资格。',
      '后续研发优先级仅为远距浅液位：参考偏置、量化与候选身份、空间相关误差及静态多帧；其他场景以固定基线防回退，不重新启动多层、极端或小漂浮物扩展。']:
        para(story,body)
    page(story,'来源与历史资料保留')
    para(story,'按顺序参考Git早期工程提交与 baseline_results、simulation_results_v1/v2_top/v3_confidence，旧进程PDF（2026-09-04、09-14、09-16，以及普通/透明优化报告），2026-10-11之后的场景边界、真实液深契约、透明V22-V33、普通V35-V41、部分失效、V11波面、照明和视觉V4-V9记录。')
    para(story,'旧PDF与CSV/Markdown共11个文件已备份为 previous-progress-records.zip，并用逐文件SHA256验证。它们的旧目标和旧资格口径不作为现版的测量承诺。全部细节保留服务器及仓库版本化源记录，本开发简史只摘录重要进展，未复述每次参数尝试。')
    para(story,'当前精度与分层表以配套《正式链路与全场景测试报告》及仓库 reports/evidence 为准。GitHub发布状态、最终提交号及两份PDF哈希以仓库发布清单和交付消息为准；不在PDF内写入尚未生成的提交号。')
    document(OUT/'2026-10-16_development_history.pdf',story,portrait=True,title='RGB-D液深系统 开发过程与重要进展')

def main():
    setup_fonts()
    for name,size,leading,font,color in [('body',10,15,'CN','#25364B'),('cell',8.5,12,'CN','#25364B'),
      ('th',8.5,12,'CNB','#FFFFFF'),('h1',18,25,'CNB','#17314D'),('h2',11,17,'CNB','#17314D')]:
        STYLES[name]=ParagraphStyle(name,fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(color),wordWrap='CJK',spaceAfter=0,alignment=TA_LEFT)
    rows,sources=load_rows();groups=summarize(rows)
    expected={'ordinary_diffuse':396,'partial_depth_loss':2346,'homogeneous_transparent':2672,
      'homogeneous_translucent':5344,'floating_objects':360,'low_light':1584,'glare':1584,'uneven_surface':1584}
    assert {s:len(select(rows,s)) for s in SCENES}==expected
    (OUT/'scene_metrics_20261016.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with gzip.open(OUT/'current_predictions_20261016.jsonl.gz','wt',encoding='utf-8') as stream:
        for r in rows:stream.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    provenance={'release':'formal-20261016','source_files':{str(p.relative_to(INPUT)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sources},
      'historical_source_documents':{str(p.relative_to(INPUT)):sha(p) for p in sorted((INPUT/'docs').glob('*.md'))},
      'cohorts':expected,'scope':'existing frozen records recounted except4752 newly captured meaningful special-scene frames; different inclusion rules remain explicitly separated',
      'absolute_bins_mm':['[0,5)','[5,20)','[20,50)','[50,100)','[100,infinity)'],
      'relative_bins_pct':['[0,1)','[1,2)','[2,5)','[5,10)','[10,infinity)'],
      'transparent_tof_supported':False,'transparent_dedicated_structured_light_qualified':False,
      'transparent_sensor':'stereo_IR_image_proxy','no_algorithm_changes_during_publication':True}
    (OUT/'source_manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    make_system(rows,groups);make_history()
    print(json.dumps({'frames':len(rows),'strata':len(groups),'files':[(p.name,p.stat().st_size) for p in OUT.iterdir()]},ensure_ascii=False))


if __name__=='__main__':main()
