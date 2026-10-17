"""Three Chinese PDF deliverables with covers, linked contents and audited tables."""
import argparse,hashlib,html,json,os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,LongTable,TableStyle,KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing,Rect,Line,Polygon,String
from reportlab.lib.utils import simpleSplit

P=argparse.ArgumentParser();P.add_argument('--input',type=Path,default=Path(__file__).parent)
P.add_argument('--out',type=Path,default=Path(__file__).parent/'output/pdf');A=P.parse_args()
OUT=A.out;OUT.mkdir(parents=True,exist_ok=True)
M=json.loads((A.input/'metrics.json').read_text(encoding='utf-8'))
AUDIT=json.loads((A.input/'evaluation-audit.json').read_text(encoding='utf-8'))
FAIL=json.loads((A.input/'far-failure-cases.json').read_text(encoding='utf-8'))
pdfmetrics.registerFont(TTFont('CN',os.environ.get('PDF_CN_FONT','C:/Windows/Fonts/msyh.ttc'),subfontIndex=0))
pdfmetrics.registerFont(TTFont('CNB',os.environ.get('PDF_CN_BOLD_FONT','C:/Windows/Fonts/msyhbd.ttc'),subfontIndex=0))
SCENES=('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent','partial_depth_loss','low_light','glare','uneven_surface','floating_objects')
NAMES=dict(zip(SCENES,('普通漫反射','均匀透明','均匀半透明','部分深度失效','低照','眩光','不平整液面','漂浮物')))
NAMES.update(tof='ToF',structured_light='结构光',all='合并',mild='低',moderate='中',severe='强',low_extinction='低消光',medium_extinction='中消光',clear='透明')
ST={n:ParagraphStyle(n,fontName=f,fontSize=s,leading=l,textColor=colors.HexColor(c),wordWrap='CJK',spaceAfter=0)
    for n,f,s,l,c in [('body','CN',10.1,16,'#243449'),('small','CN',8.3,12,'#53667D'),('cell','CN',8.8,13,'#243449'),
      ('th','CNB',8.8,13,'#FFFFFF'),('H1','CNB',18,25,'#143550'),('H2','CNB',12,19,'#143550'),('cover','CNB',28,42,'#143550')]}
ST['TOCTitle']=ParagraphStyle('TOCTitle',parent=ST['H1'])


def text(x):return str(x).replace('–','-').replace('—','-').replace('‑','-').replace('−','-')
def para(x,style='body'):return Paragraph(html.escape(text(x)).replace('\n','<br/>'),ST[style])
def add(story,x,style='body'):
    value=para(x,style);value.spaceAfter=8;story.append(value)
def chapter(story,x):
    if story:story.append(PageBreak())
    add(story,x,'H1')
def heading(story,x):add(story,x,'H2')
def fmt(x,d=2):return '-' if x is None else f'{x:.{d}f}'
def metric(scene,sensor='all',level='all',subset='all',peak=False):return M[f'{scene}|{sensor}|{level}|{subset}']['highest' if peak else 'liquid']
def bins(v):return ' / '.join(fmt(x,1) for x in v) if v else '-'


def table(headers,rows,widths):
    t=LongTable([[para(h,'th') for h in headers]]+[[para(c,'cell') for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#203D59')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F9')]),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D7E1EB')),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    return t


class Report(SimpleDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name in ('H1','H2'):
            title=f.getPlainText();key='heading_'+hashlib.sha1(title.encode()).hexdigest()
            self.canv.bookmarkPage(key)
            if f.style.name=='H1':
                self.canv.addOutlineEntry(title,key,0,False)
                self.notify('TOCEntry',(0,title,self.page,key))


def footer(c,d):
    if d.page==1:return
    w,h=d.pagesize;c.saveState();c.setStrokeColor(colors.HexColor('#D5E0EB'));c.line(36,30,w-36,30)
    c.setFillColor(colors.HexColor('#61738B'));c.setFont('CN',8)
    c.drawString(36,17,'RGB-D 液深系统 | 双相机 V2 | 2026-10-17 | 仿真研究结果，非工业计量认证')
    c.drawRightString(w-36,17,f'{d.page}');c.restoreState()


def cover(story,title,subtitle,scope):
    story.extend([Spacer(1,54),para('RGB-D 液深测量系统','H2'),Spacer(1,16),para(title,'cover'),Spacer(1,16),para(subtitle,'H2'),Spacer(1,30)])
    add(story,'版本：dual_camera_v2_20261017    日期：2026年10月17日')
    add(story,scope)
    add(story,'相机基准：ToF / 结构光。当前正式入口不接受主动双目，也不复用旧左右目图像代理的透明成绩。')
    add(story,'适配分为“输入与计算路线已接通”与“测量精度已验收”。本轮仅完成仿真与软件接口验证，不能替代真实设备、容器与现场验收。','small')
    story.append(PageBreak());add(story,'目录','TOCTitle')
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('TOC',fontName='CN',fontSize=11,leading=23,leftIndent=0,firstLineIndent=0,rightIndent=20,textColor=colors.HexColor('#203D59'))]
    toc.dotsMinLevel=0;story.append(toc)


def save(story,name,title):
    doc=Report(str(OUT/name),pagesize=landscape(A4),leftMargin=40,rightMargin=40,topMargin=36,bottomMargin=45,title=title,author='Liquid Surface Detector Project')
    doc.multiBuild(story,onFirstPage=footer,onLaterPages=footer)


def metricrow(sensor,label,m):
    return [NAMES[sensor],label,f"{m['n']} / {m['installations']}",fmt(m['mae_mm']),fmt(m['p95_mm']),fmt(m['max_mm']),fmt(m['absrel_pct']),fmt(m['pass_pct']),fmt(m['numeric_coverage_pct'])]


def performance():
    s=[];cover(s,'全场景性能测试报告','两类相机、程度小类、补充采样与误差分布',
        '按场景汇总，并按ToF /结构光及指定程度划分；本报告不按实际液深分层。包含原链路保留结果和本轮独立补采结果，来源分别标记。')
    chapter(s,'01  各场景总数据与性能')
    rows=[]
    for scene in SCENES:
        m=metric(scene);rows.append([NAMES[scene],metric(scene,'tof')['n'],metric(scene,'structured_light')['n'],m['n'],fmt(m['mae_mm']),fmt(m['p95_mm']),fmt(m['absrel_pct']),fmt(m['pass_pct']),fmt(m['numeric_coverage_pct'])])
    s.append(table(['场景','ToF帧','结构光帧','合计帧','MAE\nmm','P95\nmm','AbsRel\n%','通过率\n%','数值覆盖\n%'],rows,[115,60,70,65,65,65,70,70,80]))
    add(s,'液深均指相对容器内底面的实际液深H。普通、透明、半透明、部分失效、低照、眩光、漂浮物液深采用max(20mm,2%H)；不平整液面均值采用max(50mm,5%H)。最高高度采用max(50mm,5%真实最高高度)。总表只统计液深。')
    add(s,'“数值覆盖100%”只表示纳入输入都有有限数值，不表示误差达标。本轮对远距浅液位定向扩样，且去除了旧主动双目；合并值不能与上一版不同构成的数据直接比较来声称提升或退化。','small')
    chapter(s,'02  采样、有效输入与统计口径')
    a=AUDIT
    add(s,f"共{a['total_frames']:,}帧：保留原确定版ToF/结构光{a['old_retained_frames']:,}帧，新增远浅几何测试{a['new_far_frames']:,}帧，新增原始相位测试{a['new_phase_frames']:,}帧。旧透明/半透明左右目图像代理数据不混入本轮。")
    add(s,f"报告小类定义为场景×相机×程度。最低小类为{a['minimum_cell']}帧，全部达到不少于100帧；八个场景每种相机的远浅子集也均不少于100帧。新增普通类远浅为25组安装，每条件4次独立噪声采样，RGB/几何不复制成新安装；透明类各有效消光类别为50组安装，每相机每安装4帧测试。")
    add(s,f"新增普通类尝试{a['far_input_attempted_pairs']}组成对输入，纳入{a['far_selected_pairs']}组、排除{a['far_rejected_pairs']}组，配对合格采样率{100*a['far_selected_pairs']/a['far_input_attempted_pairs']:.2f}%。只按输入空间支撑及离线物理来源审核排除，不使用预测或液深误差筛选。受控删点派生帧另计入部分失效。")
    add(s,'新普通类输入规则：容器内部至少64个返回点、至少6/16空间格、双轴跨度至少25%；另确认至少64个直接液面来源点（而非数值接近真值的点）。该来源标签只供仿真采集审核，不进入推理。历史样本沿用各自原纳入协议，不能伪称全部重采或都经过本次来源审核。')
    add(s,f"透明类300次五帧先验标定全部完成；先验固定为1组空槽+4组已知液深，标注含0.3mm+0.1%H噪声。当前帧真值、材料参数、来源标签都不传给逆解。保留全部有效观测的输出；{a['phase_search_boundary_hits']}帧触及扩展搜索边界，未按残差剔除，仍计入全部误差。")
    add(s,'透明仿真使用常数等效折射率1.28-1.50，消光系数分别为0、0.012-0.05、0.055-0.12/m；并非850/940nm水或任意工业液体的实测光谱档案。可见光透明不保证近红外底部回波可见，深液层尤其必须实测。ToF群折射率/结构光相折射率在本轮作常数有效近似，未做光谱色散与真实功率预算认证。','small')
    add(s,'透明相位试验中的B、内参、基线与射线按给定几何标定值使用，尚未额外叠加它们的系统误差；已知H标注和当前观测有噪声。普通类则包含空槽/操作员几何误差，因此不能将本报告读成两条完整现场链路的严格同条件设备排名。透明部署需从独立空槽几何标定提供这些参数。','small')
    add(s,'N为帧数，安装数为聚类统计单元；同安装的不同噪声/程度不当作独立容器。MAE为平均绝对误差，P95为95%分位，AbsRel仅对H>0的数值输出计算。通过率分母包含未输出帧；空槽进入绝对误差和通过率但不进入相对误差。聚类bootstrap置信区间见证据JSON，不能据此宣称工业计量权威认证。','small')
    notes={
      'ordinary_diffuse':'普通链路未更改液深算法，继续使用空槽建模和冻结对称回波候选。ToF结果较稳定；结构光新远浅样本放大了已有视差噪声与候选身份问题。',
      'homogeneous_transparent':'本轮是真正按传感器观测类型分别输入：ToF为三调制频率四相相关，结构光为两周期四相投影条纹。并非把原双目代理改名。结构光远浅的相位响应很弱，接通路线不等于所有工况精度达标。',
      'homogeneous_translucent':'范围限定为均匀、底部信号仍可见的轻/中等有效消光液体，不含强散射不透明、多层或液体性质变化。结构光已接通五先验逆解，但散射、信号衰减与相位模型失配仍会造成长尾，须按本表实测结果判断，不能宣称已全部验收。',
      'partial_depth_loss':'程度指真实可见液面区域中数值深度为空的比例，不是全图面积，也不是错误回波比例。测试范围仍为0%-50%；50%归入最后一档。仅损失面积不等于剩余点可靠、分布均匀。',
      'low_light':'低/中/强为预先固定的照明、曝光及深度退化参数，不是经设备标定的lux阈值。本路线不自动调用暗光增强或视觉模型改写液深。',
      'glare':'低/中/强为预先固定的高光、空洞和回波扰动参数，不是可直接移植的饱和面积阈值。极端缺少有效证据的采集单列，不通过把测错帧排除来提高通过率。',
      'uneven_surface':'分别统计物理面积平均液深与最高包络；两者均使用50mm/5%容差。代表平面不是全场波形重建，波面局部最高的可见性、采样密度仍是最高高度误差来源。',
      'floating_objects':'视觉仅可用于漂浮物最高高度，液深保持同一冻结几何。当前只用ToF/结构光，因此V8 selected按既定策略走V5最高高度分支，不调用主动双目的V8学习排序。最高目标是底面相对的液面及漂浮物物理包络。'}
    for i,scene in enumerate(SCENES,3):
        chapter(s,f'{i:02d}  {NAMES[scene]}')
        rows=[]
        levels=['mild','moderate','severe'] if scene in ('low_light','glare') else ['0-10%','10-20%','20-30%','30-40%','40-50%'] if scene=='partial_depth_loss' else ['all']
        for sensor in ('tof','structured_light'):
            for level in levels:rows.append(metricrow(sensor,NAMES.get(level,level)+'液深' if level!='all' else '液深',metric(scene,sensor,level)))
            if scene in ('uneven_surface','floating_objects'):rows.append(metricrow(sensor,'最高高度',metric(scene,sensor,peak=True)))
        s.append(table(['相机','程度 / 输出','帧 / 安装','MAE\nmm','P95\nmm','最大\nmm','AbsRel\n%','通过率\n%','数值覆盖\n%'],rows,[75,110,90,65,65,70,65,65,70]))
        add(s,notes[scene])
        if scene=='homogeneous_translucent':
            add(s,'低消光/中消光的单独样本数：'+ '；'.join(f'{NAMES[sensor]} {NAMES[level]} {metric(scene,sensor,level)["n"]}帧' for sensor in ('tof','structured_light') for level in ('low_extinction','medium_extinction'))+'。表内先合并消光类别，细分类指标保留JSON。','small')
        for sensor in ('tof','structured_light'):
            m=metric(scene,sensor);ci=m['cluster_bootstrap_95ci']
            add(s,f"{NAMES[sensor]}液深：MAE的安装聚类95%区间 [{fmt(ci['mae_mm'][0])}, {fmt(ci['mae_mm'][1])}] mm；通过率区间 [{fmt(ci['pass_pct'][0])}, {fmt(ci['pass_pct'][1])}]%。未输出{m['no_output']}帧。",'small')
    chapter(s,'11  绝对误差区间占比')
    rows=[]
    for scene in SCENES:
        for sensor in ('tof','structured_light'):
            m=metric(scene,sensor);rows.append([NAMES[scene],NAMES[sensor],m['n'],*[fmt(v,1) for v in m['absolute_bins_pct']],m['no_output']])
    s.append(table(['场景','相机','帧数','<5mm\n%','5-20mm\n%','20-50mm\n%','50-100mm\n%','≥100mm\n%','未输出\n帧'],rows,[115,75,60,75,75,75,75,75,70]))
    add(s,'误差区间左闭右开，最后一档包含100mm；区间百分比分母为该行全部测试帧。最高高度的独立误差区间保留证据JSON。','small')
    chapter(s,'12  相对误差区间占比与相机建议')
    rows=[]
    for scene in SCENES:
        for sensor in ('tof','structured_light'):
            m=metric(scene,sensor);rows.append([NAMES[scene],NAMES[sensor],m['relative_n'],*[fmt(v,1) for v in m['relative_bins_pct']]])
    s.append(table(['场景','相机','非零H帧','<1%\n%','1-2%\n%','2-5%\n%','5-10%\n%','≥10%\n%'],rows,[115,75,75,85,85,85,85,85]))
    add(s,'仅基于本轮参数化仿真，普通类远浅优先验证ToF；35mm短基线结构光不宜据此承诺10m端浅液位精度。透明/半透明需原始相位或可控投影数据，普通深度SDK不足以自动启用该路线；ToF本轮较稳，结构光仍需特别关注弱响应与散射失配。不能把两种不同模拟输入的差异外推为所有真实设备的排名。','small')
    chapter(s,'13  延时、复现与限制')
    rows=[]
    for scene in SCENES:
        for sensor in ('tof','structured_light'):
            t=metric(scene,sensor)['latency_ms']
            rows.append([NAMES[scene],NAMES[sensor],t['timed_frames'] if t else 0,fmt(t['median']) if t else '-',fmt(t['p95']) if t else '-',fmt(t['max']) if t else '-'])
    s.append(table(['场景','相机','有计时帧','中位 ms','P95 ms','最大 ms'],rows,[180,95,100,105,105,105]))
    add(s,f"相位先验初始化中位耗时{AUDIT['phase_calibration_median_ms']:.1f}ms，为安装级一次性开销。表中为已有/本轮算法计时，不含完整硬件曝光、ToF内部调制、结构光投影序列、传输、落盘及界面刷新，因此不宣称端到端已满足0.5s。进程并行、冷启动和旧计时条件可能不同。",'small')
    chapter(s,'14  复现、软件校验与验收限制')
    add(s,'本次完整软件回归754项通过，14条依赖警告；八场景×两相机的16个实际CLI回放均与测试预测一致。独立按发布预测重算所有统计单元，核验帧数、覆盖率、通过率及误差分布。回归测试不是实际设备精度验收。')
    add(s,'复现入口：scripts/generate_dual_far_v2.py、evaluate_dual_far_v2.py、evaluate_phase_refraction_v2.py、summarize_dual_camera_v2.py。源种子、相机参数、输入排除、预测及哈希随服务器归档；正式入口为scripts/measure_dual_camera.py。原正式版保持可回放，不覆盖历史事实。','small')
    add(s,'相位初测发现散射相位定义错误后，原初测转为开发证据并排除；修正后更换独立种子生成本报告相位测试。最终测试开始后源哈希锁定，未按最终误差挑样或再调参。','small')
    save(s,'2026-10-17_dual_camera_performance.pdf','RGB-D液深系统 双相机全场景性能测试报告')


def flow(nodes,notes,title='共同几何与两相机输入',color='#DCEAF5'):
    width=748;height=60*len(nodes)+25;d=Drawing(width,height)
    for i,(node,note) in enumerate(zip(nodes,notes)):
        y=height-55-i*60;d.add(Rect(4,y,265,40,rx=5,ry=5,fillColor=colors.HexColor(color),strokeColor=colors.HexColor('#9BB3C9')))
        for j,line in enumerate(simpleSplit(text(node),'CNB',10,245)):
            d.add(String(16,y+24-j*13,line,fontName='CNB',fontSize=10,fillColor=colors.HexColor('#173C59')))
        for j,line in enumerate(simpleSplit(text(note),'CN',9.2,446)):
            d.add(String(286,y+27-j*14,line,fontName='CN',fontSize=9.2,fillColor=colors.HexColor('#334B63')))
        if i<len(nodes)-1:
            d.add(Line(136,y-1,136,y-18,strokeColor=colors.HexColor('#4E7494')))
            d.add(Polygon([132,y-14,140,y-14,136,y-20],fillColor=colors.HexColor('#4E7494'),strokeColor=None))
    return d


def architecture():
    s=[];cover(s,'正式计算链路报告','按场景说明输入、计算模块与输出边界',
        '基于修改后的DualCameraSceneMeter接口。旧DeterminedSceneMeter保持冻结；正式双相机入口仅接受ToF和结构光。结构示意图为实际模块逻辑，不是产品硬件连接验收图。')
    chapter(s,'01  总体结构与相机能力边界')
    s.append(flow(['操作员选择场景 + ToF / 结构光','普通类：空槽RGB-D + 小规模米制输入','透明类：1空槽 + 4已知液深相位组','分支测量 + 诊断输出'],[
        'DualCameraSceneMeter拒绝active_stereo及旧stereo_IR_image_proxy。场景不自动识别。',
        '普通、失效、低照、眩光、不平整、漂浮物沿用冻结空槽几何。',
        'iToF三频四相；结构光两周期四相条纹。五组先验随液体/机位变化重新建立。',
        '统一输出相对内底面的米制液深；不平整和漂浮物另给最高高度，可信度仅诊断。']))
    add(s,'硬件要求：普通类只需已对齐米制深度、内参及空槽标定；透明类新增原始调制相关图/可控投影序列。深度图不包含足够的原始相位信息，不能把任意只导出RGB-D的ToF或封闭投影SDK称为透明路线已可部署。此版支持AMCW iToF，不包括任意脉冲dToF。','small')
    add(s,'光谱边界：透明度、消光与可观测性取决于发射波长。当前低消光仿真不能代表近红外波段9m水层；ToF有效群折射率与Snell相折射率作常数近似。需实测设备波长下的吸收/散射、调制对比及功率预算，才能判断真实量程。','small')
    add(s,'透明入口仍需独立空槽几何提供B、内参/射线、投影基线与ROI；五相位先验并不替代所有几何标定。本轮相位实验给定这些标定量，未包含其系统偏差；这部分真实部署误差仍需新增验证。','small')
    chapter(s,'02  标定、观测与数据契约')
    s.append(table(['部分','实际输入与校验','不允许的捷径'],[
      ['空槽几何','固定机位、内参、空槽米制深度、容器轮廓/尺寸、重力与独立米制锚点；失效标定先修正。','不能用当前帧真液深或液面标签修正估计。'],
      ['相位五先验','五组同机位观测；已知H=0以及四个递增的非零H，均在容器容量内。','不把测试H、测试折射率、真值来源掩码传给逆解。'],
      ['ToF观测','数组[F=3,4,H,W]；20/30/50MHz，四相解调；有效相位与幅度、相机射线。','不能把ToF光程当作结构光视差或真实表面距离。'],
      ['结构光观测','数组[F=2,4,H,W]；8/13像素条纹周期；标定相机/投影器几何，35mm基线、fx470.94px的研究档案。','不是被动双目，也不把旧左右目底纹代理分数并入本轮。'],
      ['应用范围','固定容器与机位；普通0%-50%丢失；均匀透明/轻半透明；相机到底面测试≤10m。','不承诺多层、强散射、连续移动、95%-100%失效或实际商业相机10m认证。']], [120,340,280]))
    for i,scene in enumerate(SCENES,3):
        chapter(s,f'{i:02d}  {NAMES[scene]}计算链路')
        if scene.startswith('homogeneous_'):
            s.append(flow(['原始4相相关 / 投影条纹观测','空槽相位比值 + 空槽ROI','五帧先验拟合等效物理参数','当前帧相位匹配与一维液深逆解','液深 + 残差 / 灵敏度 / 外推提示'],[
              'ToF decode三频复相关；结构光decode两周期复条纹。每种相机单独标定，不混用观测。',
              'I0-I2 + i(I3-I1)，再除空槽参考的单位复相位。只依赖空槽/操作员ROI。',
              '拟合n、表面/底面混合比、消光、散射与残余偏置；这些是等效参数，不是材料折射率的计量认证。',
              'ToF用介质光程与混合回波；结构光用Snell相机-底部-投影器路径。多候选全局粗搜后局部细化。',
              '不按残差强制拒绝；允许负估计和容量外估计。搜索边界另报，不能把边界输出当确定解。'],color='#DAEDE7'))
            if scene.endswith('transparent'):
                add(s,'均匀透明假设：液体性质在标定与后续帧之间稳定，底部能提供调制/投影信号。ToF的底部有效光程随nH增加，而表面路径随H减少；结构光的表观收缩与此不同。五组先验提供安装与液体的响应约束，不是只从稀少真实表面点拟平面。','small')
            else:
                add(s,'半透明增加消光和体散射。前向仿真采用24层散射积分、模糊与空间噪声；ToF逆模型用4层近似，结构光用中间散射相位近似，保留真实的模型失配。强散射或底部调制消失不属于此路线；若变为不透明漫反射，应重新按普通场景安装标定，而非跨场景硬套。','small')
        else:
            nodes=['空槽图像与独立米制几何建模','当前ToF / 结构光米制深度','容器范围内回波候选与对称评分','稳健液面代表高度 → 内底面液深']
            notes=['empty_geometry_v41 / empty_scene_routes_v44，保留底面、轮廓、重力及相机变换。',
                '两类深度参数档案分别产生距离噪声/量化/空洞；当前研究不是全谱真实商业设备数字孪生。',
                '冻结symmetric_visual_meter / frozen_symmetric_geometry，不仅允许底面峰覆盖原估计，候选同等比较。',
                '沿用原确定版几何，不因输出负数或超容量而强制归零、裁剪。视觉不得覆盖液深。']
            if scene=='partial_depth_loss':notes[1]='丢失只限定可见液面数值空洞0%-50%，不是全图；剩余点仍可能有错层或量化偏差。'
            if scene in ('low_light','glare'):notes[2]='光照程度影响可用回波；不自动调用RGB增强或视觉分割改写液深。输入不足的采集单独记录。'
            if scene=='uneven_surface':
                nodes.append('V11几何包络 → 最高高度');notes.append('平均液深与最高高度分别输出/评价；不声称重建所有波峰、波谷与完整动态波形。')
            if scene=='floating_objects':
                nodes.append('按需RGB视觉 + V5选定分支 → 最高高度');notes.append('floating_peak_only_v4约束输出范围。ToF/结构光的V8 selected沿用V5；旧主动双目V8排序不启用。')
            s.append(flow(nodes,notes))
            if scene=='ordinary_diffuse':add(s,'处理边界为普通不透明漫反射液体。历史多帧、量化似然与公共偏置试验不是本版默认算法；本轮没有为补采结果额外调普通液深参数。','small')
            if scene=='floating_objects':add(s,'视觉开启需明确floating_objects场景。其他七类不使用此视觉模块；即使请求开关，也不能改变它们的液深。最高对象证据不足、遮挡或整体测距偏差仍可能导致误差，不以视觉置信高代替米制可信度。','small')
    chapter(s,'11  软件入口与复现文件')
    s.append(table(['层级','文件','职责'],[
      ['正式双相机接口','src/liquid_depth/dual_camera_scene_meter_v2.py','严格ToF/结构光及场景/标定类型检查，分发几何或相位。'],
      ['相位解码/标定/逆解','src/liquid_depth/phase_refraction_v2.py','只接受原始相位及五先验；输出残差、灵敏度、外推和边界提示。'],
      ['命令行','scripts/measure_dual_camera.py','empty_geometry或five_phase_priors初始化；读取npy/npz/rgb，便携小模型哈希校验。'],
      ['普通类补采','generate_dual_far_v2.py / evaluate_dual_far_v2.py','Blender RGB/底面/浮物真几何，双相机参数噪声，固定输入证据审核。'],
      ['透明类观测仿真','phase_capture_simulator_v2.py / evaluate_phase_refraction_v2.py','四相观测、独立投影求解、散射层积分、量化/模糊/相关噪声及独立种子。'],
      ['统计与报告','summarize_dual_camera_v2.py / build_dual_camera_reports.py','不按实际液深分层；所有场景和指定程度小类≥100，记录误差区间和来源。']], [135,320,285]))
    add(s,'常驻部署应只初始化一次测量对象，然后连续调用measure；单次CLI会重复加载先验，不应把该初始化开销当作每帧持续测量速度。旧工业GUI/USB驱动并未在本轮完成全部新相位接口硬件联调。','small')
    chapter(s,'12  验证结论与应用限制')
    add(s,'所有八类场景均已有ToF/结构光软件入口及本轮统计，不使用其他相机作为成绩基准。普通类保持冻结几何；透明/半透明不再借用双目代理。')
    add(s,'754项软件测试通过，16个场景/相机组合实际CLI回放一致；源哈希在最终独立测试开始前冻结。界面/真实USB硬件联调不在这次软件回放验收范围内。')
    add(s,'但“适配”不能等同“已达到工业精度”。尤其结构光远浅和半透明仍可能明显超标，依性能报告判断。原始相位/投影能力、设备噪声参数、固定机位与五组标定预算是部署先决条件。')
    add(s,'参考原理：[1] Tanaka等，CVPR2016，Recovering Transparent Shape from Time-of-Flight Distortion；支持介质光程与折射路径建模，但该论文部分假设忽略镜面/多路径，不能直接套用为本项目验收。[2] Song和Shim，ACCV2018，Depth Reconstruction of Translucent Objects from a Single ToF Camera using Deep Residual Networks；说明半透明测距还涉及复杂散射失真。本版未宣称复现上述论文完整系统。')
    add(s,'来源链接：https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Tanaka_Recovering_Transparent_Shape_CVPR_2016_paper.pdf ；https://arxiv.org/abs/1809.10917','small')
    add(s,'光谱边界参考：[3] Hale和Querry，Applied Optics12(3)，555-563，1973，水的复折射率与波长相关吸收资料。当前未将其850/940nm数据标定进仿真，不能把消光=0的结果当成真实水层指标。来源：https://ntrs.nasa.gov/citations/19730040258','small')
    save(s,'2026-10-17_dual_camera_formal_pipeline.pdf','RGB-D液深系统 双相机正式计算链路报告')


def far_report():
    s=[];cover(s,'远距浅液位不足报告','按两类相机比较性能、输入证据与问题原因',
        '远浅定义：相机到底面B≥6m且实际液深0<H≤1m。该定义只用于离线统计，不传给推理。本报告不进行实际液深区间分层。')
    chapter(s,'01  全场景远浅性能总表')
    rows=[]
    for scene in SCENES:
        for sensor in ('tof','structured_light'):rows.append([NAMES[scene],*metricrow(sensor,'液深',metric(scene,sensor,subset='far_shallow'))])
    # Remove duplicate output label to keep one compact all-scene table.
    rows=[[r[0],r[1],*r[3:]] for r in rows]
    s.append(table(['场景','相机','帧 / 安装','MAE\nmm','P95\nmm','最大\nmm','AbsRel\n%','通过率\n%','数值覆盖\n%'],rows,[105,75,95,65,65,70,65,65,70]))
    add(s,'各行均不少于100帧。透明类包含新一般安装中自然落入远浅的帧；普通类保留既有远浅并补25组新安装。不同来源不等同真实现场均匀抽样。通过率的容差与性能报告相同。','small')
    chapter(s,'02  共性与差异：为何不能只用一个原因解释')
    s.append(table(['路线','共同问题','本轮证据与解释边界'],[
      ['普通类几何 / 结构光','长距离视差小，噪声与量化把底面和液面高度分布混在一起；容器筛选、量程缺失还可带来选择偏差。','新最差案例并非没有点，而是候选只剩一个偏高层。点数多不等于有独立、无偏的层身份信息。'],
      ['普通类几何 / ToF','传感器随机噪声相对小，空槽米制锚点/法向与残余共享偏置更可能成为毫米级下限。','本轮模拟表现较稳；不能据此证明所有工业ToF都优于结构光，多路径实景仍需验证。'],
      ['透明类 / 结构光','观测是折射后的投影相位，不是直接液面深度；B增大时浅H的相位变化很小。','5帧先验可学习响应，但未增加新帧的光子/相位信噪比，不能凭先验消除所有噪声或模型失配。'],
      ['透明类 / ToF','表面、液体内散射与底部光程在复相位中相加，参数混淆、共模漂移与模型失配仍存在。','双相位通道和标定改善可辨性，但拟合的等效折射率并不是真实材料参数的独立验证。']], [145,300,295]))
    add(s,'因此先共表比较，再逐场景解释。不能因为某点接近真值就断言它是液面，也不能只看误差大就称为底部回波；下面尽量使用在线可见的候选数、支撑、分数和相位残差作为证据，并标明未验证的原因。')
    causes={
      'ordinary_diffuse':('结构光的核心仍是弱视差与有偏候选。z≈10m、fx≈471px、基线35mm时视差仅约1.65px；0.26px视差噪声的一阶距离标准差z²σd/(fxb)约1.6m，远大于几厘米或几十厘米液深。大量真实来源点仍可能落在错误高度分布里。','ToF该仿真的随机深度噪声为1.5mm+0.7mm/m，远端约8mm；空槽锚点误差与共模偏置不会靠增加当前点数无限降低。后续优先做同输入量化/截断似然、空槽共偏置差分和静态多帧配对消融。'),
      'partial_depth_loss':('在普通问题上再叠加空间分布不均。丢失50%并不必然让均值翻倍，但会减少独立区域支持，使候选峰的分数、法向与空间布局更不稳定；错误回波比例不包含在空洞比例中。','本轮只放宽数值输出，不靠拒绝把坏帧删掉。应分别核对原始点→容器内点→拟合点的数量和格点分布，再比较估计偏置；不能把“保留50%点”当成50%可靠证据。'),
      'low_light':('本轮低照为RGB与深度联合退化代理。ToF与结构光都有主动照明，因此暗RGB本身不是深度必然失效的充分原因；这里结构光仍主要暴露远浅视差噪声和候选混叠。','深度空洞/有效区域变化会影响候选，但默认没有暗光增强网络。先对相同几何的三档程度作配对，并把曝光/SNR/空间支撑与层身份分开诊断；不直接用提亮RGB替代深度证据。'),
      'glare':('高光造成局部饱和、空洞和错误回波，使独立空间支持更不均匀。已通过输入证据审核的帧仍可能有偏，尤其35mm结构光的远端分布本已宽。','本轮输入审核不看估计误差，极端无支撑采集单列。普通量化/共模问题应先共用修复，再单独分析高光边界和非饱和区域；没有证据时不能强行把高置信当正确。'),
      'uneven_surface':('非平面液面还引入“代表平面”和物理面积均值不完全等价。区域缺失、遮挡或置信采样偏好会让采样均值偏向某一波相；最高波峰需要足够的局部空间分辨率。','本轮平均与最高都用50mm/5%容差，但不是将所有波动误差视为零。后续只针对远浅共性修复时必须保护面积均值与峰高，不以强平滑抹掉真实峰值换取均值好看。'),
      'floating_objects':('液深不依赖视觉，因此远浅主要复用普通几何的噪声和层身份问题。最高高度额外受最高实例漏选、小物体像素少、遮挡与该区域共同测距偏差影响。','ToF/结构光的V8 selected实际用V5最高分支，不是主动双目V8排序。视觉概率可以帮助找区域，不能制造真实测距样本；物理插值加密不等于新增独立信息。'),
      'homogeneous_transparent':('远浅结构光的折射相位响应随距离减弱；几帧标定能估计等效折射响应，却无法消除新帧共模相位漂移。五先验覆盖液深范围、外推和数值灵敏度都需观察。','ToF的光程变化与结构光符号、几何依赖不同。表面/底部混合及等效参数误差可放大成液深偏差；不能把旧左右目底纹代理的99%成绩迁移到当前两种新观测。'),
      'homogeneous_translucent':('在透明的弱响应上增加衰减、相位模糊和散射混合。结构光尤其可能在先验残差已较大时仍给出数值，而放宽门控后这些失败应诚实计入。','当前“残差/灵敏度”只反映逆模型内部局部一致性；若标定模型本身失配，该值可能低估真实误差。应把先验拟合残差、参数到界、响应条件数与重复采集一起审查，不能只看最终当前残差。')}
    for i,scene in enumerate(SCENES,3):
        chapter(s,f'{i:02d}  {NAMES[scene]}：证据与原因')
        add(s,causes[scene][0]);add(s,causes[scene][1])
        for sensor in ('tof','structured_light'):
            row=FAIL[scene+'|'+sensor][0];err=1000*abs(row['liquid_depth_m']-row['truth']['liquid_height_m']);d=row.get('diagnostics',{})
            heading(s,f'{NAMES[sensor]}  远浅最大误差案例')
            add(s,f"B={row['camera_to_bottom_m']:.3f}m，实际H={row['truth']['liquid_height_m']:.4f}m，输出={row['liquid_depth_m']:.4f}m，绝对误差={err:.2f}mm。此帧保留在测试分母中，不因误差大被排除。")
            if scene.startswith('homogeneous_'):
                add(s,f"有效相位样本{d.get('support','-')}；先验残差={fmt(d.get('calibration_residual'),5)}；当前残差={fmt(d.get('model_residual'),5)}；局部灵敏度={fmt(d.get('local_sensitivity_per_m'),5)}/m；残差/灵敏度={fmt(1000*d.get('residual_to_sensitivity_m',0))}mm；边界触及={d.get('constraint_boundary_reached',False)}。这些是可见诊断，不是真实误差的保证区间。",'small')
            else:
                c=d.get('candidates',[])
                if c:
                    best=max(c,key=lambda r:r.get('score',0))
                    add(s,f"在线候选{len(c)}个，最高分候选高度={fmt(best.get('height_m'),4)}m，返回支撑={best.get('support_returns','-')}，空间格={best.get('tiles','-')}，候选解释={best.get('interpretation','-')}。这些指标可说明并非单纯无点，但不能单独确认候选属于真实液面。",'small')
                else:add(s,'此条旧精简预测未保留完整候选诊断，不能仅根据真值偏差确认误选层。须回到原始深度/空槽及完整日志做同输入消融；本报告不补造原因计数。','small')
    chapter(s,'11  最高高度与后续验证安排')
    rows=[]
    for scene in ('uneven_surface','floating_objects'):
        for sensor in ('tof','structured_light'):
            m=metric(scene,sensor,subset='far_shallow',peak=True)
            rows.append([NAMES[scene],NAMES[sensor],m['n'],fmt(m['mae_mm']),fmt(m['p95_mm']),fmt(m['max_mm']),fmt(m['pass_pct'])])
    s.append(table(['场景','相机','帧','最高MAE mm','最高P95 mm','最高最大 mm','最高通过 %'],rows,[130,85,65,110,110,110,110]))
    add(s,'优先级：①核查35mm结构光档案在远端是否满足实际硬件可观测条件；②按安装独立验证空槽公共偏置与量化/截断模型；③静态2-4帧融合分离独立噪声与持续相关偏差；④透明类单独优化相位响应标定、散射失配与风险提示。当前任务不按这份测试集继续调参。')
    add(s,'历史V35/V36、V37多帧、V39公共偏置、V40空间去噪、V72量化似然曾在各自试验有收益，但未默认启用到冻结普通液深。后续应在新安装级开发集选型，再换独立测试验证；其他场景只作防回退。不能靠增加内插点、强制归零、输出裁剪或降低分母来制造精度改善。')
    save(s,'2026-10-17_far_shallow_limitations.pdf','RGB-D液深系统 双相机远距浅液位不足报告')


performance();architecture();far_report()
print(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in OUT.glob('*.pdf')},ensure_ascii=False))
