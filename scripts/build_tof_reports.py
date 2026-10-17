"""Build the two official ToF PDFs from published, independently recounted evidence."""
import argparse,hashlib,html,json,os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,LongTable,TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing,Rect,Line,Polygon,String
from reportlab.lib.utils import simpleSplit

SCENES=('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent','partial_depth_loss','low_light','glare','uneven_surface','floating_objects')
NAMES=dict(zip(SCENES,('普通漫反射','均匀透明','均匀半透明','部分深度失效','低照','眩光','不平整液面','漂浮物')))
NAMES.update(all='总体',mild='低',moderate='中',severe='强',low_extinction='低消光',medium_extinction='中消光')
ST={n:ParagraphStyle(n,fontName=f,fontSize=z,leading=l,textColor=colors.HexColor(c),wordWrap='CJK',spaceAfter=8)
    for n,f,z,l,c in [('body','CN',10.2,17,'#243449'),('small','CN',8.7,14,'#53667D'),('cell','CN',9,14,'#243449'),
      ('th','CNB',9,14,'#FFFFFF'),('H1','CNB',18,26,'#143550'),('H2','CNB',12,20,'#143550'),('cover','CNB',28,43,'#143550')]}

def p(x,style='body'):
    value=str(x).replace('–','-').replace('—','-').replace('−','-')
    return Paragraph(html.escape(value).replace('\n','<br/>'),ST[style])
def add(s,x,style='body'):s.append(p(x,style))
def chapter(s,x):s.append(PageBreak());add(s,x,'H1')
def fmt(x):return '-' if x is None else f'{x:.2f}'
def table(headers,rows,widths):
    t=LongTable([[p(v,'th') for v in headers]]+[[p(v,'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#203D59')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F5F9')]),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#D7E1EB')),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t
class Report(SimpleDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='H1':
            title=f.getPlainText();key='h'+hashlib.sha1(title.encode()).hexdigest()
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,0,False);self.notify('TOCEntry',(0,title,self.page,key))
def footer(c,d):
    if d.page==1:return
    w,h=d.pagesize;c.saveState();c.setStrokeColor(colors.HexColor('#D5E0EB'));c.line(40,30,w-40,30)
    c.setFillColor(colors.HexColor('#61738B'));c.setFont('CN',8)
    c.drawString(40,17,'RGB-D 液深系统 | ToF 正式版 V1 | 2026-10-17 | 仿真验证，非现场计量认证')
    c.drawRightString(w-40,17,str(d.page));c.restoreState()
def cover(s,title,subtitle):
    s.extend([Spacer(1,46),p('RGB-D 液深测量系统','H2'),Spacer(1,17),p(title,'cover'),Spacer(1,18),p(subtitle,'H2'),Spacer(1,28)])
    add(s,'正式版本：tof_formal_v1_20261017    发布日期：2026年10月17日')
    add(s,'仅以 ToF 作为当前开发、软件接口和性能统计基准。场景由操作员明确选择；统一输出相对容器内底面的米制高度。')
    add(s,'本报告为固定机位、限定物理假设的仿真与软件验证，不等同于任意真实设备的工业计量精度保证。','small')
    s.append(PageBreak());add(s,'目录','H2')
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('TOC',fontName='CN',fontSize=11,leading=24,textColor=colors.HexColor('#203D59'))]
    toc.dotsMinLevel=0;s.append(toc)
def save(s,path,title):
    Report(str(path),pagesize=landscape(A4),leftMargin=40,rightMargin=40,topMargin=36,bottomMargin=45,title=title,author='Liquid Surface Detector Project').multiBuild(s,onFirstPage=footer,onLaterPages=footer)
def flow(nodes,notes):
    h=60*len(nodes)+18;d=Drawing(748,h)
    for i,(node,note) in enumerate(zip(nodes,notes)):
        y=h-53-i*60;d.add(Rect(3,y,265,40,rx=5,ry=5,fillColor=colors.HexColor('#DCEAF5'),strokeColor=colors.HexColor('#9BB3C9')))
        for j,line in enumerate(simpleSplit(node,'CNB',10,243)):d.add(String(15,y+25-j*13,line,fontName='CNB',fontSize=10,fillColor=colors.HexColor('#173C59')))
        for j,line in enumerate(simpleSplit(note,'CN',9.2,448)):d.add(String(283,y+27-j*14,line,fontName='CN',fontSize=9.2,fillColor=colors.HexColor('#334B63')))
        if i<len(nodes)-1:
            d.add(Line(136,y-1,136,y-18,strokeColor=colors.HexColor('#4E7494')))
            d.add(Polygon([132,y-14,140,y-14,136,y-20],fillColor=colors.HexColor('#4E7494'),strokeColor=None))
    return d
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('CN',os.environ.get('PDF_CN_FONT','C:/Windows/Fonts/msyh.ttc'),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNB',os.environ.get('PDF_CN_BOLD_FONT','C:/Windows/Fonts/msyhbd.ttc'),subfontIndex=0))
    m=json.loads((a.input/'metrics.json').read_text(encoding='utf-8'));audit=json.loads((a.input/'evaluation-audit.json').read_text(encoding='utf-8'))
    checks=json.loads((a.input/'verification.json').read_text(encoding='utf-8'))
    metric=lambda sc,level='all',quantity='liquid':m[f'{sc}|tof|{level}|all'][quantity]
    def row(label,v):return [label,f"{v['n']} / {v['installations']}",fmt(v['mae_mm']),fmt(v['p95_mm']),fmt(v['max_mm']),fmt(v['absrel_pct']),fmt(v['pass_pct']),fmt(v['numeric_coverage_pct'])]
    headers=['场景 / 输出','帧 / 安装','MAE mm','P95 mm','最大 mm','AbsRel %','通过率 %','数值覆盖 %'];widths=[160,95,80,80,80,80,80,90]
    s=[];cover(s,'ToF 全场景性能测试报告','正式版 | 场景总表、程度分层与误差分布')
    chapter(s,'01  全场景液深性能总表')
    s.append(table(headers,[row(NAMES[sc],metric(sc)) for sc in SCENES],widths))
    add(s,'液深真值H为相对容器内底面的实际高度。普通、透明、半透明、部分失效、低照、眩光、漂浮物液深容差为max(20mm,2%H)；不平整液面代表高度为max(50mm,5%H)。最高高度单独采用max(50mm,5%真实最高高度)。')
    add(s,'这是同一正式测试集的ToF子集重统计，不是新一轮训练效果。保留全部符合原输入协议的ToF帧，不按预测误差或置信度删帧；不强制归零、不裁剪输出。','small')
    chapter(s,'02  数据来源、采样及指标定义')
    add(s,f"总计{audit['total_frames']:,}帧，仅含ToF；原冻结几何结果{audit['retained_opaque_frames']:,}帧、补充几何观察{audit['additional_geometry_frames']:,}帧、原始相位测试{audit['phase_frames']:,}帧。来源保持原标记，不把同一安装的噪声重复帧称为独立容器。最小报告小类{audit['minimum_cell']}帧，均不少于100帧。")
    add(s,'新增几何观察按输入审核：容器内部至少64个返回点、至少6/16空间格且双轴跨度至少25%；离线来源审核至少64个直接液面来源点。审核不使用预测误差，来源标签不进入推理；历史帧沿用各自原协议。原采集成对审核事实保留于来源档案，本次没有改成新的采集或追加筛选。')
    add(s,'透明类共150组安装级五先验标定，预算为1帧空槽+4帧已知液深；每安装4帧测试，共600帧。已知液深标签含0.3mm+0.1%H噪声；测试液深、材料真值、来源掩码均不传给逆解。给定的底面距离B、内参与射线尚未额外叠加系统标定误差。')
    add(s,'MAE是平均绝对误差，P95是绝对误差95%分位，最大误差不得由P95代替。AbsRel为平均|预测H-H|/H，只对H>0计算；空槽仍进入绝对误差和通过率。数值覆盖率=有限输出帧/全部纳入帧；通过率=容差内帧/全部纳入帧，未输出计未通过。')
    add(s,'本报告不按实际液深分层。程度分层为预设仿真参数，并非现场lux或饱和比例的计量阈值；不同程度和采样分布不可直接当作随机工业现场代表性。聚类bootstrap区间和来源哈希见随附JSON。','small')
    notes={
      'ordinary_diffuse':'适用于正常不透明漫反射液体。空槽建模与对称候选几何保持冻结；当前毫米级误差不等于真实硬件已验收，共模偏置和底面几何误差仍需现场检查。',
      'homogeneous_transparent':'仅适用于性质稳定、底部调制仍可观测的均匀透明液体。需要AMCW iToF三频四相原始相关图；只有深度图的SDK不能直接启用本路线。不得把等效折射率拟合当作真实材料参数的计量认证。',
      'homogeneous_translucent':'范围为底部信号仍可见的均匀轻/中消光液体；不包含强散射不透明或多层液体。相位混合、响应参数混淆与模型失配仍会产生长尾，须同时看最大误差和P95。',
      'partial_depth_loss':'程度为可见液面区域中数值空洞占比，不是全图面积，也不是错误回波比例。仅承诺本轮0%-50%范围；剩余点数量不等于可靠层身份信息。',
      'low_light':'低、中、强为预设照明/曝光及深度退化代理；不是设备标定lux。液深不自动调用RGB提亮或视觉分割。主动照明的ToF不因RGB变暗必然失效，真实红外信噪比仍须实测。',
      'glare':'低、中、强为预设高光与回波扰动程度。输入证据审核不按误差排除坏帧；高通过率不代表所有饱和、多径和镜面角度均已验证。',
      'uneven_surface':'液深评价目标为物理面积平均高度；另评价最高包络。代表平面不等于完整波形重建，局部波峰可见性与采样分布仍影响最高高度。',
      'floating_objects':'视觉按需仅用于最高高度，液深计算不变。V8 selected在ToF下沿用V5最高分支。高概率区域不等于可靠测距；局部错层或异常峰仍可能产生很大的最高高度误差。'}
    for i,sc in enumerate(SCENES,3):
        chapter(s,f'{i:02d}  {NAMES[sc]}：ToF性能')
        levels=['all','mild','moderate','severe'] if sc in ('low_light','glare') else ['all','0-10%','10-20%','20-30%','30-40%','40-50%'] if sc=='partial_depth_loss' else ['all','low_extinction','medium_extinction'] if sc=='homogeneous_translucent' else ['all']
        rows=[row(NAMES.get(l,l)+'液深',metric(sc,l)) for l in levels]
        if sc in ('uneven_surface','floating_objects'):rows.append(row('最高高度',metric(sc,quantity='highest')))
        s.append(table(headers,rows,widths));add(s,notes[sc])
        v=metric(sc);ci=v['cluster_bootstrap_95ci'];add(s,f"安装聚类bootstrap 95%区间：MAE [{fmt(ci['mae_mm'][0])}, {fmt(ci['mae_mm'][1])}] mm；通过率 [{fmt(ci['pass_pct'][0])}, {fmt(ci['pass_pct'][1])}]%。未输出{v['no_output']}帧。",'small')
        if sc=='floating_objects':
            add(s,'必须保留的异常：一帧真实最高高度134.11mm，估计3565.19mm，误差3431.08mm；其液深真实47.48mm、估计46.30mm。说明液深稳定与最高高度可靠是两个不同问题；该帧未被剔除，最高高度不宜直接用于安全联锁。','small')
    chapter(s,'11  全场景绝对误差区间占比')
    rows=[[NAMES[sc],metric(sc)['n'],*[f'{v:.2f}' for v in metric(sc)['absolute_bins_pct']]] for sc in SCENES]
    s.append(table(['场景','全部帧','<5mm %','5-20mm %','20-50mm %','50-100mm %','>=100mm %'],rows,[160,75,100,100,100,110,100]))
    add(s,'区间左闭右开；未输出不落入任何误差区间，另列未输出数。此表为液深，不把漂浮物最高高度与液深混为一个指标。','small')
    chapter(s,'12  全场景相对误差区间占比')
    rows=[[NAMES[sc],metric(sc)['relative_n'],*[f'{v:.2f}' for v in metric(sc)['relative_bins_pct']]] for sc in SCENES]
    s.append(table(['场景','非零H输出帧','<1% %','1-2% %','2-5% %','5-10% %','>=10% %'],rows,[160,100,95,95,95,100,100]))
    add(s,'小液深下，毫米级绝对误差也可能对应较大相对误差。相对误差区间与带绝对下限的工程容差通过率必须同时解读。','small')
    chapter(s,'13  延时、复现及应用限制')
    rows=[]
    for sc in SCENES:
        t=metric(sc)['latency_ms'];rows.append([NAMES[sc],t['timed_frames'],fmt(t['median']),fmt(t['p95']),fmt(t['max'])])
    s.append(table(['场景','有计时帧','中位 ms','P95 ms','最大 ms'],rows,[200,120,140,140,145]))
    add(s,'仅算法计时，包含旧/新采样及冷启动条件差异，不含硬件曝光、调制序列、传输和界面刷新；不能宣称完整系统已保证0.5s。漂浮物最高高度模块存在约1.96s最大值，P95约424ms。应常驻初始化对象，现场再测完整延时。')
    add(s,f"本次收敛验证：{checks['cli_replays']}个场景CLI回放与原ToF预测一致；{checks['phase_rows_replayed']}帧相位结果重算一致；单元回归结果见verification.json。统计独立复算覆盖{checks['metric_quantity_groups']}组数量指标。未重新训练或宣称新测试集精度提升。")
    add(s,'透明仿真使用常数等效折射率1.28-1.50，消光0、0.012-0.05或0.055-0.12/m，不是850/940nm实测水光谱；不证明近红外可以穿透9m水层。场景中的底部信号可见性、硬件多径、温漂及系统标定误差需要现场补验。','small')
    add(s,'证据：reports/evidence/tof-formal-v1-20261017/；复算：scripts/recount_tof.py；正式入口：scripts/measure_tof.py。所有数值保留原测试来源，报告选择ToF是开发范围决策，不是所有真实设备类别的普遍排名。','small')
    save(s,a.out/'2026-10-17_tof_performance.pdf','ToF正式版全场景性能测试报告')
    s=[];cover(s,'ToF 正式计算链路报告','正式版 | 按场景的结构示意图、模块注释与输入契约')
    chapter(s,'01  总体结构与正式入口')
    s.append(flow(['操作员选择八类场景之一','普通类：空槽建模 + 米制深度','透明类：五组先验 + 原始相位','分支计算 → 米制液深与诊断'],[
      'ToFSceneMeter只接受sensor_family=tof；相机类型不自动回退。',
      '六类几何场景共用空槽容器范围、内底面与坐标变换。',
      'AMCW iToF三频四相；一空槽加四组已知液深，固定液体与机位。',
      '不平整、漂浮物另输出最高高度；视觉不得覆盖液深。']))
    add(s,'硬件并未完成通用USB/工业GUI联调。普通深度SDK可服务几何链路；透明链路必须能读取原始调制相关图，不能从只有深度的输出反推为已获得原始信号。当前透明算法不承诺任意脉冲dToF硬件。','small')
    chapter(s,'02  初始化、标定与数据契约')
    s.append(table(['阶段','输入与操作','校验及限制'],[
      ['空槽几何','固定机位；内参、空槽RGB-D、容器轮廓及独立米制尺寸/锚点、重力方向。','先修复失败标定再测量；不使用当前液深真值标定当前帧。'],
      ['透明五先验','同一容器/液体/机位，H=0及四个递增已知非零H，均在容量内。','本轮预算固定5组；几何B、射线与ROI需独立标定，先验不能替代所有几何信息。'],
      ['相位观测','correlations数组[3,4,H,W]；20/30/50MHz，四个相移。','有限值、通道数、空间尺寸一致；至少64个有效调制采样点。'],
      ['输出','liquid_depth_m、diagnostics；指定场景另有highest_height_m。','数值输出不是精度保证；缺少信号可返回无数值和原因，不伪造测量。'],
      ['重新标定','机位、液体性质或容器发生显著变化时，重建几何/五先验。','强散射、多层、持续移动及超过50%失效不纳入当前承诺。']], [130,320,295]))
    for i,sc in enumerate(SCENES,3):
        chapter(s,f'{i:02d}  {NAMES[sc]}计算链路')
        if sc.startswith('homogeneous_'):
            nodes=['三频四相观测 → 复相关解调','空槽相位参考 + 空槽ROI','五先验拟合等效物理参数','全局粗搜 + 局部液深逆解','液深与可观测性诊断']
            notes=['I0-I2 + i(I3-I1)，不把相位光程直接等同于真实液面距离。',
              '仅使用空槽和操作员ROI，不使用当前帧真液面掩码。',
              '等效折射率、表面/底部混合比、消光、散射与残余偏置。',
              '介质光程与混合回波模型；不强制归零、不后裁剪。',
              '输出残差、灵敏度、外推及搜索边界；不是校准过的误差保证区间。']
        else:
            nodes=['空槽图像 → 容器几何','当前对齐ToF米制深度','容器内候选回波 → 对称评分','稳健代表液面 → 内底面液深']
            notes=['empty_geometry_v41 / empty_scene_routes_v44；保留底面、轮廓、重力及坐标变换。',
              '原ToF几何分支不变；1mm量化与距离噪声为仿真代理，不是硬件认证参数。',
              'frozen_symmetric_geometry / symmetric_visual_meter；候选按同等地位比较。',
              '相对内底面输出；液深不使用视觉覆盖、强制归零或输出裁剪。']
            if sc=='partial_depth_loss':notes[1]='按可见液面内0%-50%数值空洞适配；保留真实来源审核与空间支持。'
            if sc in ('low_light','glare'):notes[2]='按固定程度模拟回波退化；不自动以RGB增强代替测距。'
            if sc=='uneven_surface':nodes.append('V11几何包络 → 最高高度');notes.append('代表高度与最高包络分别输出，不等同完整动态波形重建。')
            if sc=='floating_objects':nodes.append('按需RGB视觉 + V5分支 → 最高高度');notes.append('V8 selected的ToF策略；视觉仅定位最高对象候选，液深保持独立。')
        s.append(flow(nodes,notes));add(s,notes_by_scene(sc),'small')
    chapter(s,'11  文件组织、运行与维护边界')
    s.append(table(['层级','正式文件','职责'],[
      ['门面','src/liquid_depth/tof_scene_meter.py','仅ToF；场景与初始化类型检查，分发几何/相位。'],
      ['透明逆解','src/liquid_depth/tof_phase_refraction.py','解调、五先验拟合、光程逆解与诊断。'],
      ['命令行','scripts/measure_tof.py','读取初始化JSON与npy/npz/RGB；按需加载最高高度视觉。'],
      ['场景配置','configs/application_route.json / tof_scene_routes.json','唯一当前支持入口、输入契约、指标与场景边界。'],
      ['模拟与回放','generate_tof_far.py / evaluate_tof_far.py / evaluate_tof_phase.py','仅ToF采样；旧观察保持原来源，不假称本次重新采集。'],
      ['复算与报告','recount_tof.py / build_tof_reports.py','独立核对误差、通过率、覆盖率；生成两份正式PDF。']], [110,355,280]))
    add(s,'启动：PYTHONPATH=src:scripts python scripts/measure_tof.py --scene ordinary_diffuse --initialization init.json --depth current.npy。透明类使用--phase-capture current.npz；漂浮物可加--rgb current.png --enable-floating-peak-vision。')
    add(s,'长期运行应只初始化一次ToFSceneMeter，然后连续measure；命令行逐帧启动会重复加载与标定。共享冻结几何源码保留历史研究兼容实现，但当前注册、相位模型、采样入口和验收均只支持ToF，不将研究脚本视为产品入口。','small')
    chapter(s,'12  验证结论与未完成事项')
    add(s,f"八场景ToF软件入口已接通；本轮{checks['cli_replays']}场景回放、{checks['phase_rows_replayed']}帧相位重算与原预测一致。此为删除未选路线后的防回退验证，不是新硬件试验或新的训练收益。")
    add(s,'主要限制：真实近红外信号预算与多径未充分验证；透明参数为等效常数；漂浮物最高高度有严重孤立异常；算法计时不含完整采集。请以性能报告的最高高度最大误差及延时尾部作为部署风险依据。')
    add(s,'参考原理：[1] Tanaka等，CVPR2016，Recovering Transparent Shape from Time-of-Flight Distortion；[2] Song与Shim，ACCV2018，Depth Reconstruction of Translucent Objects from a Single ToF Camera using Deep Residual Networks；[3] Hale与Querry，Applied Optics12(3)，555-563，1973。仅作原理参考，不宣称完整复现或已满足其真实采集条件。','small')
    add(s,'原理来源：https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Tanaka_Recovering_Transparent_Shape_CVPR_2016_paper.pdf ；https://arxiv.org/abs/1809.10917 ；https://ntrs.nasa.gov/citations/19730040258','small')
    save(s,a.out/'2026-10-17_tof_formal_pipeline.pdf','ToF正式计算链路报告')
    print(json.dumps({p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in a.out.glob('*tof*.pdf')},ensure_ascii=False))

def notes_by_scene(sc):
    if sc=='homogeneous_transparent':return '固定性质均匀液体、底部调制可见。五先验约束安装响应，不能制造新光子或证明所有透明液体都可测。'
    if sc=='homogeneous_translucent':return '前向24层散射积分、逆模型4层近似；保留模型失配。强散射或底部信号消失不强行套用。'
    if sc=='floating_objects':return '默认视觉关闭，只有明确漂浮物最高高度任务才开启；最高高度异常不能由良好液深结果掩盖。'
    if sc=='uneven_surface':return '评价物理面积平均液深及最高点；局部遮挡或非均匀采样会造成代表高度与物理面积均值偏差。'
    return '空槽标定和米制锚点是必需输入；保留原几何数值策略，不将相机类型收敛当作算法精度改进。'

if __name__=='__main__':main()
