"""Extend the original milestone report without importing its evaluation runner."""
from __future__ import annotations

import ast
import html
import json
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'output/pdf/2026-10-21_development_history.pdf'
SOURCE = ROOT / 'reports/2026-10-21_development_history.rst'
STYLES = {}


def original_milestones():
    tree = ast.parse((ROOT / 'scripts/build_publication_pdfs.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'make_history')
    assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'milestones' for t in node.targets))
    records = ast.literal_eval(assignment.value)
    assert len(records) == 20
    # Original statements remain historical facts, not current support promises.
    records[-1] = ('2026-10-16', '阶段发布与双PDF整理', records[-1][2].replace(
        '桌面目标文件夹仅保留', '当时桌面目标文件夹仅保留'))
    return records


def recent_milestones(data):
    s = data['summary']
    high_applicable = s['n'] - data['by_category']['transparent']['n'] - data['by_category']['translucent']['n']
    return [
        ('2026-10-17', '双相机比较后确定ToF主线',
         '完成ToF与结构光分场景比较，按本项目当前仿真表现与部署方向选择ToF继续开发，结构光退出正式链路。保留原双相机性能PDF，作为路线取舍的历史依据；不将这一选择解释为ToF在所有真实条件下都更优。'),
        ('2026-10-17至10-18', 'ToF透明相位与最高高度尾部问题',
         '透明/轻半透明转为均匀液体的AMCW iToF多频相位与混合回波校正，需要原始相关图及1空槽+4已知液深先验。分析漂浮物最高高度曾出现的米级错误：有空间支撑的错误区域仍可能同时具有视觉可信度与错误测距，不能只靠分类置信度排除。'),
        ('2026-10-19', '模块化统一输入与仿真工作台',
         '建立统一观测、安装标定与测量入口，普通几何和透明相位后端互斥。增加可选多标签场景建议及三维仿真展示，可调整液深、光照、漂浮物、起伏和缺失，并对照真实与估计高度。最初自动组合仍属实验入口，不替换既有性能结论。'),
        ('2026-10-20', '复合场景输入修复与正式v2',
         '修正无命中背景噪声与眩光误触发多层退化，加入空槽/湿表面候选竞争、输入有效性和有支撑最高点。新旧算法使用同一修正后输入对照，避免把仿真器纠错算成算法增益；经确认后设为正式默认，旧对称几何版继续封存。'),
        ('2026-10-20', '小漂浮物最高高度增强与正式v3',
         '增加小漂浮物训练与最高高度专用存在性探测，补救分类漏启用。独立专项中，合格漂浮物最高高度MAE从22.0降至19.3 mm，通过率从92.18%升至94.24%；液深逐帧不变，但仍有1.381 m最大误差，未以均值改善掩盖长尾。该专项与后续正式测试不是同一数据集。'),
        ('2026-10-20至10-21', '视觉分类候选与未合并记录',
         '尝试原生局部小目标、RGB-D空间特征和复合条件均衡训练。该阶段预选候选完整组合正确率88.56%、漂浮物召回88.17%，未满足预声明严格门槛，未覆盖当时正式分类头。保留失败与研究版本，不按独立测试成绩倒选模型；后续新视野优化另作记录。'),
        ('2026-10-21', '顶部安装、自动构图与独立先验',
         '补齐圆柱、椭圆、矩形容器尺寸、容器高度、相机到底面距离与偏心设置。安装或材质变化时重新生成独立空槽和先验，并联动视野、缩放与顶部安装；不合格输入在分类和测量前跳过，不能按输出误差决定是否纳入。'),
        ('2026-10-21', '新视野视觉辅助与延时控制',
         '结合原生局部证据、眩光分支与候选特征改进新视野选路，并使用原生树推理等方式减少计算开销。视觉测量仍只允许影响漂浮物最高高度，液深不由语义标签改写；标签组合正确率转为诊断，正式判断以测量误差、覆盖率和延时为主。'),
        ('2026-10-21', 'V14当前三维内部与空湿身份修复',
         '修复容器变化与湿表面投影导致的证据遗漏：按当前XYZ判断容器内部，不再要求湿帧像素在空槽时必须看见底部；结合空湿回波身份与空间一致性选取液面。统一工厂及深度接口采用V14修复，透明相位数值流程保持独立，不强制归零或裁剪。'),
        ('2026-10-21', '正式量程与固定标定误差重新测试',
         f"将正式液深范围统一为0.10-9 m、相机到底面不超过10 m，米制标定量采用截断前σ=10/3 mm、误差界限±10 mm的零均值截断高斯扰动。服务器主体测试{s['n']}帧，其中单场景1400帧、复合1100帧，各报告小类100帧。液深MAE {s['liquid']['mae_mm']:.3f} mm、通过率{s['liquid_pass']*100:.2f}%；功能通过率{s['functional_pass']*100:.2f}%，最高高度通过{s['pass_counts']['highest_pass']}/{high_applicable}帧。"),
        ('2026-10-21', 'CPU独立演示版与服务器HTML入口',
         '完成随包模型、仿真器与运行资源的Windows CPU便携演示软件，支持容器和工况交互；另增加由服务器计算支撑的HTML入口，方便其他联网电脑浏览器展示。两种演示均沿用正式测量核心，但CPU渲染/耗时与服务器测试不同，不把演示功能检查混入正式精度统计。'),
        ('2026-10-21', '正式性能报告精简与版本同步',
         f"将正式性能报告从76页精简为10页，保留场景总表、程度分层、复合场景、最高高度、误差区间与延时。仅引用服务器正式测试：稳态计算P95 {s['latency']['p95_ms']:.1f} ms、最大{s['latency']['max_ms']:.1f} ms，不含初始化、渲染、网络及界面。报告与源码同步仓库，本次开发简史同步补齐重要进展，不另附部署步骤或逐项验证记录。"),
    ]


def p(text, style='body'):
    clean = str(text).replace('–', '-').replace('—', '-').replace('−', '-')
    return Paragraph(html.escape(clean), STYLES[style])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#D5DFE9'))
    canvas.line(32, 29, A4[0] - 32, 29)
    canvas.setFont('CN', 8)
    canvas.setFillColor(colors.HexColor('#62748A'))
    canvas.drawString(32, 17, 'RGB-D 液深系统 | 更新至2026-10-21 | 仿真研发进展')
    canvas.drawRightString(A4[0] - 32, 17, f'第 {doc.page} 页')
    canvas.restoreState()


def main():
    pdfmetrics.registerFont(TTFont('CN', os.environ.get('PDF_CN_FONT', 'C:/Windows/Fonts/msyh.ttc'), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNB', os.environ.get('PDF_CN_BOLD_FONT', 'C:/Windows/Fonts/msyhbd.ttc'), subfontIndex=0))
    for name, size, leading, font, color in [
        ('body', 10, 15, 'CN', '#25364B'), ('small', 8.5, 13, 'CN', '#526578'),
        ('h1', 18, 25, 'CNB', '#17314D'), ('h2', 11, 17, 'CNB', '#17314D')]:
        STYLES[name] = ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                                     textColor=colors.HexColor(color), wordWrap='CJK', spaceAfter=0)
    data = json.loads((ROOT / 'reports/evidence/formal-v15-range01-9-fixed10mm-20261021/performance.json').read_text(encoding='utf-8'))
    old = original_milestones()
    new = recent_milestones(data)
    story, source = [], []

    def title(text, new_page=True):
        if story and new_page:
            story.append(PageBreak())
        story.extend([p(text, 'h1'), Spacer(1, 7)])
        source.append(text + '\n' + '=' * 55 + '\n')

    def text(value, style='body'):
        story.extend([p(value, style), Spacer(1, 7)])
        source.append(value + '\n\n')

    def entries(records, offset):
        for i, (date, heading, body) in enumerate(records, offset):
            caption = f'{i:02d}  {date}  {heading}'
            story.append(KeepTogether([p(caption, 'h2'), p(body), Spacer(1, 10)]))
            source.extend([caption + '\n' + '-' * 45 + '\n', body + '\n\n'])

    title('RGB-D 液深系统 开发过程与重要进展')
    text('截至2026年10月21日。沿用原版按日期记录重要工作、路线变更、失败回退与当前状态的格式。原8月至10月16日条目保留；本次补充10月17日至21日进展，不开展新训练或测量。历史阶段状态不等同于当前支持范围。')
    entries(old[:11], 1)
    title('开发过程与重要进展（续）')
    entries(old[11:], 12)
    title('近期进展：ToF主线、模块化与复合场景')
    entries(new[:6], 21)
    title('近期进展：V14正式链路、测试与演示软件')
    entries(new[6:], 27)
    title('关键决策、当前状态与尚未完成的部分')
    for value in [
        '评价量保持相对容器内底面的实际液深；波面另评面积平均高度和最高包络。普通液深容差max(20 mm,2%实际液深)，波面液深与最高高度分别采用max(50 mm,5%各自真值)，不同容差下的通过率不可直接当精度排序。',
        '当前正式开发只适配ToF：普通几何与均匀透明/半透明AMCW相位后端互斥，材料由安装先验确认；不声称视觉已自动解决材质判定。低照、眩光、波面、漂浮物可在既定普通材料边界内组合，部分缺失限0%-50%。',
        '视觉辅助选路与视觉数值测量分开：前者提出模块建议，后者仅用于漂浮物最高高度；液深仍由物理后端估计。标签不完全正确但测量合格可通过，标签正确却数值超差仍失败。',
        '服务器正式结果是在预先合格的仿真输入条件下统计；真值来源只用于测量前资格与测量后评分，不进入估计器。覆盖率不是任意原始输入的可用率，同安装多帧也不是独立安装重复。不同批次不得直接归因为算法增益。',
        '当前剩余超差主要为漂浮物最高高度低估；远距浅液位仍需关注底面基准、共同测距偏置和候选身份。多层透明、强散射、超过50%失效及持续移动未新增适配承诺；当前高通过率不代表上述难题已解决。',
        '已形成服务器统一测量与交互入口、CPU离线演示及服务器HTML展示。演示软件不是USB相机/PLC联调完成的产品；仍需真实相机噪声与相位数据、现场标定、域差验证及工业安全/计量验收。软件著作权准备不自动证明工业精度。',
    ]:
        text(value)
    text('来源与历史资料保留', 'h2')
    text('早期条目沿用2026-10-16开发记录及其归档资料；近期条目依据双相机/ToF历史报告、统一Console、复合修复、小漂浮物与选路研究记录，以及V14正式链路和固定10 mm标定测试统计。旧PDF和研究失败版本保留为开发过程，不覆盖为现版结论。', 'small')
    text('当前性能以配套ToF全场景性能报告及reports/evidence/formal-v15-range01-9-fixed10mm-20261021/performance.json为准；CPU与HTML仅记录功能交付进展，不参与该正式性能统计。完整代码与阶段资料在Git历史中保留，本简史不展开部署命令、验证明细或全部参数迭代。', 'small')
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(REPORT), pagesize=A4, leftMargin=36, rightMargin=36,
                      topMargin=34, bottomMargin=40, title='RGB-D液深系统 开发过程与重要进展',
                      author='Liquid Surface Detector Project').build(story, onFirstPage=footer, onLaterPages=footer)
    SOURCE.write_text('\n'.join(source).rstrip() + '\n', encoding='utf-8')
    print(REPORT)


if __name__ == '__main__':
    main()
