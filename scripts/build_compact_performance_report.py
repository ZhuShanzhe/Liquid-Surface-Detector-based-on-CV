"""Build the concise formal report exclusively from frozen test statistics."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Spacer, PageBreak, Table, TableStyle, SimpleDocTemplate

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/evidence/formal-v15-range01-9-fixed10mm-20261021/performance.json'
FAMILIES = ['ordinary', 'transparent', 'translucent', 'low_light', 'glare',
            'uneven_surface', 'floating_objects', 'partial_depth_loss']
WIDTH = 765
STYLES = {
    key: ParagraphStyle(key, fontName='CNB' if key in ('title', 'h1', 'h2', 'th') else 'CN',
                        fontSize=size, leading=leading, textColor=colors.HexColor(color),
                        wordWrap='CJK', spaceAfter=after)
    for key, size, leading, color, after in [
        ('title', 30, 45, '#183B53', 16), ('h1', 19, 28, '#183B53', 14),
        ('h2', 12, 19, '#183B53', 8), ('body', 10.2, 17, '#2D4052', 9),
        ('small', 8.5, 13, '#526578', 7), ('cell', 8.8, 12.5, '#243C50', 0),
        ('th', 8.7, 12.5, '#FFFFFF', 0)]
}


def p(value, style='body'):
    text = str(value).replace('–', '-').replace('—', '-').replace('−', '-')
    return Paragraph(html.escape(text).replace('\n', '<br/>'), STYLES[style])


def number(value, digits=2):
    return '-' if value is None else f'{value:.{digits}f}'


def percentage(value):
    return number(value * 100)


class CompactReport:
    def __init__(self, data):
        self.data = data
        self.story = []
        self.source = []

    def text(self, text, style='body'):
        self.story.append(p(text, style))
        self.source.append(text + '\n')

    def chapter(self, title):
        self.story.append(PageBreak())
        self.text(title, 'h1')
        self.source.append('-' * 60 + '\n')

    def table(self, headers, rows, weights, caption=None, compact=False):
        if caption:
            self.text(caption, 'h2')
        widths = [WIDTH * w / sum(weights) for w in weights]
        table = Table([[p(x, 'th') for x in headers]] +
                      [[p(x, 'cell') for x in row] for row in rows], colWidths=widths,
                      repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#20465F')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#EFF4F7')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#CFDCE5')),
            ('LEFTPADDING', (0, 0), (-1, -1), 7),
            ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5 if compact else 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5 if compact else 5),
        ]))
        self.story.extend([table, Spacer(1, 9)])
        self.source.extend(['.. list-table::\n   :header-rows: 1\n\n',
                            '   * - ' + '\n     - '.join(map(str, headers)) + '\n'])
        for row in rows:
            self.source.append('   * - ' + '\n     - '.join(map(str, row)) + '\n')
        self.source.append('\n')

    def liquid_row(self, label, st):
        q = st['liquid']
        return [label, str(st['n']), number(q['mae_mm']), number(q['p95_mm']),
                number(q['max_mm']), percentage(q['absrel']),
                percentage(st['liquid_pass']), percentage(q['coverage'])]

    def liquid_table(self, entries, caption=None):
        self.table(['场景 / 程度', '帧数', 'MAE mm', 'P95 mm', '最大 mm', 'AbsRel %',
                    '通过率 %', '覆盖率 %'],
                   [self.liquid_row(label, st) for label, st in entries],
                   [2.6, .65, .9, .9, .9, .9, .95, .95], caption)

    def build(self):
        d = self.data
        names = d['category_labels']
        cats = d['by_category']
        families = d['by_scene_family']
        family_name = lambda key: d['scene_family_meta'][key]['label']
        compounds = [k for k in names if k.startswith('compound_')]
        summary = d['summary']

        self.story.append(Spacer(1, 60))
        self.text('RGB-D 液深测量系统', 'h2')
        self.text('ToF 全场景性能测试报告', 'title')
        self.text('正式链路 V14 | 精简结果版', 'h1')
        self.story.append(Spacer(1, 24))
        self.text('报告日期：2026年10月21日')
        self.text('数据来源：服务器正式链路测试，不包含CPU桌面软件的测试结果。')
        self.text('本次仅精简报告，不改算法、不重构测试结果。')
        self.text('包含单场景、复合场景、程度分层、误差分布、最高高度及计算延时。')
        self.text('本报告为合格仿真输入条件下的软件测试结果，不等同于真实相机的现场计量认证。', 'small')

        self.chapter('目录与简要测试说明')
        self.table(['章节', '页码'], [
            ['全场景结果总览', '3'], ['普通、透明与半透明', '4'], ['低照与眩光程度分层', '5'],
            ['部分失效、不平整液面与漂浮物', '6'], ['复合场景液深与任务通过率', '7'],
            ['最高高度结果', '8'], ['液深误差区间占比', '9'], ['计算延时与主要结论', '10']], [6, 1])
        self.text('测试数据：使用自带仿真构建多种容器、顶部相机安装、光照、液面与回波条件；部分历史合格输入按新范围复用，其余补采，全部主体测量重新执行。每个报告小类均保留合格输入，不按输出误差删帧。', 'small')
        self.text('范围与先验：液深0.10-9 m，相机到底面不超过10 m；米制标定量使用零均值高斯扰动，截断前σ=10/3 mm、绝对误差限制在±10 mm。空槽图建立底面基准；透明与半透明采用1帧空槽加4帧已知液深先验。', 'small')
        self.text('测量前资格检查使用仿真可见性与真实回波来源，排除客观证据不足输入，真值不传给估计器。因此以下覆盖率以纳入测试的合格帧为分母，不是任意原始输入的可用率。同安装可有多帧，样本比例是描述性结果。', 'small')

        self.chapter('01  全场景结果总览')
        self.liquid_table([(family_name(k), families[k]) for k in FAMILIES] +
                          [('复合场景合计', d['by_cohort']['composite']), ('全部测试', summary)])
        self.text('液深为相对容器内底面的实际高度；不平整液面采用面积平均高度。MAE为平均绝对误差，P95为绝对误差的95%分位，AbsRel为平均绝对相对误差。', 'small')
        self.text('液深通过：普通及其非波面组合使用max(20 mm, 2%实际液深)；含不平整液面的情况使用max(50 mm, 5%实际液深)。覆盖率=有限液深输出帧/纳入帧；未输出计未通过。', 'small')
        self.text(f"全部液深通过{summary['pass_counts']['liquid_pass']}/{summary['n']}帧；按任务所需高度联合判断，通过{summary['pass_counts']['functional_pass']}/{summary['n']}帧。分类标签不完全一致但测量合格的帧仍计通过。", 'small')

        self.chapter('02  普通、透明与半透明')
        for key, note in [
            ('ordinary', '正常不透明漫反射液体。空槽建模、容器范围约束与液面候选几何共同估计液深。'),
            ('transparent', '限均匀、平整、无遮挡透明液体。使用ToF多频原始相关量及已知液深先验校正混合回波。'),
            ('translucent', '限底部信号仍可观测的均匀轻/中消光液体；不包含强散射不透明液体与多层混合液体。')]:
            self.liquid_table([(names[key], cats[key])], names[key])
            self.text(note, 'small')
        self.text('透明两类的结果依赖相位先验和当前物理代理模型，不能直接推广到只有SDK深度图的任意ToF相机；本报告不把拟合光学参数作为材料参数计量值。', 'small')

        self.chapter('03  低照与眩光程度分层')
        for key in ['low_light', 'glare']:
            entries = [('总体', families[key])] + [(names[f'{key}_{level}'], cats[f'{key}_{level}'])
                                                   for level in ['low', 'medium', 'strong']]
            self.liquid_table(entries, family_name(key))
        self.text('低、中、强是预设仿真程度，不是现场照度或饱和比例的计量等级。RGB外观变化与主动ToF回波退化分别模拟；本表不代表任意强暗光、镜面角度或红外饱和条件均可测量。', 'small')

        self.chapter('04  部分失效、不平整液面与漂浮物')
        self.liquid_table([('总体', families['partial_depth_loss'])] +
                          [(names[k], cats[k]) for k in ['loss_low', 'loss_medium', 'loss_high']],
                          '部分深度失效：按可见液面缺失比例划分')
        self.text('失效比例以可见液面区域为分母，不是全图面积，也不是错误回波比例。本报告只覆盖0%-50%缺失，且保留足够真实液面证据。', 'small')
        self.liquid_table([(names[k], cats[k]) for k in ['uneven_surface', 'floating_objects']],
                          '不平整液面与漂浮物：液深输出')
        self.text('波面液深评价面积平均高度，最高包络另列。视觉仅在漂浮物最高高度测量时使用，不接管液深计算；漂浮物顶部漏选仍可能造成最高高度低估。', 'small')

        self.chapter('05  复合场景液深与任务通过率')
        rows = []
        for key in compounds:
            st = cats[key]
            q = st['liquid']
            rows.append([names[key], st['n'], number(q['mae_mm']), number(q['p95_mm']),
                         number(q['max_mm']), percentage(q['absrel']), percentage(st['liquid_pass']),
                         percentage(st['functional_pass']), percentage(q['coverage'])])
        self.table(['复合类型', '帧数', 'MAE mm', 'P95 mm', '最大 mm', 'AbsRel %',
                    '液深通过 %', '任务通过 %', '覆盖率 %'], rows,
                   [3.0, .6, .9, .9, .9, .85, .95, .95, .9])
        self.text('任务通过=液深合格，且真实波面或漂浮物存在时最高高度也合格；不要求视觉标签组合完全一致。最高高度结果见下一页。复合测试限已实现的普通不透明液体组合，不包含透明液体与波面/漂浮物跨物理边界组合。', 'small')

        self.chapter('06  最高高度结果')
        entries = [(family_name(k), families[k]) for k in FAMILIES if k not in ['transparent', 'translucent']]
        entries += [(names[k], cats[k]) for k in compounds]
        rows = []
        for label, st in entries:
            q = st['highest']
            rows.append([label, f"{q['numeric']}/{st['n']}", number(q['mae_mm']), number(q['p95_mm']),
                         number(q['max_mm']), percentage(q['absrel']),
                         f"{st['pass_counts']['highest_pass']}/{st['n']}"])
        self.table(['场景', '输出/适用帧', 'MAE mm', 'P95 mm', '最大 mm', 'AbsRel %', '通过帧/适用帧'],
                   rows, [3, 1.1, .9, .9, .9, .9, 1.3])
        q = summary['highest']
        applicable = sum(st['n'] for st in families.values()) - families['transparent']['n'] - families['translucent']['n'] + d['by_cohort']['composite']['n']
        self.text(f"最高高度容差max(50 mm, 5%实际最高高度)。适用几何链路输出{q['numeric']}/{applicable}帧，通过{summary['pass_counts']['highest_pass']}/{applicable}帧（{percentage(summary['pass_counts']['highest_pass']/applicable)}%）；透明/半透明不提供该输出，不计漏输出。", 'small')

        self.chapter('07  液深误差区间占比')
        entries = [(family_name(k), families[k]) for k in FAMILIES] + [('复合合计', d['by_cohort']['composite']), ('全部', summary)]
        for field, labels, caption in [
            ('absolute_bins', ['≤5 mm', '(5,20] mm', '(20,50] mm', '>50 mm'], '绝对误差占比（%）'),
            ('relative_bins', ['≤1%', '(1,2]%', '(2,5]%', '>5%'], '相对误差占比（%）')]:
            rows = []
            for label, st in entries:
                q = st['liquid']
                bins = list(q[field].values())
                assert len(bins) == 6 and sum(b['n'] for b in bins) == q['numeric']
                shares = [b['fraction'] for b in bins[:3]] + [sum(b['fraction'] for b in bins[3:])]
                rows.append([label, q['numeric'], *map(percentage, shares)])
            self.table(['场景', '帧数', *labels], rows, [2.7, .7, 1, 1, 1, 1], caption, compact=True)
        self.text('区间按绝对误差或|预测-真值|/真值统计；各占比因四舍五入可能不严格加和为100%。本报告不按实际液深再分层。', 'small')

        self.chapter('08  计算延时与主要结论')
        self.table(['类别', '帧数', 'P50 ms', 'P95 ms', '最大 ms', '>500 ms帧数'],
                   [[label, st['n'], number(st['latency']['p50_ms'], 1),
                     number(st['latency']['p95_ms'], 1), number(st['latency']['max_ms'], 1),
                     st['latency']['over_500']] for label, st in entries[:-1]] +
                   [['全部', summary['n'], number(summary['latency']['p50_ms'], 1),
                     number(summary['latency']['p95_ms'], 1), number(summary['latency']['max_ms'], 1),
                     summary['latency']['over_500']]], [2.7, .7, 1, 1, 1, 1])
        self.text('延时为服务器测试环境下的稳态测量计算，不含初始化、仿真渲染、磁盘读取、网络与界面，不是CPU软件或浏览器交互的端到端耗时。', 'small')
        self.text('主要结论', 'h2')
        self.text('• 本批合格输入的全部液深均满足各自容差，但通过率不等同于相对误差始终很小，应同时看绝对/相对分布及最大误差。', 'small')
        self.text('• 剩余功能超差集中在漂浮物最高高度，均为低估；最高实例存在真实返回仍不保证峰顶被正确选中。液深合格不能代替最高高度合格。', 'small')
        self.text('• 远距离浅液位对底面基准、共模测距偏置与层候选混淆更敏感。该轮统计是冻结算法的结果，不是新优化增益；缺少证据的任意极端输入不在已验证范围。', 'small')
        self.text('• 仿真材质、回波与相位模型仍有边界，真实设备多径、温漂及污染遮挡需另行验证，不能将此结果作为工业现场精度保证。', 'small')


def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont('CN', 8)
    canvas.setFillColor(colors.HexColor('#63788A'))
    canvas.drawString(38, 20, 'ToF 正式链路 V14 | 仿真性能测试 | 2026-10-21')
    canvas.drawRightString(803, 20, str(doc.page))
    canvas.restoreState()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'output/pdf/2026-10-21_tof_v14_range01-9_fixed10mm_performance.pdf')
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont('CN', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNB', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
    data = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    assert data['primary_calibration_variant'] == 'operator_noise'
    assert data['summary']['n'] == 2500
    report = CompactReport(data)
    report.build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(args.output), pagesize=landscape(A4), leftMargin=38, rightMargin=38,
                      topMargin=30, bottomMargin=38, title='ToF全场景性能测试报告 - 精简结果版',
                      author='Liquid Surface Detector Project').build(report.story, onFirstPage=footer, onLaterPages=footer)
    if args.source:
        args.source.write_text('\n'.join(report.source), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
