ToF 正式 v2（2026-10-20）
=======================

用户已确认将 opaque_tof_compound_evidence_v2_20261020 作为当前正式数值基线。
这是软件选版，不代表已经通过现场工业计量认证。

ToFSceneMeter.from_empty_geometry 默认启用已验证的输入有效性、湿/干候选竞争和
有原始点支撑的最高包络。统一控制台也默认开启，避免重复修复。
视觉仅参与漂浮物最高高度，不参与液深；透明/半透明 AMCW 相位数值算法不变。

旧源码、报告和 2026-10-20 的 1,600 帧独立对照证据封存保留。
原 2026-10-17 正式 PDF 是 v1 历史基线，不应当作 v2 新统计。
v2 测试数据见 reports/evidence/compound-repair-v2-20261020/summary.json。
部分输入审计/0--50%缺失合格子类不足100，保留该限制，不因正式选版而抹掉。

诊断旧版：ToFSceneMeter.from_empty_geometry(..., evidence_repair=False)，
或控制台 --legacy-geometry；不是另一个受维护的正式版本。
新小漂浮物训练、顶部和实例排序实验应在冻结 v2 基线之外运行，
只有在新的独立测试中改善且液深逐帧不变，才可选择合并。
