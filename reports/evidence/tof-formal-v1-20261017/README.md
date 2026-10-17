# ToF 正式版证据

`tof_formal_v1_20261017`：仅保留冻结比较测试中全部5744帧ToF，预测和真值不变；不是重新训练或新测试集。

- `current-tof-predictions.jsonl.gz`：全部正式ToF帧，保留原cohort、安装聚类、原输入路径。
- `metrics.json`：场景总体及程度分层，不按实际深度分层。
- `evaluation-audit.json`：来源哈希、总帧数、来源构成、100帧配额。
- `verification.json`：八场景CLI及600帧相位回放、统计核对、正式入口源哈希。
- `pytest.log`：本次完整软件回归，不代表硬件精度验收。
- `pdf-qa.json`：两份正式报告文本、目录与页面边界检查，另有PNG视觉检查。
- `release_manifest.json`：正式源码、结果与PDF校验和。

复算：`python scripts/recount_tof.py`（只需NumPy）。原相机比较报告和证据用于说明开发决策；原实现可从提交 `15bfe99` 获取。当前正式配置只接受 `tof`。
