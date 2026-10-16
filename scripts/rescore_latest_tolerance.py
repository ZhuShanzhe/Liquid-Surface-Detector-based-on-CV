"""Re-score retained independent simulation records in metres of liquid height.

The threshold is max(0.020 m, 0.02 * true liquid height above the container
bottom). Missing/non-finite predictions count against all-frame pass rate.
This is a reporting-only operation; it never changes predictions or gates.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')
OUT = ROOT / 'latest-scene-tolerance-20261013'


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def score(rows: list[dict], truth_key: str, pred_key: str) -> dict:
    errors = []
    passing = 0
    for row in rows:
        truth, pred = row.get(truth_key), row.get(pred_key)
        if not isinstance(truth, (int, float)) or not math.isfinite(truth):
            raise ValueError(f'Invalid liquid-height ground truth: {truth!r}')
        if isinstance(pred, (int, float)) and math.isfinite(pred):
            error = abs(pred - truth)
            errors.append(error)
            passing += error <= max(0.020, 0.02 * truth)
    errors.sort()
    n = len(rows)
    return {
        'frames': n,
        'numeric': len(errors),
        'missing': n - len(errors),
        'numeric_coverage_pct': round(100 * len(errors) / n, 2) if n else None,
        'all_frame_pass_pct': round(100 * passing / n, 2) if n else None,
        'conditional_pass_pct': round(100 * passing / len(errors), 2) if errors else None,
        'conditional_mae_mm': round(1000 * statistics.mean(errors), 2) if errors else None,
        'conditional_p95_mm': round(1000 * errors[math.ceil(0.95 * len(errors)) - 1], 2) if errors else None,
    }


def main() -> None:
    result = {
        'criterion': 'abs(predicted_liquid_height_m - true_liquid_height_m) <= max(0.020, 0.02 * true_liquid_height_m)',
        'metric_reference': 'actual liquid height relative to the container bottom, not camera range',
        'missing_policy': 'no numeric output fails the all-frame tolerance test',
        'datasets': {},
    }

    transparent_path = ROOT / 'transparent-sr-v33-20261012' / 'test-raw.json'
    transparent = json.loads(transparent_path.read_text(encoding='utf-8'))['records']
    tr_result = {}
    for condition in ('transparent', 'translucent_mild'):
        stage = 'baseline_v32' if condition == 'transparent' else 'midpoint_dense'
        rows = [r for r in transparent if r['stage'] == stage and r['condition'] == condition]
        tr_result[condition] = score(rows, 'true_height_m', 'predicted_height_m')
        tr_result[condition]['route_stage'] = stage
        for regime in sorted({r['material_regime'] for r in rows}):
            tr_result[regime] = score([r for r in rows if r['material_regime'] == regime], 'true_height_m', 'predicted_height_m')
    result['datasets']['transparent_v33'] = {'source': str(transparent_path),
        'stage_by_condition': {'transparent': 'baseline_v32', 'translucent_mild': 'midpoint_dense'},
        'scenes': tr_result}

    other_path = ROOT / 'shared-empty-v45-independent-full-20261012' / 'records.jsonl'
    other = read_jsonl(other_path)
    other_result = {}
    for scene in ('low_light', 'glare', 'floating_objects', 'uneven_surface'):
        route = 'adapted_v45'
        rows = [r for r in other if r['scenario'] == scene and r['route'] == route and r['geometry'] == 'estimated']
        other_result[scene] = {'route': route, 'geometry': 'estimated', **score(rows, 'true_height_m', 'prediction_m')}
    result['datasets']['other_scenes_v45'] = {'source': str(other_path), 'scenes': other_result,
                                             'note': 'Research-only shared empty-container initialization; depth_failure superseded by V50.'}

    recovery_path = ROOT / 'recovery-v50-confirmation-20261013' / 'records.jsonl'
    recovery = read_jsonl(recovery_path)
    rec_result = {}
    for level in range(0, 51, 10):
        rows = [r for r in recovery if r['level'] == level]
        rec_result[str(level)] = score(rows, 'truth_m', 'new_m')
    result['datasets']['depth_failure_v50'] = {'source': str(recovery_path), 'scenes': rec_result,
                                                 'note': 'Targeted 0-50% liquid-surface depth loss; gated optional recovery route.'}

    unrestricted_path = ROOT / 'unrestricted-v51b-v50-attribution-20261013' / 'records.jsonl'
    unrestricted = read_jsonl(unrestricted_path)
    un_result = {}
    for level in range(0, 51, 10):
        rows = [r for r in unrestricted if r['target_loss_pct'] == level and r['paired_eligible']]
        un_result[str(level)] = score(rows, 'true_height_m', 'broad_container_median_m')
    native = [r for r in unrestricted if r['target_loss_pct'] == 0]
    native_score = score(native, 'true_height_m', 'broad_container_median_m')
    attempted_frames = 396
    if len(native) > attempted_frames:
        raise ValueError('Native records exceed the documented attempted-frame count')
    native_passes = round(native_score['all_frame_pass_pct'] * len(native) / 100)
    native_score['frames'] = attempted_frames
    native_score['missing'] += attempted_frames - len(native)
    native_score['numeric_coverage_pct'] = round(100 * native_score['numeric'] / attempted_frames, 2)
    native_score['all_frame_pass_pct'] = round(100 * native_passes / attempted_frames, 2)
    result['datasets']['ordinary_v51b_diagnostic'] = {'source': str(unrestricted_path), 'scenes': un_result,
        'all_native_attempts': native_score,
        'note': 'Candidate-free diagnostic, not the default industrial output; paired rows are separate from the 396-attempt native audit, including eight geometry-initialization failures.'}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = [
        '# 各场景最新版仿真结果：统一液深容差（2026-10-13）', '',
        '通过条件：`|预测液深−真实液深| ≤ max(20 mm, 真实液深×2%)`。液深相对容器底面，**不是相机距离**。',
        '数值覆盖率=输出有限数值的帧数/全部测试帧数；全帧通过率的分母包括未输出帧。MAE、P95仅对数值帧计算。',
        '此处仅重算记录，未改预测、归零、裁剪或门控；均为仿真结果，不代表工业现场验收。', '',
        '| 场景与链路 | 帧数 | 数值覆盖率 | 全帧通过率 | 数值帧MAE | 数值帧P95 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    labels = {
        'transparent_v33': {'transparent': '透明 V33', 'translucent_mild': '半透明 V33'},
        'other_scenes_v45': {'low_light': '低照 V45', 'glare': '眩光 V45', 'floating_objects': '漂浮物 V45', 'uneven_surface': '不平整 V45'},
    }
    for dataset in ('transparent_v33', 'other_scenes_v45'):
        for key, label in labels[dataset].items():
            s = result['datasets'][dataset]['scenes'][key]
            lines.append(f"| {label} | {s['frames']} | {s['numeric_coverage_pct']:.2f}% | {s['all_frame_pass_pct']:.2f}% | {s['conditional_mae_mm']:.2f} mm | {s['conditional_p95_mm']:.2f} mm |")
    for dataset, label in (('depth_failure_v50', '深度丢失 V50'), ('ordinary_v51b_diagnostic', '普通无候选诊断 V51b')):
        for level, s in result['datasets'][dataset]['scenes'].items():
            lines.append(f"| {label}：{level}% | {s['frames']} | {s['numeric_coverage_pct']:.2f}% | {s['all_frame_pass_pct']:.2f}% | {s['conditional_mae_mm']:.2f} mm | {s['conditional_p95_mm']:.2f} mm |")
    s = native_score
    lines.append(f"| 普通 V51b 原生完整审计 | {s['frames']} | {s['numeric_coverage_pct']:.2f}% | {s['all_frame_pass_pct']:.2f}% | {s['conditional_mae_mm']:.2f} mm | {s['conditional_p95_mm']:.2f} mm |")
    lines += ['', '## 解释与边界', '',
        '- 透明 V33 使用冻结的 baseline_v32，半透明使用 midpoint_dense；均为顶部视角、单一均匀液体、5 帧已知液深先验的冻结独立测试，不代表多层透明容器或真实传感器验证。',
        '- V45 其他场景沿用“空槽 RGB-D + 人工米制锚点”几何估计。其低数值覆盖率主要是旧门控造成，不能把条件 MAE 当成全帧性能。',
        '- 深度丢失 V50 是 0–50% 专用恢复链路；不将 75–97% 的超范围试验混入本表。',
        '- 普通 V51b 是放宽候选的诊断链路，不是已部署默认链路；“原生完整审计”包含 353 配对帧、35 自然高失效帧及 8 帧几何初始化失败，故不能只报配对集的通过率。',
        '- 多层透明 V10 的现存评估使用相机射线表面深度代理而非相对底面的真实液深，不能伪装成同口径重算；保留原始独立评估并标记待重测。',
        '- 每行源记录路径及精确分母见 `summary.json`。旧 5 mm/2% 报告仅具历史对照价值，不应与本表混算。',
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(OUT / 'REPORT.md')


if __name__ == '__main__':
    main()
