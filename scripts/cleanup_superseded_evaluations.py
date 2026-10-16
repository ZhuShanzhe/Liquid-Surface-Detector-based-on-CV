"""Remove explicitly superseded *evaluation outputs* after exact-path checks.

Raw datasets, model weights, source code, and documents are deliberately out
of scope. Run without --execute to inspect the exact inventory first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation').resolve(strict=True)
MANIFEST = ROOT / 'latest-scene-tolerance-20261013' / 'cleanup-manifest.json'

SUPERSEDED = (
    'empty-geometry-v41-e2e-check-20261012',
    'empty-geometry-v41-e2e-dev-20261012',
    'empty-geometry-v41-e2e-drift-20261012',
    'empty-geometry-v41-e2e-drift-control-20261012',
    'image-refraction-v23-20261011',
    'ordinary-v35-dev', 'ordinary-v35-dev-adaptive', 'ordinary-v35-test-20261012',
    'ordinary-v36-dev', 'ordinary-v36-dev-weights', 'ordinary-v36-test-20261012',
    'ordinary-v37-dev', 'ordinary-v37-dev2', 'ordinary-v37-dev2-guarded', 'ordinary-v37-test-20261012',
    'ordinary-v39-dev-20261012', 'ordinary-v39-drift-20261012', 'ordinary-v39-replay-20261012',
    'ordinary-v39-test-20261012', 'ordinary-v39b-drift-20261012',
    'ordinary-v39b-over10-analysis-20261012', 'ordinary-v39b-replay-20261012',
    'ordinary-v39b-test-20261012', 'ordinary-v40-dev-20261012',
    'ordinary-v40-outliers-20261012', 'ordinary-v40-outliers-20261012-final',
    'ordinary-v40-rho0-20261012', 'ordinary-v40-rho0-20261012-final',
    'ordinary-v40-rho05-20261012', 'ordinary-v40-rho05-20261012-final',
    'ordinary-v40-test-20261012', 'ordinary-v40-test-20261012-final',
    'other-scenes-v34-20261012',
    'prior-reliability-v24-20261011', 'prior-reliability-v24b-20261011',
    'recovery-v49-development-20261013', 'recovery-v49-heldout-20261013',
    'refraction-prior-v22-20261011',
    'refraction-shared-empty-v42-final-pilot-20261012',
    'refraction-shared-empty-v42-pilot-20261012',
    'refraction-shared-empty-v42-stratified-20261012',
    'refraction-shared-empty-v42-stratified2-20261012',
    'scene_eval_v2_glare_v1', 'scene_eval_v2_glare_v1_final', 'scene_eval_v2_glare_v1_fixed',
    'scene_eval_v2_identity', 'scene_eval_v2_identity_fixed',
    'scene_eval_v2_precision_v2_replace',
    'shared-empty-v42-full-20261012', 'shared-empty-v42-smoke-20261012',
    'shared-empty-v43-geometry-probe-20261012', 'shared-empty-v43-geometry-probe2-20261012',
    'shared-empty-v43-independent-full-20261012', 'shared-empty-v43-independent-full2-20261012',
    'shared-empty-v43-independent-geometry-probe-20261012',
    'shared-empty-v44-development-full-20261012', 'shared-empty-v44-development-smoke-20261012',
    'shared-empty-v45-smoke-20261012', 'shared-empty-v45-smoke-real-20261012',
    'support-audit-v46-full-20261012', 'support-audit-v46-smoke-20261012',
    'transparent-capacity-v28-20261011', 'transparent-diagnostic-v25-20261011',
    'transparent-finite-v30-20261011', 'transparent-focus-v32-20261012',
    'transparent-range-v31-20261011', 'transparent-refinement-v26-20261011',
    'transparent-relative-v29-20261011', 'transparent-shallow-v27-20261011',
    'uniform-v48-geometry-probe-20261013', 'uniform-v48-independent-full-20261013',
    'uniform-v48-smoke-20261013', 'uniform-v49-heldout-baseline-20261013',
    'unrestricted-v51-v50-allframes-20261013', 'unrestricted-v51c-v50-trimmed-20261013',
)

RETAIN = (
    'empty-geometry-v41-e2e-test-20261012',
    'shared-empty-v45-independent-full-20261012',
    'transparent-sr-v33-20261012',
    'recovery-v50-confirmation-20261013',
    'uniform-v50-confirmation-baseline-20261013',
    'unrestricted-v51b-v50-attribution-20261013',
    'support-audit-v47-natural-to50-20261012',
    'scene_eval_v2_glare_v1_fused',
    'scene_eval_v2_identity_final',
    'scene_eval_v2_precision_v2_fused',
    'latest-scene-tolerance-20261013',
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert not set(SUPERSEDED) & set(RETAIN)
    entries = []
    for name in SUPERSEDED:
        path = ROOT / name
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != ROOT:
            raise RuntimeError(f'Unexpected cleanup target: {path}')
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        entries.append({'name': name, 'bytes': size})
    for name in RETAIN:
        path = ROOT / name
        if not path.is_dir() or path.is_symlink() or path.resolve().parent != ROOT:
            raise RuntimeError(f'Retained evaluation is missing or unexpected: {path}')
    total_gib = sum(e['bytes'] for e in entries) / 1024 ** 3
    print(f'{len(entries)} exact superseded directories; {total_gib:.2f} GiB; execute={args.execute}')
    for entry in entries:
        print(f"{entry['bytes'] / 1024 ** 3:7.2f} GiB  {entry['name']}")
    if not args.execute:
        return
    manifest = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'root': str(ROOT),
        'policy': 'Only exact, superseded evaluation outputs removed; raw datasets, weights, source, documents retained.',
        'deleted': entries,
        'retained_core': list(RETAIN),
        'recovery': 'Deleted output directories are not directly recoverable; rerun the versioned evaluator from source/datasets to regenerate.',
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for entry in entries:
        shutil.rmtree(ROOT / entry['name'])
    print(f'Completed; manifest: {MANIFEST}')


if __name__ == '__main__':
    main()
