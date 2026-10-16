"""Hash test evidence and archive additive source files without deleting history."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from special_scene_protocol_v1 import BANDS


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    assert json.loads((a.out/'status.json').read_text())['state']=='complete'
    protocol=json.loads((a.out/'protocol.lock.json').read_text())
    assert all(sha(Path(f))==h for f,h in protocol['source_sha256'].items())
    assert '726 passed' in (a.out.parent/'full-tests-correct-path.log').read_text()
    records=[json.loads(s) for s in (a.out/'records.jsonl').read_text().splitlines()]
    assert len(records)==4752
    assert all(r['input_evidence']['eligible'] and r['input_evidence']['capture_origin']['eligible'] for r in records)
    assert all(BANDS[r['band']][0]<=r['truth']['liquid_height_m']<=BANDS[r['band']][1] for r in records)
    previous=[json.loads(s) for s in (a.out.parent/'evaluation-final'/'records.jsonl').read_text().splitlines()]
    lookup={(r['physical_id'],r['sensor'],r['scenario'],r['level'],r['repeat']):r for r in previous}
    shared=0
    for r in records:
        key=(r['physical_id'],r['sensor'],r['scenario'],r['level'],r['repeat'])
        if key not in lookup:continue
        old=lookup[key]
        for field in ('liquid_depth_m','highest_height_m'):
            if r[field] is None or old[field] is None:assert r[field]==old[field]
            else:assert abs(r[field]-old[field])<1e-9
        shared+=1
    link_checks=0;table_checks=0
    for doc in [repo/'docs/special_scene_expanded_evaluation_20261016.md',
                *(repo/'docs').glob('*_expanded_strata_20261016.md')]:
        content=doc.read_text()
        for target in re.findall(r'\]\(([^)]+)\)',content):
            if '://' not in target:
                assert (doc.parent/target).exists(),target
                link_checks+=1
        columns=None
        for line in content.splitlines():
            if line.startswith('|'):
                n=line.count('|')
                if columns is not None:assert columns==n,(doc,line)
                columns=n;table_checks+=1
            else:columns=None
    (a.out/'final-quality-audit.json').write_text(json.dumps({
        'accepted_input_frames_matching_unfiltered_same_input_predictions':shared,
        'shared_prediction_tolerance_m':1e-9,'actual_depth_strata_verified':True,
        'markdown_relative_links_verified':link_checks,'markdown_table_rows_verified':table_checks,
        'accepted_direct_origin_and_spatial_evidence_verified':True},indent=2))
    sources=sorted(set([*map(Path,protocol['source_sha256']),
        repo/'scripts/report_illumination_wave_strata_v1.py',Path(__file__),
        repo/'tests/test_special_scene_strata_v1.py',repo/'configs/current_scene_evaluation_v1.json',
        repo/'docs/special_scene_expanded_evaluation_20261016.md',
        *(repo/'docs').glob('*_expanded_strata_20261016.md')]))
    with tarfile.open(a.out/'reproducible-source.tar.gz','w:gz') as tar:
        for s in sources:tar.add(s,arcname=str(s.relative_to(repo)))
    raw={str(f.relative_to(a.root)):{'bytes':f.stat().st_size,'sha256':sha(f)}
         for f in sorted(a.root.rglob('*')) if f.is_file()}
    (a.out/'data-sha256.json').write_text(json.dumps({'root':str(a.root),'files':raw},indent=2))
    artifacts={str(f.relative_to(a.out)):{'bytes':f.stat().st_size,'sha256':sha(f)}
               for f in sorted(a.out.rglob('*')) if f.is_file() and f.name!='manifest.json'}
    manifest={'formal_evaluation':str(a.out),'primary_selected_frames':4752,
        'data_root':str(a.root),'dataset_files':len(raw),'dataset_bytes':sum(v['bytes'] for v in raw.values()),
        'source_sha256':{str(f.relative_to(repo)):sha(f) for f in sources},'artifacts':artifacts,
        'baseline_algorithm_archive_unchanged':True,'no_deletions_performed':True,
        'intermediate_runs':'evaluation,evaluation-final,evaluation-qualified,evaluation-balanced are historical diagnostic iterations, not current formal scores',
        'completed':True}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    assert all(sha(a.out/f)==v['sha256'] for f,v in artifacts.items())
    assert all(sha(repo/f)==h for f,h in manifest['source_sha256'].items())
    with tarfile.open(a.out/'reproducible-source.tar.gz') as tar:
        for f in tar.getmembers():
            if not f.isfile():continue
            digest=hashlib.sha256(tar.extractfile(f).read()).hexdigest()
            assert digest==manifest['source_sha256'][f.name]
    print(json.dumps({'archived_sources':len(sources),'dataset_files':len(raw),
        'dataset_gib':manifest['dataset_bytes']/2**30,'artifact_hashes_verified':len(artifacts),
        'source_archive_content_verified':True}),flush=True)


if __name__=='__main__':main()
