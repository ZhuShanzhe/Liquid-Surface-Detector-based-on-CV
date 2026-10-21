"""Reuse render inputs only, selected by prespecified height quotas, never errors."""
import hashlib,json,os,shutil
from pathlib import Path
from range_protocol import ROOT,OLD,QUOTAS,BANDS,band_of

ROOT.mkdir(parents=True,exist_ok=True)
assert not (ROOT/'selection-manifest.json').exists(),'already prepared'
def link_capture(src,dst):
    dst.mkdir(parents=True,exist_ok=True)
    for p in src.iterdir():
        if not p.is_file() or p.name=='phase_observations.npz':continue
        q=dst/p.name
        if q.exists():continue
        try:os.link(p,q)
        except OSError:shutil.copy2(p,q)

manifest=dict(evaluation_revision='range_0p1_9m_fixed_10mm',old_root=str(OLD),
    selection='stable original input order; true liquid depth only; quota by depth band; no predictions consulted',
    bands_m=BANDS,per_category_quotas=QUOTAS,attempts_includes_reused=True,
    old_measurement_values_reused=0,categories=[])
for p in sorted(OLD.glob('capture-??.json')):
    s=json.loads(p.read_text());ci=s['category_id'];selected=[];excluded=[];outside=[];counts=[0]*9
    for item in s['qualified']:
        src=Path(item['frame']);h=json.loads((src/'truth.json').read_text())['liquid_height_m'];band=band_of(h)
        if band is None:outside.append(str(src));continue
        if counts[band]>=QUOTAS[band]:excluded.append(str(src));continue
        dst=ROOT/'data'/f'reused_{src.name}';link_capture(src,dst)
        gid=item['installation_id'];link_capture(OLD/'empty'/f'{gid:04d}',ROOT/'empty'/f'{gid:04d}')
        selected.append({**item,'frame':str(dst),'source':'reused_input','source_frame':str(src),'depth_band':band})
        counts[band]+=1
    rec=dict(category=s['category'],old_qualified=len(s['qualified']),range_excluded=len(outside),quota_not_selected=len(excluded),reused=len(selected),selected_per_band=counts,
        range_excluded_frames=outside,quota_not_selected_frames=excluded)
    manifest['categories'].append(rec)
    state=dict(category=s['category'],category_id=ci,bits=s['bits'],cohort=s['cohort'],attempts=len(selected),new_attempts=0,
        qualified=selected,skipped=[],next_attempt_by_band=[0]*9,selection=rec)
    (ROOT/p.name).write_text(json.dumps(state))
manifest['reused_input_images']=sum(x['reused'] for x in manifest['categories'])
manifest['range_excluded']=sum(x['range_excluded'] for x in manifest['categories'])
manifest['quota_not_selected']=sum(x['quota_not_selected'] for x in manifest['categories'])
(ROOT/'selection-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='categories'}))
