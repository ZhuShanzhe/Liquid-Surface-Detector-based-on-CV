import collections,json,sys
from pathlib import Path
import cv2,numpy as np
from range_protocol import ROOT as R,band_of
audit=[]
for sp in sorted(R.glob('capture-??.json')):
    s=json.loads(sp.read_text());rows=[];fail=[]
    for accepted,items in [(True,s['qualified']),(False,s['skipped'])]:
     for item in items:
        f=Path(item['frame']);req=json.loads((f/'request.json').read_text());md=json.loads((f/'metadata.json').read_text());sem=np.load(f/'offline_geometry.npz')['semantic'];rgb=cv2.imread(str(f/'rgb.png'));mask=sem==2
        gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)/255.;pixels=rgb[mask]
        rec=dict(eligible=accepted,frame=str(f),luminance_mean=float(gray[mask].mean()) if mask.any() else None,
          luminance_p95=float(np.percentile(gray[mask],95)) if mask.any() else None,
          saturated_any_channel_fraction=float((pixels.max(axis=1)>=250).mean()) if len(pixels) else None,
          underexposed_fraction=float((gray[mask]<.04).mean()) if mask.any() else None,liquid_visible_pixels=int(mask.sum()))
        rows.append(rec)
        if md['container_shape']!=req['shape']:fail.append(dict(frame=str(f),reason='shape_request_mismatch'))
        if not np.allclose(md['surface_radius_x_m'],req['radius_m']):fail.append(dict(frame=str(f),reason='radius_request_mismatch'))
        if not np.allclose(md['surface_radius_y_m'],req['radius_m']*req['aspect']):fail.append(dict(frame=str(f),reason='aspect_request_mismatch'))
        if not np.allclose(md['horizontal_fov_deg'],req['horizontal_fov_deg']):fail.append(dict(frame=str(f),reason='fov_request_mismatch'))
        if not np.allclose(md['camera_position_m'][:2],req['camera_xy_m']):fail.append(dict(frame=str(f),reason='camera_offset_request_mismatch'))
        if accepted:
            truth=json.loads((f/'truth.json').read_text())
            if band_of(truth['liquid_height_m'])!=item['depth_band']:fail.append(dict(frame=str(f),reason='actual_liquid_outside_selected_band'))
            em=json.loads((R/'empty'/f"{item['installation_id']:04d}"/'metadata.json').read_text())
            ep=np.asarray(em['camera_to_world'],float);wp=np.asarray(md['camera_to_world'],float).copy();wp[2,3]+=req['height_m']
            if not np.allclose(ep,wp,atol=2e-5):fail.append(dict(frame=str(f),reason='fixed_camera_world_pose_changed'))
            if not np.allclose(em['camera_intrinsics'],md['camera_intrinsics']):fail.append(dict(frame=str(f),reason='fixed_camera_intrinsics_changed'))
    summary={}
    for eligible in (False,True):
        rr=[r for r in rows if r['eligible']==eligible];summary[str(eligible)]=dict(n=len(rr))
        for key in ('luminance_mean','luminance_p95','saturated_any_channel_fraction','underexposed_fraction','liquid_visible_pixels'):
            vals=[r[key] for r in rr if r[key] is not None]
            summary[str(eligible)][key]=dict(min=float(np.min(vals)),median=float(np.median(vals)),p95=float(np.percentile(vals,95)),max=float(np.max(vals))) if vals else None
    audit.append(dict(category=s['category'],summary=summary,metadata_failures=fail))
(R/'capture-domain-audit.json').write_text(json.dumps(audit,indent=2));assert not any(x['metadata_failures'] for x in audit)
print('CAPTURE_METADATA_AUDIT_OK',len(audit))
