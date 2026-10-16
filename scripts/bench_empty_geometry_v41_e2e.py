"""Serial saved-profile timing; not a camera capture or network latency claim."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from liquid_depth.rail_runtime import make_product_system


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--repeats',type=int,default=8);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text())
    if not all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in protocol['source_hashes'].items()):
        raise AssertionError('source_changed_since_evaluation')
    rows=[json.loads(x) for x in (a.run/'records.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['method']=='empty_distance' and r['predicted_height_m'] is not None and r['repeat']==0]
    selected=[]
    for regime in ('nominal','adverse'):
      for group in ('far_shallow','uniform_0','uniform_4','uniform_8'):
        for shape in ('circle','ellipse','rectangle'):
            choices=[r for r in rows if r['regime']==regime and r['group']==group and r['shape']==shape]
            if choices:selected.append(choices[0])
    timings=[]
    for i,r in enumerate(selected):
        profile=a.run/'cases'/f'{r["case"]:04d}'/r['regime']/'empty_distance.yaml'
        started=time.perf_counter();meter=make_product_system(profile);initialization=time.perf_counter()-started
        meter.measure(r['frame_dir'])
        for j in range(a.repeats):
            output=a.run/'timing_outputs'/f'{i:02d}_{j:02d}'
            started=time.perf_counter();result=meter.measure(r['frame_dir'],output);elapsed=time.perf_counter()-started
            if result['accepted']:raise AssertionError('unexpected_production_output')
            timings.append(dict(case=r['case'],regime=r['regime'],group=r['group'],shape=r['shape'],repeat=j,
                initialization_ms=initialization*1000,frame_load_and_meter_ms=result['latency']['inference_total_ms'],
                including_json_overlay_ms=elapsed*1000))
    def stats(field):
        v=np.asarray([r[field] for r in timings])
        return dict(p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),p99=float(np.percentile(v,99)),
            maximum=float(v.max()),over_500ms_pct=float(np.mean(v>500)*100))
    result=dict(configurations=len(selected),timed_measurements=len(timings),serial_warm_cache=True,
        camera_capture_network_gui_scheduling_excluded=True,
        frame_load_and_meter_ms=stats('frame_load_and_meter_ms'),including_json_overlay_ms=stats('including_json_overlay_ms'),records=timings)
    (a.run/'serial-latency.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
