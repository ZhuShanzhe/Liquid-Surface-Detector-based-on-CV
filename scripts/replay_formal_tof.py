"""Replay a calibrated RGB-D manifest against the official ToF factory.

The manifest contains relative paths to RGB-D, empty depth and geometry, never
ground-truth heights as estimator input. Optional truth is loaded after inference.
"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from threadpoolctl import threadpool_limits
from liquid_depth.formal_tof_runtime import create_formal_meter
from liquid_depth.unified_tof import ToFInstallation,ToFObservation
from liquid_depth.sim_input_qualification import qualify_simulated_input,measure_if_eligible
from measure_release import validated_profile
REPO=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.dataset/'replay-manifest.json').read_text());profile=a.output/'profile.json';profile.write_text(json.dumps(validated_profile(REPO)))
    guard=threadpool_limits(limits=2);torch.set_num_threads(2);cv2.setNumThreads(1)
    def path(relative):
        root=a.dataset.resolve();target=(root/relative).resolve()
        if target!=root and root not in target.parents:raise ValueError('manifest_path_outside_dataset')
        return target
    frames=manifest['frames'][:a.limit or None]
    with (a.output/'measurements.jsonl').open('w') as stream:
        for i,item in enumerate(frames):
            folder=path(item['folder']);req=json.loads((folder/'request.json').read_text())
            raw=np.load(folder/'depth_m.npy',allow_pickle=False);valid=np.load(folder/'capture_valid.npy',allow_pickle=False)
            rgb=cv2.imread(str(folder/'rgb.png'));semantic=np.load(folder/'offline_geometry.npz',allow_pickle=False)['semantic']
            q=qualify_simulated_input(raw,semantic,material='ordinary_diffuse',has_floats=req['float_count']>0,minimum_depth_m=.08)
            if not q['eligible']:
                stream.write(json.dumps(dict(frame=item['folder'],qualification=q,inference_executed=False))+'\n');continue
            bundle=json.loads(path(item['geometry']).read_text());dry=np.load(path(item['dry']),allow_pickle=False)
            install=ToFInstallation('replay-'+str(i),bundle,dry,confirmed_material='ordinary_diffuse')
            meter=create_formal_meter(install,profile_path=profile)
            observation=ToFObservation(str(i),rgb,raw,0.,installation_id=install.installation_id,capture_validity=valid)
            begin=time.perf_counter();q=qualify_simulated_input(raw,semantic,material='ordinary_diffuse',has_floats=req['float_count']>0,minimum_depth_m=.08)
            result=measure_if_eligible(meter,observation,q);torch.cuda.synchronize();ms=(time.perf_counter()-begin)*1000
            truth=json.loads((folder/'truth.json').read_text())
            stream.write(json.dumps(dict(frame=item['folder'],qualification=q,inference_executed=True,liquid=result.get('liquid_depth_m'),highest=result.get('highest_height_m'),
                truth={k:truth[k] for k in ('liquid_height_m','highest_height_m')},wall_ms=ms,classification=result['classification']))+'\n')
            stream.flush();print('REPLAY',i+1,len(frames),round(ms,2),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(frames=len(frames),exact_tags_are_gate=False)))
if __name__=='__main__':main()
