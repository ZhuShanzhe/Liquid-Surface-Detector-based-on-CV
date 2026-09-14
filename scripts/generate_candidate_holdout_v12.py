#!/usr/bin/env python3
"""Small new-seed simulation holdout, never used for fitting or calibration."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import bpy

sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026091601)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output_root.exists():raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    groups=('transparent','translucent','multilayer','compound')
    bands=((.1,.3),(.3,1.),(1.,3.),(3.,10.))
    settings=argparse.Namespace(engine='eevee',render_samples=32)
    schedule=gen._simulation.SCENARIO_PROFILES['calibration']
    for i in range(64):
        scenario=groups[i//16]; low,high=bands[(i//4)%4]
        # New seeds, four independent draws per scene/range bucket.
        severity_slot=2*(i%4)+((i//4)%2)
        index=schedule.index(scenario)+len(schedule)*severity_slot
        scene=gen.sample_scene(index,seed=args.seed+i*1009,width=320,height=180,
                               min_distance_m=low,max_distance_m=high,
                               scenario_profile='calibration',camera_profile='industrial_top')
        scene=replace(scene,split='test',sequence_id=f'v12_fresh_{args.seed}_{i:04d}')
        folder=args.output_root/'samples'/f'{i:08d}';folder.mkdir(parents=True)
        rng=random.Random(args.seed+i*104729)
        gen.clear_scene();gen.create_environment(scene,rng);gen.create_container(scene)
        gen.create_liquid(scene);gen.create_floating_objects(scene,rng);gen.create_camera(scene);gen.create_lighting(scene,rng)
        backend=gen.configure_render(scene,settings)
        bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
        labels=gen.render_geometric_labels(scene);sensor=gen.simulate_raw_depth(scene,labels)
        gen.write_sample_arrays(folder,labels,sensor)
        metadata=gen.scene_metadata(scene,folder);metadata['render_backend']=backend;metadata['holdout']='v12_new_seed_not_new_renderer'
        (folder/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(json.dumps({'completed':i+1,'total':64,'scenario':scenario,'band':[low,high]}),flush=True)
    count=gen.build_manifest(args.output_root,args.output_root/'manifest.csv')
    print(json.dumps({'manifest':str(args.output_root/'manifest.csv'),'count':count}),flush=True)


if __name__=='__main__':main()
