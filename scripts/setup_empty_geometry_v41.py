import argparse,json
from pathlib import Path
from liquid_depth.empty_geometry_runtime_v41 import create_empty_geometry_profile,EmptyGeometryLiquidDepthSystem


def main():
    p=argparse.ArgumentParser(description='Empty RGB-D plus sparse measured parameters to ordinary liquid metrology')
    p.add_argument('--frame',required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--profile',required=True);p.add_argument('--depth-scale-to-m',type=float,required=True)
    p.add_argument('--measure-frame');p.add_argument('--output-dir')
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8'))
    profile,bundle=create_empty_geometry_profile(dict(frame_dir=a.frame,geometry=spec,
        profile_path=a.profile,depth_scale_to_m=a.depth_scale_to_m))
    print(json.dumps(dict(profile_path=str(profile),geometry=bundle),ensure_ascii=False,indent=2))
    if a.measure_frame:
        meter=EmptyGeometryLiquidDepthSystem(profile)
        print(json.dumps(meter.measure(a.measure_frame,a.output_dir),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
