#!/usr/bin/env python3
"""Audit current task coverage and run analytical stratified-liquid controls."""
import argparse,ast,csv,hashlib,json,math,subprocess
from collections import Counter
from pathlib import Path
from liquid_depth.stratified_diagnostics import fresnel_unpolarized,slab_geometry,match_interfaces

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    manifest=Path('/root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v3-calibration-top-12000/manifest_multilayer_v4_refined.csv')
    rows=list(csv.DictReader(manifest.open()))
    tree=ast.parse((root/'src/liquid_depth/simulation.py').read_text())
    scene=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SyntheticScene')
    fields=[n.target.id for n in scene.body if isinstance(n,ast.AnnAssign)]
    audit=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),manifest_rows=len(rows),scenario_counts=dict(Counter(r['scenario'] for r in rows)),manifest_columns=list(rows[0]),liquid_fields=[s for s in fields if 'liquid' in s],task='Two or more immiscible transparent liquids, overhead camera',current_label_events=['wall intersections','one free liquid surface','container bottom'],stratified_accuracy_available=False)
    (a.output/'source-audit.json').write_text(json.dumps(audit,indent=2))
    fresnel=[]
    for n1,n2,label in [(1.,1.333,'air to example lower liquid'),(1.,1.47,'air to example upper liquid'),(1.47,1.333,'example larger liquid contrast'),(1.37,1.333,'example moderate contrast'),(1.343,1.333,'example small contrast'),(1.333,1.333,'index matched control')]:
        fresnel.append(dict(label=label,n1=n1,n2=n2,normal_reflectance=fresnel_unpolarized(n1,n2)))
    (a.output/'fresnel.json').write_text(json.dumps(fresnel,indent=2))
    paths=[]
    for degree in (0,15,30,45,60):
        r=slab_geometry([.02,.08],[1.47,1.333],math.radians(degree))
        paths.append(dict(air_angle_deg=degree,**r,lateral_error_if_straight_m=r['straight_air_displacement_m']-r['lateral_displacement_m']))
    (a.output/'ray-paths.json').write_text(json.dumps(paths,indent=2))
    equal_a=slab_geometry([.02,.08],[1.4,1.4],math.radians(30))
    equal_b=slab_geometry([.06,.04],[1.4,1.4],math.radians(30))
    metric=[]
    for z in (.2,.5,1.,3.):
        for gap_mm in (1,2,5,10,20):
            metric.append(dict(front_depth_m=z,gap_mm=gap_mm,**match_interfaces([z+gap_mm/2000],[z,z+gap_mm/1000])))
    result=dict(
        scope='Analytical sanity controls only; no RGB render, sensor emulator, network inference or accuracy benchmark',
        illustrative_parameters='Refractive indices are simulation values; not universal oil properties or deployment inputs',
        equal_index_control=dict(first=equal_a,second=equal_b,internal_reflectance=0,interface_shift_mm=40,identical_bottom_ray_displacement=True,identical_paraxial_apparent_depth=True,assumptions='lossless planar media, matched refractive index at modeled wavelength, no absorption/scattering/meniscus differences'),
        upper_interface_apparent_depth=dict(true_thickness_mm=20,paraxial_apparent_mm=20/1.47,naive_air_inversion_error_mm=20/1.47-20,note='Paraxial visual apparent depth, NOT ToF depth'),
        thin_layer_regularizer_conflict=dict(camera_depth_m=.5,physical_layer_gap_mm=2,existing_minimum_separation_mm=max(3,.01*.5*1000),penalty_positive_even_for_correct_gap=True),
        thickness_error_example=dict(true_top_m=.5,true_interface_m=.502,predicted_top_m=.505,predicted_interface_m=.497,each_surface_error_mm=5,true_thickness_mm=2,predicted_signed_thickness_mm=-8),
        collapsed_predictions=metric)
    (a.output/'analytical-controls.json').write_text(json.dumps(result,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    labels=['Air / 1.47','1.47 / 1.333','1.37 / 1.333','1.343 / 1.333']
    values=[100*fresnel[i]['normal_reflectance'] for i in (1,2,3,4)]
    bars=axes[0].bar(labels,values,color='#276ca6');axes[0].set_yscale('log');axes[0].bar_label(bars,fmt='%.4f',padding=3)
    axes[0].set_ylabel('Normal-incidence reflected power (%)');axes[0].tick_params(axis='x',rotation=20)
    axes[1].plot([x['air_angle_deg'] for x in paths],[1000*x['lateral_error_if_straight_m'] for x in paths],marker='o',color='#cf5d28')
    axes[1].set_xlabel('Viewing ray angle in air (degrees)');axes[1].set_ylabel('Bottom-ray lateral shift missed by straight ray (mm)')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Stratified-liquid analytical controls | NOT camera accuracy')
    fig.tight_layout();fig.savefig(a.output/'stratified-controls.png',dpi=160)
    print(json.dumps(dict(audit=audit,fresnel=fresnel,paths=paths,controls={k:v for k,v in result.items() if k!='collapsed_predictions'}),indent=2))
if __name__=='__main__':main()
