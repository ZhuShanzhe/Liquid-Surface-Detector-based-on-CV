"""Measurement illustration of native finite-container IR crops."""
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_finite_v30 import OUT,DATA


def main():
    cs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for i,condition in enumerate(('transparent','translucent_mild')):
        for j,width in enumerate((.4,.8,1.6)):
            c=next(c for c in cs if c['condition']==condition and c['camera_to_bottom_m']==9.6
                and c['container_width_m']==width and c['container_max_height_m']==.12)
            cap=c['tests'][5];im=cv2.imread(str(Path(cap['folder'])/'left.png'),0)
            mask=cv2.imread(str(Path(c['folder'])/'roi_left.png'),0)
            axes[i,j].imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            axes[i,j].contour(mask>0,levels=[.5],colors=['tab:orange'],linewidths=.8)
            axes[i,j].set_title(f"{condition}\nD={width:.1f} m, bottom width={c['empty_width_px']:.2f} px")
            axes[i,j].set_xlabel('Native image x (px)');axes[i,j].set_ylabel('Native image y (px)')
    fig.suptitle('B=9.6 m, capacity=0.12 m, liquid height=0.0432 m\nOrange: fixed empty-bottom ROI; nearest-neighbor display only',fontsize=12)
    fig.savefig(OUT/'finite-bottom-native-pixels.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
