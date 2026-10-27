#!/usr/bin/env python3
"""Full-aperture phase-only point control for literal BP; not raw FMCW."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from three_dried.baseline.config import load_config,coordinates,support_slices,ranges_m
from three_dried.baseline.sampling import nested_masks
from three_dried.imaging.basic_bp import direct_slice_bp


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('new output required')
    args.output.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/'configs/baseline_3dried.yaml');x,z=coordinates(cfg);zs,xs=support_slices(cfg)
    ax,az=np.meshgrid(x[xs],z[zs]);depth=float(ranges_m(cfg)[5]);phase=4*math.pi*cfg['radar']['center_frequency_hz']/299792458.
    center_x,center_z=float(x[255]),float(z[255]);images={};rows=[]
    candidates_x=center_x+np.arange(-4,5)*cfg['aperture']['dx_m']
    candidates_z=center_z+np.arange(-4,5)*cfg['aperture']['dz_m']
    masks=nested_masks(ax.shape,[1.,.7,.5,.3],cfg['sampling']['seed'],'height_lines')
    for scene,points in [('single',[(center_x,center_z)]),('two_points',[(center_x-.003,center_z),(center_x+.003,center_z)])]:
        echo=np.zeros(ax.shape,complex)
        for px,pz in points:
            echo+=np.exp(1j*phase*np.sqrt((ax-px)**2+depth**2+(az-pz)**2))
        for rate,mask in masks.items():
            started=time.monotonic()
            image=direct_slice_bp(echo,x[xs],z[zs],candidates_x,candidates_z,depth,cfg['radar']['center_frequency_hz'],mask,max_seconds=120)
            elapsed=time.monotonic()-started
            where=np.unravel_index(np.argmax(np.abs(image)),image.shape)
            row={'scene':scene,'rate':rate,'observations':int(mask.sum()),'seconds_single_diagnostic':elapsed,
                 'timing_note':'CPU direct loops, no warmup/repetitions; diagnostic cost only',
                 'scope':'single-frequency phase-only analytic forward; full acquired aperture, local 9x9 candidate grid; not full-FOV FMCW'}
            if scene=='single':
                row['location_error_m']=float(np.hypot(candidates_x[where[1]]-center_x,candidates_z[where[0]]-center_z))
                row['coherent_center_relative_error']=float(abs(image[4,4]-mask.sum())/mask.sum())
                row['status']='PASS' if row['location_error_m']==0 and row['coherent_center_relative_error']<1e-10 else 'FAIL'
            else:
                profile=np.abs(image[4]);left=int(np.argmax(profile[:4]));right=5+int(np.argmax(profile[5:]));valley=float(profile[left:right+1].min()/min(profile[left],profile[right]))
                row['two_point_valley_to_weaker_peak']=valley;row['resolved_under_3db_valley']=bool(valley<=10**(-3/20))
                row['status']='COMPLETED_DIAGNOSTIC'
            rows.append(row);images['%s_%s'%(scene,rate)]=image
            print(scene,rate,elapsed,flush=True)
    np.savez_compressed(args.output/'images.npz',x_m=candidates_x,z_m=candidates_z,**images)
    (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    (args.output/'command.txt').write_text(' '.join(sys.argv)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,8));fig.subplots_adjust(hspace=.5,wspace=.4);peak=max(float(np.abs(a).max()) for a in images.values())
    for row,(scene,_) in enumerate([('single',0),('two_points',0)]):
        for col,rate in enumerate(masks):
            db=20*np.log10(np.maximum(np.abs(images['%s_%s'%(scene,rate)])/peak,1e-12))
            im=axes[row,col].imshow(db,origin='lower',vmin=-40,vmax=0,extent=[candidates_x[0]*1000,candidates_x[-1]*1000,candidates_z[0]*1000,candidates_z[-1]*1000],cmap='viridis')
            axes[row,col].set_title('%s / %.0f%%'%(scene,rate*100));axes[row,col].set_xlabel('x (mm)');axes[row,col].set_ylabel('z (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='dB, one common absolute reference')
    fig.savefig(args.output/'point_controls.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
