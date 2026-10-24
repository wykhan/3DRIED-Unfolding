"""Common-scale and per-panel views with physical axes and explicit labels."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .config import coordinates,ranges_m

def db(mag,peak,dynamic_range):
    return np.maximum(20*np.log10(np.maximum(mag,np.finfo(float).tiny)/max(peak,np.finfo(float).tiny)),-dynamic_range)

def render_pattern(out_dir,volumes,masks,cfg,pattern,reference=None):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    methods=cfg['algorithms']['methods']; rates=cfg['sampling']['rates']
    x,z=coordinates(cfg); lim=cfg['visualization']['crop_size_mm']/2000
    zi=np.where(np.abs(z)<=lim)[0]; xi=np.where(np.abs(x)<=lim)[0]
    extent=[x[xi[0]]*1000,x[xi[-1]]*1000,z[zi[0]]*1000,z[zi[-1]]*1000]
    bins=cfg['imaging']['matlab_range_bins']; rep=bins.index(cfg['imaging']['representative_matlab_bin'])
    dr=cfg['visualization']['dynamic_range_db']
    refpeak=float(np.abs(reference).max()) if reference is not None else max(float(np.abs(v).max()) for v in volumes.values())
    for view in ['mip','slice']:
        for scale in ['common_scale','panel_normalized']:
            fig,axes=plt.subplots(len(rates),len(methods),figsize=(3.3*len(methods),3*len(rates)),squeeze=False,layout='constrained')
            for ir,sr in enumerate(rates):
                for ic,method in enumerate(methods):
                    vol=volumes[(float(sr),method)]
                    mag=np.max(np.abs(vol),axis=0) if view=='mip' else np.abs(vol[rep])
                    peak=refpeak if scale=='common_scale' else float(mag.max())
                    ax=axes[ir,ic]
                    im=ax.imshow(db(mag[np.ix_(zi,xi)],peak,dr),origin='lower',extent=extent,vmin=-dr,vmax=0,cmap='turbo',aspect='equal')
                    ax.set_title(f'{method} | SR={sr:.0%}'); ax.set_xlabel('x / mm'); ax.set_ylabel('z / mm')
            title=f'{pattern} | {view} | {scale.replace("_"," ")} | {dr:g} dB'
            if view=='slice': title+=f' | MATLAB bin {bins[rep]}'
            fig.suptitle(title); fig.colorbar(im,ax=axes,label='dB')
            fig.savefig(out_dir/f'{view}_{scale}.png',dpi=120); plt.close(fig)
    fig,axes=plt.subplots(1,len(rates),figsize=(3.5*len(rates),3),squeeze=False,layout='constrained')
    for ax,sr in zip(axes[0],rates):
        ax.imshow(masks[float(sr)],origin='lower',interpolation='nearest',cmap='gray',vmin=0,vmax=1,aspect='auto')
        ax.set_title(f'{sr:.0%}; actual {masks[float(sr)].mean():.4f}'); ax.set_xlabel('x aperture index');ax.set_ylabel('z aperture index')
    fig.suptitle(pattern+' | observed aperture positions');fig.savefig(out_dir/'sampling_masks.png',dpi=140);plt.close(fig)
    # A 3-D point cloud for every method at every SR, with bounded plot size.
    fig=plt.figure(figsize=(3.5*len(methods),3*len(rates)),layout='constrained')
    ys=ranges_m(cfg); threshold=cfg['visualization']['point_cloud_threshold_db']
    for ir,sr in enumerate(rates):
        for ic,method in enumerate(methods):
            ax=fig.add_subplot(len(rates),len(methods),ir*len(methods)+ic+1,projection='3d')
            mag=np.abs(volumes[(float(sr),method)]); peak=float(mag.max())
            ind=np.argwhere((mag>0)&(db(mag,peak,dr)>=threshold))
            if len(ind):
                cap=cfg['visualization']['max_point_cloud_points']
                if len(ind)>cap: ind=ind[np.linspace(0,len(ind)-1,cap,dtype=int)]
                iy,iz,ix=ind.T; ax.scatter(x[ix]*1000,ys[iy]*1000,z[iz]*1000,c=db(mag[iy,iz,ix],peak,dr),vmin=threshold,vmax=0,s=1,cmap='turbo')
            ax.set_title(f'{method} | {sr:.0%}');ax.set_xlabel('x/mm');ax.set_ylabel('y/mm');ax.set_zlabel('z/mm')
    fig.suptitle(f'{pattern} | per-panel {threshold:g} dB point clouds')
    fig.savefig(out_dir/'volume_3d.png',dpi=110);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for method in methods:
        for sr in rates:
            energy=np.sum(np.abs(volumes[(float(sr),method)])**2,axis=(1,2))
            ax.plot(ys*1000,energy,label=f'{method} {sr:.0%}')
    ax.set_xlabel('reference range y/mm');ax.set_ylabel('slice energy (no per-slice rescaling)')
    ax.legend(fontsize=6,ncol=4);fig.savefig(out_dir/'range_energy_profiles.png',dpi=140);plt.close(fig)
