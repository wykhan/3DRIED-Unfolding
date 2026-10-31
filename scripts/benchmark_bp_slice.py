#!/usr/bin/env python3
"""Validated single-frequency BP slice timings and one/five point controls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import scipy
import torch
import yaml
from three_dried.baseline.config import load_config, coordinates, ranges_m, support_slices, C0
from three_dried.baseline.sampling import nested_masks, mask_record
from three_dried.baseline.validation_metrics import point_localization, sidelobes_single_point
from three_dried.imaging.basic_bp import direct_slice_bp
from three_dried.imaging.fft_bp import FFTSliceBP, TorchFFTSliceBP


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def measured(call, warmup, repeats, device=None):
    def sync():
        if device is not None:
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        call()
    sync()
    seconds = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = call()
        sync()
        seconds.append(time.perf_counter()-start)
    return result, dict(median_seconds=float(np.median(seconds)),
                        samples_seconds=seconds, warmup=warmup, repetitions=repeats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/bp_fft_slice.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('new output required')
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output.resolve()
    settings = yaml.safe_load(args.config.read_text())
    cfg = load_config(ROOT/settings['acquisition_config'])
    (out/'settings_used.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
    (out/'acquisition_used.yaml').write_text(yaml.safe_dump(
        {k:v for k,v in cfg.items() if not k.startswith('_')}, sort_keys=False))
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    cpu = next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
               if line.startswith('model name'))
    device = settings['timing']['device'] if torch.cuda.is_available() else None
    if device:
        torch.cuda.set_device(torch.device(device))
        torch.cuda.init()
    write(out/'environment.json', dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        torch=torch.__version__, platform=platform.platform(), cpu_model=cpu,
        gpu=torch.cuda.get_device_name(device) if device else None,
        cuda=torch.version.cuda, fft_norm='backward', dtype='complex128',
        cpu_fft_workers=settings['timing']['cpu_fft_workers'],
        gpu_status=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used',
            '--format=csv,noheader'], capture_output=True, text=True).stdout if device else None))
    paths = ['src/three_dried/imaging/fft_bp.py','src/three_dried/imaging/basic_bp.py',
             'scripts/benchmark_bp_slice.py', 'configs/bp_fft_slice.yaml']
    write(out/'provenance.json', dict(commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}))
    x,z = coordinates(cfg)
    zs,xs = support_slices(cfg)
    ax,az = np.meshgrid(x[xs],z[zs])
    depth = float(ranges_m(cfg)[cfg['imaging']['matlab_range_bins'].index(settings['representative_matlab_bin'])])
    phase = 4*np.pi*cfg['radar']['center_frequency_hz']/C0
    shape = (len(z),len(x))
    op_args = (shape,cfg['aperture']['dx_m'],cfg['aperture']['dz_m'],depth,cfg['radar']['center_frequency_hz'])
    start = time.perf_counter()
    cpu_op = FFTSliceBP(*op_args,workers=settings['timing']['cpu_fft_workers'])
    setup = {'cpu_kernel_setup_seconds':time.perf_counter()-start}
    gpu_op = None
    if device:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        gpu_op = TorchFFTSliceBP(*op_args,device=device)
        torch.cuda.synchronize(device)
        setup['gpu_kernel_setup_seconds_excluding_cuda_context'] = time.perf_counter()-start
    write(out/'setup.json', dict(setup,fft_shape=cpu_op.fft_shape,physical_shape=ax.shape,
        image_shape=shape,depth_m=depth,
        cached_timing_excludes='kernel setup, range preprocessing, file IO and plots',
        cpu_uncached_timing_includes='kernel generation, kernel FFT, echo FFT, multiply, inverse FFT, crop',
        gpu_host_timing_includes='host to device echo transfer, reconstruction, device to host image transfer',
        gpu_resident_timing_includes='device reconstruction only, synchronized wall time'))
    masks = nested_masks(ax.shape,settings['sampling']['rates'],settings['sampling']['seed'],settings['sampling']['pattern'])
    for rate,mask in masks.items():
        write(out/('mask_%s.json'%rate),mask_record(mask,rate,settings['sampling']['pattern'],settings['sampling']['seed']))
        np.save(out/('mask_%s.npy'%rate),mask)
    rows=[]; checks=[]; images={}; targets={}; placed_scenes={}
    warmup=settings['timing']['warmup']; repeats=settings['timing']['repetitions']
    for scene,points in settings['scenes'].items():
        echo=np.zeros(ax.shape,complex);truth=np.zeros(shape,complex);placed=[]
        for point in points:
            ix=int(np.argmin(np.abs(x-point['x_m'])));iz=int(np.argmin(np.abs(z-point['z_m'])))
            rho=point['amplitude']*np.exp(1j*point['phase_rad'])
            truth[iz,ix]+=rho
            echo+=rho*np.exp(1j*phase*np.sqrt((ax-x[ix])**2+depth**2+(az-z[iz])**2))
            placed.append(dict(point,ix=ix,iz=iz,grid_x_m=float(x[ix]),grid_z_m=float(z[iz]),depth_m=depth))
        targets[scene]=truth;placed_scenes[scene]=placed
        for rate,mask in masks.items():
            padded=np.zeros(shape,complex);padded[zs,xs]=echo*mask
            # Full aperture / full output grid, with direct-sum spot checks at
            # the scene points and widely separated edges to expose wrapping.
            expected=cpu_op(padded)
            sample_locations=[(0,0),(shape[0]-1,shape[1]-1)]+[(p['iz'],p['ix']) for p in placed]
            observed=[];reference=[]
            for iz,ix in sample_locations:
                value=direct_slice_bp(echo,x[xs],z[zs],[x[ix]],[z[iz]],depth,
                    cfg['radar']['center_frequency_hz'],mask)[0,0]
                observed.append(expected[iz,ix]);reference.append(value)
            relative=float(np.linalg.norm(np.array(observed)-reference)/np.linalg.norm(reference))
            maximum=float(np.max(np.abs(np.array(observed)-reference))/mask.sum())
            if relative>2e-10 or maximum>2e-10:
                raise AssertionError('full-size FFT/direct sum mismatch')
            check=dict(scene=scene,rate=rate,direct_spot_relative_error=relative,
                       direct_spot_max_error_per_observation=maximum,spots=len(reference))
            check.update(point_localization(expected[None],truth[None],cfg['aperture']['dx_m'],cfg['aperture']['dz_m']))
            check.update(sidelobes_single_point(expected[None],truth[None]))
            checks.append(check);images['%s_%s'%(scene,rate)]=expected
            calls=[('cpu_uncached',lambda: FFTSliceBP(*op_args,workers=settings['timing']['cpu_fft_workers'])(padded),None),
                   ('cpu_cached',lambda: cpu_op(padded),None)]
            if gpu_op:
                resident=torch.as_tensor(padded,device=device)
                calls += [('gpu_cached_with_transfers',lambda: gpu_op(padded).cpu().numpy(),device),
                          ('gpu_cached_resident',lambda: gpu_op(resident),device)]
            for backend,call,sync_device in calls:
                result,timing=measured(call,warmup,repeats,sync_device)
                if isinstance(result,torch.Tensor):result=result.cpu().numpy()
                error=float(np.linalg.norm(result-expected)/np.linalg.norm(expected))
                if error>1e-10:raise AssertionError('backend mismatch')
                rows.append(dict(scene=scene,rate=rate,observations=int(mask.sum()),backend=backend,
                                 relative_to_cpu=error,**timing))
                print(scene,rate,backend,'%.6f s'%timing['median_seconds'],flush=True)
    np.savez_compressed(out/'images.npz',x_m=x,z_m=z,**images)
    np.savez_compressed(out/'targets.npz',**targets)
    write(out/'targets.json',placed_scenes)
    write(out/'checks.json',checks);write(out/'timings.json',rows)
    keys=list(rows[0])
    with (out/'timings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(targets),len(masks),figsize=(17,8),squeeze=False)
    peak=max(float(np.abs(a).max()) for a in images.values())
    for i,scene in enumerate(targets):
        for j,rate in enumerate(masks):
            db=20*np.log10(np.maximum(np.abs(images['%s_%s'%(scene,rate)])/peak,1e-12))
            im=axes[i,j].imshow(db,origin='lower',extent=[x[0]*1e3,x[-1]*1e3,z[0]*1e3,z[-1]*1e3],
                vmin=-40,vmax=0,cmap='viridis',aspect='equal')
            axes[i,j].scatter([p['grid_x_m']*1e3 for p in placed_scenes[scene]],
                [p['grid_z_m']*1e3 for p in placed_scenes[scene]],s=20,marker='+',color='red')
            axes[i,j].set_title('%s / %.0f%%'%(scene,100*rate))
            axes[i,j].set_xlabel('x (mm)');axes[i,j].set_ylabel('z (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='dB; common absolute reference, no gain compensation')
    fig.savefig(out/'point_comparison.png',dpi=160);plt.close(fig)
    write(out/'summary.json',dict(status='PASS',conditions=len(checks),timing_rows=len(rows),
        all_locations_on_grid=all(c.get('localization_max_m')==0 for c in checks),
        scope='analytic single-frequency fixed-depth control; full 3DRIED aperture and image lattice; not raw FMCW/Hammer acceptance'))


if __name__=='__main__':main()
