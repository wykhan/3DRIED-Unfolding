#!/usr/bin/env python3
"""Benchmark exact regular-lattice FMCW caches against saved full-size FP64 BP."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from benchmark_fmcw_bp import timed,write
from three_dried.baseline.config import load_config,coordinates,ranges_m,support_slices
from three_dried.baseline.sampling import nested_masks
from three_dried.baseline.runtime_environment import capture_runtime_environment
from three_dried.imaging.fmcw_bp import synthetic_if,range_profiles,FusedFMCWBP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--historical-reference',type=Path,
                        help='also regress against a previous compatible full-size run; preserve its provenance')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('new output required')
    args.output.mkdir(parents=True,exist_ok=True);out=args.output.resolve()
    settings=yaml.safe_load((args.reference/'settings_used.yaml').read_text())
    if args.historical_reference:
        historical_settings=yaml.safe_load((args.historical_reference/'settings_used.yaml').read_text())
        if historical_settings!=settings:raise ValueError('historical reference settings differ')
    cfg=load_config(ROOT/settings['acquisition_config'])
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    (out/'settings_used.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    sources=['src/three_dried/imaging/fmcw_bp.py','src/three_dried/imaging/fmcw_bp_kernel.cu',
             'src/three_dried/baseline/runtime_environment.py',
             'scripts/benchmark_fmcw_bp.py',
             'scripts/benchmark_fmcw_bp_cache.py']
    write(out/'provenance.json',dict(sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        reference_directory=str(args.reference.resolve()),reference_images_sha256=hashlib.sha256((args.reference/'images.npz').read_bytes()).hexdigest()))
    write(out/'reference_provenance.json',dict(
        environment_sha256=hashlib.sha256((args.reference/'environment.json').read_bytes()).hexdigest(),
        source_provenance=json.loads((args.reference/'provenance.json').read_text()),
        historical_reference_directory=str(args.historical_reference.resolve()) if args.historical_reference else None,
        historical_images_sha256=hashlib.sha256((args.historical_reference/'images.npz').read_bytes()).hexdigest() if args.historical_reference else None,
        historical_source_provenance=json.loads((args.historical_reference/'provenance.json').read_text()) if args.historical_reference else None))
    x,z=coordinates(cfg);zs,xs=support_slices(cfg);ax,az=np.meshgrid(x[xs],z[zs]);px,pz=np.meshgrid(x,z)
    aperture=np.column_stack([ax.ravel(),az.ravel()]);pixels=np.column_stack([px.ravel(),pz.ravel()])
    depth=float(ranges_m(cfg)[cfg['imaging']['matlab_range_bins'].index(settings['reference_depth_matlab_bin'])])
    fs=cfg['radar']['sample_rate_hz'];slope=cfg['radar']['chirp_slope_hz_s'];delay=cfg['radar']['instrument_delay_s']
    n=cfg['data']['adc_samples'];nf=n*settings['range_fft_oversampling'];f0=settings['adc_first_sample_frequency_hz']
    fref=f0+slope*(n-1)/(2*fs);device=settings['timing']['device']
    torch.cuda.set_device(torch.device(device));torch.set_num_threads(1)
    write(out/'environment.json',capture_runtime_environment(device))
    masks=nested_masks(ax.shape,settings['sampling']['rates'],settings['sampling']['seed'],settings['sampling']['pattern'])
    rows=[];checks=[];setup=[];warm=settings['timing']['warmup'];repeats=settings['timing']['repetitions']
    with np.load(args.reference/'images.npz') as reference:
        for scene in settings['scenes']:
            placed=json.loads((args.reference/(scene+'_targets.json')).read_text())
            points=[(p['grid_x_m'],depth,p['grid_z_m'],p['amplitude']*np.exp(1j*p['phase_rad'])) for p in placed]
            raw=synthetic_if(aperture,points,f0,slope,fs,n,delay)
            full=range_profiles(torch.as_tensor(raw,device=device),fs,slope,nf)
            for rate,mask in masks.items():
                chosen=np.flatnonzero(mask.ravel());profiles=full[:,torch.as_tensor(chosen,device=device)].contiguous()
                op=FusedFMCWBP(aperture[chosen],pixels,depth,fs,slope,fref,nf,delay,out/'build',device)
                torch.cuda.synchronize(device);start=time.perf_counter()
                size=op.prepare_lattice_cache(cfg['aperture']['dx_m'],cfg['aperture']['dz_m'])
                torch.cuda.synchronize(device)
                setup.append(dict(scene=scene,rate=rate,cache_bytes=size,cache_setup_seconds=time.perf_counter()-start))
                result,timing=timed(lambda:op(profiles,'cached_double'),device,warm,repeats)
                image=result.cpu().numpy().reshape(px.shape);expected=reference['%s_%s'%(scene,rate)]
                error=float(np.linalg.norm(image-expected)/np.linalg.norm(expected))
                if error>1e-10:raise AssertionError('cached/full migrated BP mismatch')
                check=dict(scene=scene,rate=rate,relative_to_full_double=error)
                if args.historical_reference:
                    with np.load(args.historical_reference/'images.npz') as historical:
                        previous=historical['%s_%s'%(scene,rate)]
                    history_error=float(np.linalg.norm(image-previous)/np.linalg.norm(previous))
                    if history_error>1e-10:raise AssertionError('historical full BP regression mismatch')
                    check['relative_to_historical_double']=history_error
                checks.append(check)
                rows.append(dict(scene=scene,rate=rate,precision='cached_double',scope='device_BP_only',**timing))
                print(scene,rate,timing['median_seconds'],'relative',error,flush=True)
                if scene=='single' and rate==1.0:
                    _,timing=timed(lambda:op(profiles,'cached_double').cpu().numpy(),device,warm,repeats)
                    rows.append(dict(scene=scene,rate=rate,precision='cached_double',scope='BP_and_image_download',**timing))
                    def pipeline():
                        return op(range_profiles(torch.as_tensor(raw,device=device),fs,slope,nf),'cached_double').cpu().numpy()
                    _,timing=timed(pipeline,device,warm,repeats)
                    rows.append(dict(scene=scene,rate=rate,precision='cached_double',scope='ADC_upload_range_FFT_BP_image_download',**timing))
                    print('cached full pipeline',timing['median_seconds'],flush=True)
                write(out/'timings.json',rows);write(out/'checks.json',checks);write(out/'setup.json',setup)
                del profiles,result,op
            del full,raw
    with (out/'timings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'summary.json',dict(status='PASS',conditions=len(checks),
        maximum_complex_relative_error=max(c['relative_to_full_double'] for c in checks),
        maximum_historical_relative_error=max(c['relative_to_historical_double'] for c in checks) if args.historical_reference else None,
        scope='exact regular-lattice cache; full FMCW distance queries retained'))


if __name__=='__main__':main()
