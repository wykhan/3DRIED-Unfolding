#!/usr/bin/env python3
"""Full-size FMCW BP assessment; synthetic IF, migrated sampling, bounded GPU run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from three_dried.baseline.config import load_config,coordinates,ranges_m,support_slices
from three_dried.baseline.sampling import nested_masks,mask_record
from three_dried.baseline.runtime_environment import capture_runtime_environment
from three_dried.baseline.validation_metrics import point_localization,sidelobes_single_point
from three_dried.imaging.fmcw_bp import synthetic_if,range_profiles,reference_bp,FusedFMCWBP,C0


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def timed(call,device,warmup,repeats):
    for _ in range(warmup):call()
    torch.cuda.synchronize(device)
    seconds=[]
    for _ in range(repeats):
        torch.cuda.synchronize(device);start=time.perf_counter();output=call()
        torch.cuda.synchronize(device);seconds.append(time.perf_counter()-start)
    return output,dict(median_seconds=float(np.median(seconds)),samples_seconds=seconds,
                       warmup=warmup,repetitions=repeats)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/fmcw_bp_benchmark.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('new output required')
    args.output.mkdir(parents=True,exist_ok=True);out=args.output.resolve()
    settings=yaml.safe_load(args.config.read_text());cfg=load_config(ROOT/settings['acquisition_config'])
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    (out/'settings_used.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    (out/'acquisition_used.yaml').write_text(yaml.safe_dump({k:v for k,v in cfg.items() if not k.startswith('_')},sort_keys=False))
    device=settings['timing']['device'];torch.cuda.set_device(torch.device(device));torch.set_num_threads(1)
    environment=capture_runtime_environment(device)
    environment['precision_note']='double: geometry/phase/interpolation/accumulation FP64; mixed: geometry and sincos FP32, interpolation/accumulation FP64, complex128 storage'
    write(out/'environment.json',environment)
    sources=['src/three_dried/imaging/fmcw_bp.py','src/three_dried/imaging/fmcw_bp_kernel.cu',
             'src/three_dried/baseline/runtime_environment.py',
             'scripts/benchmark_fmcw_bp.py','configs/fmcw_bp_benchmark.yaml']
    write(out/'provenance.json',dict(parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}))
    x,z=coordinates(cfg);zs,xs=support_slices(cfg);ax,az=np.meshgrid(x[xs],z[zs]);px,pz=np.meshgrid(x,z)
    aperture=np.column_stack([ax.ravel(),az.ravel()]);pixels=np.column_stack([px.ravel(),pz.ravel()])
    depth=float(ranges_m(cfg)[cfg['imaging']['matlab_range_bins'].index(settings['reference_depth_matlab_bin'])])
    radar=cfg['radar'];fs=radar['sample_rate_hz'];slope=radar['chirp_slope_hz_s'];delay=radar['instrument_delay_s']
    n_adc=cfg['data']['adc_samples'];nfft=n_adc*settings['range_fft_oversampling']
    f0=settings['adc_first_sample_frequency_hz'];fref=f0+slope*(n_adc-1)/(2*fs)
    masks=nested_masks(ax.shape,settings['sampling']['rates'],settings['sampling']['seed'],settings['sampling']['pattern'])
    write(out/'physical_contract.json',dict(physical_aperture_shape=list(ax.shape),image_shape=list(px.shape),
        adc_samples=n_adc,n_fft=nfft,depth_m=depth,first_adc_frequency_hz=f0,
        effective_center_frequency_hz=fref,accepted_legacy_rma_frequency_hz=radar['center_frequency_hz'],
        one_way_range_bin_m=C0*fs/(2*slope*nfft),physical_range_resolution_m=C0/(2*slope*n_adc/fs),
        full_contributions=int(len(pixels)*len(aperture)),distance_matrix_float32_gib=len(pixels)*len(aperture)*4/2**30,
        cached_objects='range-compressed profiles and coordinate vectors; no pixel-by-aperture distance/phase matrix',
        kernel_timing_excludes='compilation, coordinate setup, range preprocessing, profile compaction and file IO',
        host_image_timing_includes='BP and output download; input profiles already on GPU',
        pipeline_timing_includes='raw ADC upload, padded range FFT, centering, BP and output download; no disk IO or synthetic echo generation',
        real_hammer_status='NOT_RUN_UNVERIFIED_MIXER_AND_ADC_PHASE_CONVENTION'))
    rows=[];checks=[];images={};truths={};setup_rows=[]
    warm=settings['timing']['warmup'];repeats=settings['timing']['repetitions']
    for rate,mask in masks.items():
        write(out/('mask_%s.json'%rate),mask_record(mask,rate,settings['sampling']['pattern'],settings['sampling']['seed']))
        np.save(out/('mask_%s.npy'%rate),mask)
    for scene,points in settings['scenes'].items():
        placed=[];truth=np.zeros(px.shape,complex);spec=[]
        for point in points:
            ix=int(np.argmin(abs(x-point['x_m'])));iz=int(np.argmin(abs(z-point['z_m'])))
            rho=point['amplitude']*np.exp(1j*point['phase_rad']);truth[iz,ix]+=rho
            spec.append((x[ix],depth,z[iz],rho));placed.append(dict(point,ix=ix,iz=iz,grid_x_m=float(x[ix]),grid_z_m=float(z[iz])))
        write(out/(scene+'_targets.json'),placed);truths[scene]=truth
        start=time.perf_counter();raw=synthetic_if(aperture,spec,f0,slope,fs,n_adc,delay)
        print(scene,'synthetic IF seconds',time.perf_counter()-start,flush=True)
        raw_gpu=torch.as_tensor(raw,device=device)
        torch.cuda.synchronize(device);start=time.perf_counter()
        full_profiles=range_profiles(raw_gpu,fs,slope,nfft)
        torch.cuda.synchronize(device)
        setup_rows.append(dict(scene=scene,range_preprocessing_once_seconds=time.perf_counter()-start))
        del raw_gpu
        for rate,mask in masks.items():
            selected=np.flatnonzero(mask.ravel());a=aperture[selected]
            profiles=full_profiles[:,torch.as_tensor(selected,device=device)].contiguous()
            op=FusedFMCWBP(a,pixels,depth,fs,slope,fref,nfft,delay,out/'build',device)
            print(scene,rate,'start',len(selected),'aperture positions',flush=True)
            double,timing=timed(lambda:op(profiles),device,warm,repeats)
            image=double.cpu().numpy().reshape(px.shape);images['%s_%s'%(scene,rate)]=image
            rows.append(dict(scene=scene,rate=rate,observations=len(selected),precision='double',
                scope='device_BP_only',**timing));print(scene,rate,'double',timing['median_seconds'],flush=True)
            # Full-size CPU direct-sum verification at targets and two corners.
            spots=[0,len(pixels)-1]+[p['iz']*len(x)+p['ix'] for p in placed]
            selected_profiles_cpu=profiles.cpu().numpy()
            reference=reference_bp(selected_profiles_cpu,a,pixels[spots],depth,fs,slope,fref,nfft,delay)
            del selected_profiles_cpu
            error=float(np.linalg.norm(image.ravel()[spots]-reference)/np.linalg.norm(reference))
            if error>1e-10:raise AssertionError('CUDA/NumPy migrated sum mismatch')
            check=dict(scene=scene,rate=rate,cpu_spot_relative_error=error)
            check.update(point_localization(image[None],truth[None],cfg['aperture']['dx_m'],cfg['aperture']['dz_m']))
            check.update(sidelobes_single_point(image[None],truth[None]))
            if scene=='single':
                iz,ix=placed[0]['iz'],placed[0]['ix'];rho=truth[iz,ix]
                check['coherent_peak_relative_error']=float(abs(image[iz,ix]/len(selected)-rho)/abs(rho))
                if check['coherent_peak_relative_error']>.02:raise AssertionError('FMCW point coherent phase/gain error')
            if 'mixed' in settings['timing']['precisions']:
                mixed,t=timed(lambda:op(profiles,'mixed'),device,warm,repeats)
                mixed_image=mixed.cpu().numpy().reshape(px.shape)
                relative=float(np.linalg.norm(mixed_image-image)/np.linalg.norm(image))
                check['mixed_relative_error']=relative
                check['mixed_eligible']=relative<=settings['timing']['mixed_relative_error_limit']
                check['mixed_localization']=point_localization(mixed_image[None],truth[None],cfg['aperture']['dx_m'],cfg['aperture']['dz_m'])
                check['mixed_sidelobes']=sidelobes_single_point(mixed_image[None],truth[None])
                rows.append(dict(scene=scene,rate=rate,observations=len(selected),precision='mixed',scope='device_BP_only',**t))
                print(scene,rate,'mixed',t['median_seconds'],'relative',relative,flush=True)
            checks.append(check)
            if scene=='single' and rate==1.0:
                for precision in settings['timing']['precisions']:
                    _,t=timed(lambda:op(profiles,precision).cpu().numpy(),device,warm,repeats)
                    rows.append(dict(scene=scene,rate=rate,observations=len(selected),precision=precision,scope='BP_and_image_download',**t))
                    def pipeline():
                        uploaded=torch.as_tensor(raw,device=device)
                        processed=range_profiles(uploaded,fs,slope,nfft)
                        return op(processed,precision).cpu().numpy()
                    _,t=timed(pipeline,device,warm,repeats)
                    rows.append(dict(scene=scene,rate=rate,observations=len(selected),precision=precision,scope='ADC_upload_range_FFT_BP_image_download',**t))
                    print('full pipeline',precision,t['median_seconds'],flush=True)
            write(out/'timings.json',rows);write(out/'checks.json',checks)
            del profiles,double
        del full_profiles,raw
    np.savez_compressed(out/'images.npz',x_m=x,z_m=z,**images)
    np.savez_compressed(out/'truths.npz',**truths)
    write(out/'setup.json',setup_rows)
    with (out/'timings.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(17,10));fig.subplots_adjust(hspace=.35,wspace=.25)
    peak=max(float(np.abs(v).max()) for v in images.values())
    for i,scene in enumerate(truths):
        for j,rate in enumerate(masks):
            db=20*np.log10(np.maximum(np.abs(images['%s_%s'%(scene,rate)])/peak,1e-12))
            im=axes[i,j].imshow(db,origin='lower',extent=[x[0]*1e3,x[-1]*1e3,z[0]*1e3,z[-1]*1e3],
                vmin=-40,vmax=0,aspect='equal',cmap='viridis')
            axes[i,j].set_title('%s / %.0f%%'%(scene,100*rate));axes[i,j].set_xlabel('x (mm)');axes[i,j].set_ylabel('z (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='dB; shared absolute reference')
    fig.savefig(out/'point_comparison.png',dpi=150);plt.close(fig)
    write(out/'summary.json',dict(status='PASS' if all(c['localization_max_m']==0 for c in checks) else 'FAIL_LOCALIZATION',
        conditions=len(checks),all_mixed_eligible=all(c.get('mixed_eligible',False) for c in checks),
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        scope='full-size physical FMCW simulation and migrated BP; Hammer phase/calibration NOT validated'))


if __name__=='__main__':main()
