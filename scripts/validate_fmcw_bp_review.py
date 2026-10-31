#!/usr/bin/env python3
"""Off-grid/depth/oversampling evidence using independent raw ADC matching."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from three_dried.baseline.config import load_config
from three_dried.baseline.runtime_environment import capture_runtime_environment
from three_dried.imaging.fmcw_bp import synthetic_if,range_profiles,reference_bp,FusedFMCWBP
from three_dried.simulation.fmcw_adc_reference import direct_adc_match


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/fmcw_bp_review.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('new output required')
    args.output.mkdir(parents=True,exist_ok=True);out=args.output.resolve()
    settings=yaml.safe_load(args.config.read_text());cfg=load_config(ROOT/settings['acquisition_config'])
    device=settings['device'];cuda=torch.cuda.is_available()
    if cuda:torch.cuda.set_device(torch.device(device))
    torch.set_num_threads(1)
    write(out/'environment.json',capture_runtime_environment(device if cuda else 'cpu'))
    (out/'command.txt').write_text(shlex.join([sys.executable]+sys.argv)+'\n')
    (out/'settings_used.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    (out/'acquisition_used.yaml').write_text(yaml.safe_dump({k:v for k,v in cfg.items() if not k.startswith('_')},sort_keys=False))
    sources=['src/three_dried/imaging/fmcw_bp.py','src/three_dried/imaging/fmcw_bp_kernel.cu',
             'src/three_dried/simulation/fmcw_adc_reference.py','src/three_dried/baseline/runtime_environment.py',
             'scripts/validate_fmcw_bp_review.py','tests/test_fmcw_bp.py','configs/fmcw_bp_review.yaml']
    write(out/'provenance.json',dict(reviewed_commit=settings['reviewed_commit'],
        parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}))
    dx=cfg['aperture']['dx_m'];dz=cfg['aperture']['dz_m']
    aperture=np.array([(x*dx,z*dz) for z in settings['aperture_z_indices'] for x in settings['aperture_x_indices']])
    lattice=np.array([(x*dx,z*dz) for z in settings['query_z_indices'] for x in settings['query_x_indices']])
    points=[(p['x_m'],p['y_m'],p['z_m'],complex(p['rho_real'],p['rho_imag'])) for p in settings['points']]
    fs=cfg['radar']['sample_rate_hz'];s=cfg['radar']['chirp_slope_hz_s'];delay=cfg['radar']['instrument_delay_s']
    n=cfg['data']['adc_samples'];f0=settings['first_adc_frequency_hz'];fref=f0+s*(n-1)/(2*fs)
    rows=[];complex_values={};limits=settings['limits']
    for scene in ('single','multidepth'):
        for depth in settings['depths_m']:
            selected=[(.00137,depth,-.00243,.7+.3j)] if scene=='single' else points
            raw=synthetic_if(aperture,selected,f0,s,fs,n,delay)
            # Include a regular query grid and a genuinely off-grid query.
            query_sets={'lattice':lattice,'offgrid':np.array([[.00137,-.00243],[-.02317,.01141],[.03723,-.03183]])}
            for query_kind,pixels in query_sets.items():
                xyz=np.column_stack((pixels[:,0],np.full(len(pixels),depth),pixels[:,1]))
                exact=direct_adc_match(raw,aperture,xyz,fs,s,f0,delay)
                scale=len(aperture)*sum(abs(p[3]) for p in selected)
                for factor in settings['oversampling']:
                    nfft=n*factor;profile=range_profiles(torch.as_tensor(raw),fs,s,nfft).numpy()
                    interpolated=reference_bp(profile,aperture,pixels,depth,fs,s,fref,nfft,delay)
                    bound=np.pi**2*(1-1/n**2)/(24*factor**2)
                    interp=float(np.max(np.abs(interpolated-exact))/scale)
                    if interp>bound+1e-11:raise AssertionError('interpolation curvature bound exceeded')
                    significant=np.abs(exact)>=scale*limits['phase_reference_minimum_fraction_of_known_signal_bound']
                    row=dict(scene=scene,query_kind=query_kind,depth_m=depth,oversampling=factor,
                        interpolation_max_complex_error_over_known_signal_bound=interp,
                        interpolation_curvature_bound=bound,
                        interpolation_relative_l2=float(np.linalg.norm(interpolated-exact)/np.linalg.norm(exact)),
                        amplitude_max_absolute_error_over_known_signal_bound=float(np.max(np.abs(np.abs(interpolated)-np.abs(exact)))/scale),
                        phase_reference_count=int(significant.sum()),
                        phase_max_error_rad=float(np.max(np.abs(np.angle(interpolated[significant]*np.conj(exact[significant]))))) if significant.any() else None,
                        gain_fit=False,phase_fit=False,cuda_status='PASS' if cuda else 'SKIPPED_NO_CUDA',
                        cached_status=('PASS' if cuda else 'SKIPPED_NO_CUDA') if query_kind=='lattice'
                                      else 'SKIPPED_OFF_LATTICE_QUERY_GEOMETRY')
                    key='%s_%s_%s_%s'%(scene,query_kind,depth,factor)
                    complex_values[key+'_adc']=exact;complex_values[key+'_cpu_interpolated']=interpolated
                    if cuda:
                        gpu_profile=torch.tensor(profile,device=device)
                        operator=FusedFMCWBP(aperture,pixels,depth,fs,s,fref,nfft,delay,out/'build',device)
                        modes=['double','mixed']
                        if query_kind=='lattice':operator.prepare_lattice_cache(dx,dz);modes+=['cached_double']
                        for mode in modes:
                            actual=operator(gpu_profile,mode).cpu().numpy()
                            error=float(np.linalg.norm(actual-interpolated)/np.linalg.norm(interpolated))
                            row[mode+'_implementation_relative']=error
                            row[mode+'_total_relative_to_adc']=float(np.linalg.norm(actual-exact)/np.linalg.norm(exact))
                            tolerance=limits['mixed_implementation_relative' if mode=='mixed' else 'double_implementation_relative']
                            if error>tolerance:raise AssertionError(mode+' implementation error')
                            complex_values[key+'_'+mode]=actual
                    rows.append(row)
    np.savez_compressed(out/'complex_values.npz',**complex_values)
    write(out/'errors.json',rows)
    def maximum(column):
        values=[r[column] for r in rows if column in r]
        return max(values) if values else None
    write(out/'summary.json',dict(status='PASS',cases=len(rows),cpu_cases=len(rows),
        cuda_cases=len(rows) if cuda else 0,skipped_cuda_cases=0 if cuda else len(rows),
        cached_cases=sum(r['cached_status']=='PASS' for r in rows),
        cached_skipped_off_lattice_cases=sum(r['query_kind']=='offgrid' for r in rows),
        maximum_double_implementation_relative=maximum('double_implementation_relative'),
        maximum_cached_implementation_relative=maximum('cached_double_implementation_relative'),
        maximum_mixed_implementation_relative=maximum('mixed_implementation_relative'),
        scope='declared positive-beat simulation; independent ADC reference; Hammer not accepted'))
    print(json.dumps(json.loads((out/'summary.json').read_text()),indent=2))


if __name__=='__main__':main()
