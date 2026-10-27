#!/usr/bin/env python3
"""Unified five-method entry; missing/invalid conditions are explicit SKIPPED."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from three_dried.baseline.config import load_config, coordinates, ranges_m, support_slices
from three_dried.baseline.datasets import point_scene, real_echo
from three_dried.baseline.operators import SliceRM
from three_dried.baseline.methods import rm_iteration, load_rmist_parameters
from three_dried.baseline.sampling import nested_masks, mask_record
from three_dried.baseline.validation_metrics import complex_nmse, point_localization, sidelobes_single_point, timed_reconstruction
from three_dried.imaging.basic_bp import direct_slice_bp
from three_dried.imaging.fft_bp import FFTSliceBP, TorchFFTSliceBP
from three_dried.imaging.sar_operator import classical_ista
from three_dried.models.ista_net import ISTANet
from three_dried.simulation.ista_net_data import make_operators


METHODS=['BP','RMA','ISTA','ISTA-Net','RMIST-Net']
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def figures(folder,volumes,cfg,peak):
    if not volumes:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x,z=coordinates(cfg);roi=cfg['visualization']['crop_size_mm']/2000
    use_x=np.abs(x)<roi;use_z=np.abs(z)<roi
    bins=cfg['imaging']['matlab_range_bins'];representative=bins.index(17) if 17 in bins else 0
    for kind in ['mip_xz','representative_slice']:
        fig,axes=plt.subplots(1,len(volumes),figsize=(5*len(volumes),5),squeeze=False)
        for ax,(method,volume) in zip(axes[0],volumes.items()):
            value=np.abs(volume).max(axis=0) if kind=='mip_xz' else np.abs(volume[representative])
            db=20*np.log10(np.maximum(value/peak,1e-12))
            im=ax.imshow(db[np.ix_(use_z,use_x)],origin='lower',extent=[x[use_x][0]*1000,x[use_x][-1]*1000,z[use_z][0]*1000,z[use_z][-1]*1000],vmin=-40,vmax=0,cmap='viridis',aspect='equal')
            ax.set_title(method);ax.set_xlabel('x (mm)');ax.set_ylabel('z (mm)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='dB; shared absolute peak')
        fig.savefig(folder/(kind+'.png'),dpi=140);plt.close(fig)
    method=next(iter(volumes));volume=volumes[method]
    magnitude=np.abs(volume);locations=np.argwhere(magnitude>=peak*10**(-18/20))
    if len(locations):
        locations=locations[::max(1,len(locations)//12000)]
        fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
        ax.scatter(x[locations[:,2]]*1000,ranges_m(cfg)[locations[:,0]]*1000,z[locations[:,1]]*1000,s=1)
        ax.set_xlabel('x (mm)');ax.set_ylabel('range y (mm)');ax.set_zlabel('z (mm)');ax.set_title(method+'; -18 dB common threshold')
        fig.savefig(folder/'volume_3d.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm',nargs='+',choices=METHODS,default=METHODS)
    p.add_argument('--dataset',choices=['point_sim','hammer'],required=True)
    p.add_argument('--scene',choices=['single','multi'],default='multi')
    p.add_argument('--rates',nargs='+',type=float,default=[1.,.7,.5,.3])
    p.add_argument('--pattern',choices=['height_lines','random_2d'],default='height_lines')
    p.add_argument('--config',type=Path,default=ROOT/'configs/ista_net_baseline.yaml')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--checkpoint-index',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bp-seconds',type=float,default=5.)
    p.add_argument('--bp-backend',choices=['fft_cpu','fft_cuda','literal'],default='fft_cpu',
                   help='exact phase-only slice sum; FFT acceleration authorized 2026-10-27')
    p.add_argument('--bp-workers',type=int,default=1)
    p.add_argument('--rmist-diagnostic',action='store_true')
    p.add_argument('--no-figures',action='store_true')
    p.add_argument('--warmup',type=int)
    p.add_argument('--repetitions',type=int)
    p.add_argument('--timing-note',default='isolated run; check saved hardware record')
    args=p.parse_args();settings=yaml.safe_load(args.config.read_text())
    settings['bp_execution']={'backend':args.bp_backend,'cpu_fft_workers':args.bp_workers,
                              'model':'fixed_depth_phase_only_spherical_sum','dtype':'complex128'}
    if args.warmup is not None:settings['evaluation']['warmup']=args.warmup
    if args.repetitions is not None:settings['evaluation']['repetitions']=args.repetitions
    if settings['evaluation']['warmup']<0 or settings['evaluation']['repetitions']<1:raise ValueError('invalid timing counts')
    cfg=load_config(ROOT/settings['acquisition_config'])
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('new output directory required')
    args.output.mkdir(parents=True,exist_ok=True);out=args.output.resolve()
    (out/'command.txt').write_text(' '.join(sys.argv)+'\n')
    (out/'settings_used.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    write(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'platform':platform.platform(),'cpu_model':cpu_model,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'timing_excludes':'loading, range FFT, operator/model construction, file I/O, plotting'})
    if args.scene=='single':
        cfg=copy.deepcopy(cfg);cfg['simulation']['points']=[dict(matlab_bin=17,x_m=-.05,z_m=.12,amplitude=1.,phase_rad=.4)]
    (out/'acquisition_used.yaml').write_text(yaml.safe_dump({k:v for k,v in cfg.items() if not k.startswith('_')},sort_keys=False))
    echo,gt,_,meta=point_scene(cfg) if args.dataset=='point_sim' else real_echo(cfg)
    write(out/'input_metadata.json',meta)
    full_reference=np.stack([SliceRM(cfg,y).inverse(SliceRM(cfg,y).pad(echo[i])) for i,y in enumerate(ranges_m(cfg))])
    common_peak=float(np.abs(gt).max()) if gt is not None else float(np.abs(full_reference).max())
    write(out/'display_scale.json',{'peak':common_peak,'source':'known point GT peak' if gt is not None else 'RMA100 peak','db_limits':[-40,0],'shared_across_all_rates_and_methods':True})
    del full_reference
    np.savez_compressed(out/'coordinates.npz',x_m=coordinates(cfg)[0],z_m=coordinates(cfg)[1],range_m=ranges_m(cfg))
    if gt is not None:np.savez_compressed(out/'ground_truth.npz',volume=gt)
    device=settings['device'];torch.set_num_threads(4);rows=[]
    index=json.loads(args.checkpoint_index.read_text()) if args.checkpoint_index else {}
    physical_masks=nested_masks((cfg['data']['nz'],cfg['data']['nx']),args.rates,cfg['sampling']['seed'],args.pattern)
    for rate in args.rates:
        folder=out/('sr%03d'%round(rate*100));folder.mkdir()
        physical=physical_masks[rate];record=mask_record(physical,rate,args.pattern,cfg['sampling']['seed'])
        write(folder/'mask.json',record);np.save(folder/'mask.npy',physical)
        ops=make_operators(cfg,rate,args.pattern,device)
        numpy_ops={b:SliceRM(cfg,y) for b,y in zip(cfg['imaging']['matlab_range_bins'],ranges_m(cfg))}
        physical_measurements={b:numpy_ops[b].pad(echo[i])*ops[b].mask.cpu().numpy() for i,b in enumerate(cfg['imaging']['matlab_range_bins'])}
        measurement={b:ops[b].encode_physical_measurement(torch.as_tensor(SliceRM(cfg,y).pad(echo[i]),dtype=torch.complex64,device=device)).unsqueeze(0)
                     for i,(b,y) in enumerate(zip(cfg['imaging']['matlab_range_bins'],ranges_m(cfg)))}
        volumes={}
        for method in args.algorithm:
            row={'algorithm':method,'dataset':args.dataset,'scene':args.scene if gt is not None else None,'rate':rate,'pattern':args.pattern,'mask_sha256':record['sha256_uint8'],'status':'SKIPPED','reason':None,'timing_note':args.timing_note}
            checkpoint=args.checkpoint or (Path(index[str(rate)]) if str(rate) in index else None)
            net=None
            if method=='ISTA-Net':
                if checkpoint is None or not checkpoint.is_file():
                    row['reason']='NO_TRAINED_CHECKPOINT';rows.append(row);continue
                summary_path=checkpoint.parent/'summary.json'
                if not summary_path.exists() or json.loads(summary_path.read_text())['status']!='PASS':
                    row['reason']='CHECKPOINT_NOT_VALIDATED';rows.append(row);continue
                state=torch.load(checkpoint,map_location=device)
                trained=state['config']
                acquisition_path=checkpoint.parent/'acquisition_used.yaml'
                if not acquisition_path.exists():
                    row['reason']='MISSING_TRAINING_ACQUISITION_PROVENANCE';rows.append(row);continue
                training_acquisition=yaml.safe_load(acquisition_path.read_text())
                if any(training_acquisition[group]!=cfg[group] for group in ['data','radar','aperture','imaging']):
                    row['reason']='TRAINING_PHYSICAL_CONFIGURATION_MISMATCH';rows.append(row);continue
                if rate!=1.0 and training_acquisition['sampling']['seed']!=cfg['sampling']['seed']:
                    row['reason']='TRAINING_MASK_REALIZATION_MISMATCH';rows.append(row);continue
                source_path=checkpoint.parent/'implementation_hashes.json'
                if not source_path.exists():
                    row['reason']='MISSING_TRAINING_SOURCE_PROVENANCE';rows.append(row);continue
                source_hashes=json.loads(source_path.read_text())
                core_paths=['src/three_dried/models/ista_net.py','src/three_dried/imaging/sar_operator.py']
                if any(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=source_hashes.get(path) for path in core_paths):
                    row['reason']='TRAINED_MODEL_OR_OPERATOR_SOURCE_CHANGED';rows.append(row);continue
                if trained['sampling']['rate']!=rate or (rate!=1.0 and trained['sampling']['pattern']!=args.pattern):
                    row['reason']='CHECKPOINT_RATE_OR_PATTERN_MISMATCH';rows.append(row);continue
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=json.loads(summary_path.read_text())['checkpoint_sha256']:
                    raise ValueError('checkpoint hash differs from validation evidence')
                net=ISTANet(**{k:trained['model'][k] for k in ('phases','features','theta_init','initialization')}).to(device).eval();net.load_state_dict(state['model_state_dict'])
                row['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                row['generalization']='sim-to-real; no Hammer tuning' if gt is None else 'independent fixed point control'
            if method=='RMIST-Net' and not args.rmist_diagnostic:
                row['reason']='ACCEPTED_WEIGHTS_NOT_VALIDATED_ON_THIS_MASK';rows.append(row);continue
            params=load_rmist_parameters(ROOT/'results/review/M6/summary_metrics.json') if method=='RMIST-Net' else None
            bp_ops={}
            if method=='BP' and args.bp_backend!='literal':
                for b,y in zip(cfg['imaging']['matlab_range_bins'],ranges_m(cfg)):
                    bp_args=((cfg['imaging']['n_fft_space'],)*2,cfg['aperture']['dx_m'],
                             cfg['aperture']['dz_m'],float(y),cfg['radar']['center_frequency_hz'])
                    bp_ops[b]=(FFTSliceBP(*bp_args,workers=args.bp_workers) if args.bp_backend=='fft_cpu'
                               else TorchFFTSliceBP(*bp_args,device=device))
                row['bp_backend']=args.bp_backend
                row['bp_kernel_setup_included']=False
            def reconstruct():
                images=[]
                with torch.no_grad():
                    for i,(b,y) in enumerate(zip(cfg['imaging']['matlab_range_bins'],ranges_m(cfg))):
                        op=ops[b];m=measurement[b]
                        if method=='ISTA-Net':image=net(m,op)[0].cpu().numpy()
                        elif method=='ISTA':image=classical_ista(m,op,settings['evaluation']['ista_iterations'],settings['evaluation']['ista_lambda_relative'])[0][0].cpu().numpy()
                        elif method=='RMA':image=numpy_ops[b].inverse(physical_measurements[b])
                        elif method=='RMIST-Net':image=rm_iteration(physical_measurements[b],numpy_ops[b],*params)
                        elif method=='BP' and args.bp_backend!='literal':
                            image=bp_ops[b](physical_measurements[b])
                            if isinstance(image,torch.Tensor):image=image.cpu().numpy()
                        else:
                            x,z=coordinates(cfg);zs,xs=support_slices(cfg)
                            image=direct_slice_bp(echo[i],x[xs],z[zs],x,z,float(y),cfg['radar']['center_frequency_hz'],physical,args.bp_seconds)
                        images.append(image)
                return np.stack(images)
            try:
                if method=='BP' and args.bp_backend=='literal':
                    image=reconstruct();timing={'runtime_note':'literal BP; CPU; no acceleration; no completed repeated benchmark'}
                else:
                    timing_device=device if method in ['ISTA','ISTA-Net'] or (method=='BP' and args.bp_backend=='fft_cuda') else 'cpu'
                    image,timing=timed_reconstruction(reconstruct,timing_device,settings['evaluation']['warmup'],settings['evaluation']['repetitions'])
            except TimeoutError as exc:
                row['reason']='LITERAL_BP_BUDGET_EXCEEDED';row['detail']=str(exc);rows.append(row);continue
            if not np.isfinite(image).all():raise FloatingPointError('nonfinite reconstruction')
            row.update(status='PASS',**timing)
            if method in ['BP','RMIST-Net']:
                row['status']='DIAGNOSTIC';row['reason']='CROSS_MODEL_PHASE_ONLY_GAIN' if method=='BP' else 'UNVALIDATED_MASK_AND_REAL_SCALE'
            if gt is not None and method!='BP':
                row['complex_nmse']=complex_nmse(image,gt)
                row.update(point_localization(image,gt,cfg['aperture']['dx_m'],cfg['aperture']['dz_m']))
                row.update(sidelobes_single_point(image,gt))
            residual_sq=0.;measurement_sq=0.
            for i,b in enumerate(cfg['imaging']['matlab_range_bins']):
                t=torch.as_tensor(image[i],dtype=torch.complex64,device=device).unsqueeze(0)
                residual_sq+=float((ops[b](t)-measurement[b]).abs().square().sum())
                measurement_sq+=float(measurement[b].abs().square().sum())
            row['observed_relative_residual']=float(np.sqrt(residual_sq/measurement_sq)) if measurement_sq else None
            if gt is None:
                mag=np.abs(image).astype('float64');power=mag**2;powers=power.ravel()/max(power.sum(),1e-300);nonzero=powers>0
                row['image_energy']=float(power.sum());row['energy_entropy']=float(-(powers[nonzero]*np.log(powers[nonzero])).sum())
                row['quality_note']='energy/entropy are descriptive; no real GT or superiority claim'
            volumes[method]=image.astype('complex64');np.savez_compressed(folder/(method.replace('-','_')+'.npz'),volume=volumes[method])
            rows.append(row)
        if not args.no_figures:figures(folder,volumes,cfg,common_peak)
    write(out/'summary.json',{'rows':rows,'status_meaning':'PASS = completed finite computation with declared input/model checks; independent scientific review remains required','limitations':['RM point data are model matched; not raw FMCW validation','literal BP cannot be ranked from RM-generated points without model/gain validation','RMIST new-mask outputs are unvalidated; no M9 normalization used']})
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'statuses':{status:sum(r['status']==status for r in rows) for status in set(r['status'] for r in rows)}},indent=2))


if __name__=='__main__':main()
