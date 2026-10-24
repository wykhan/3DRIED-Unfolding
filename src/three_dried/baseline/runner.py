"""Reproducible experiment matrix, explicit missing-data records, and outputs."""
import csv
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from .config import ranges_m,coordinates,support,configuration_fingerprint
from .datasets import point_scene,real_echo,echo_sha256
from .methods import METHODS,reconstruct,load_rmist_parameters
from .operators import SliceRM
from .sampling import nested_masks,mask_record
from .metrics import image_metrics
from .visualization import render_pattern

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def run_suite(cfg,repo_root,datasets,output_dir,dataset_dir=None,make_figures=True):
    root=Path(repo_root); out=Path(output_dir)
    # A directory corresponds to exactly one immutable run; avoid mixing stale outputs.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('output directory is not empty; choose a new --output-dir')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'implementation_hashes.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/three_dried/baseline').glob('*.py'))})
    clean={k:v for k,v in cfg.items() if not k.startswith('_')}
    (out/'config_used.yaml').write_text(yaml.safe_dump(clean,sort_keys=False))
    (out/'command.txt').write_text(' '.join(sys.argv)+'\n')
    write_json(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
        processor=platform.processor(),execution_backend='NumPy/SciPy CPU; no GPU timing claims'))
    param_path=root/cfg['algorithms']['rmist']['parameter_summary']
    params=load_rmist_parameters(param_path)
    write_json(out/'rmist_provenance.json',dict(source=str(param_path.relative_to(root)),
        sha256=hashlib.sha256(param_path.read_bytes()).hexdigest(),epoch=99,
        tau=params[0].tolist(),theta=params[1].tolist(),trainable=False,
        source_kind='exact reconstruction from committed 18 accepted float32 scalars'))
    rows=[]; statuses={}; mask_records=[]
    methods=cfg['algorithms']['methods']
    if not methods or not set(methods)<=set(METHODS): raise ValueError('invalid method selection')
    if not datasets or not set(datasets)<={'point_sim','real'}:raise ValueError('invalid datasets')
    ys=ranges_m(cfg); n=cfg['imaging']['n_fft_space']; d=cfg['data']
    x,z=coordinates(cfg);np.savez(out/'coordinates.npz',x_m=x,z_m=z,y_m=ys)
    for dataset in datasets:
        dest=out/dataset;dest.mkdir()
        try:
            loaded=point_scene(cfg) if dataset=='point_sim' else real_echo(cfg,dataset_dir)
            echo,gt,projected_gt,meta=loaded
        except FileNotFoundError as exc:
            statuses[dataset]=dict(status='BLOCKED_MISSING_DATA',reason=str(exc))
            write_json(dest/'status.json',statuses[dataset]);continue
        meta['range_focused_echo_sha256']=echo_sha256(echo);write_json(dest/'input_metadata.json',meta)
        if gt is not None:np.savez_compressed(dest/'ground_truth.npz',point_gt=gt.astype('complex64'),bandlimited_gt=projected_gt.astype('complex64'))
        reference=np.stack([SliceRM(cfg,y).inverse(SliceRM(cfg,y).pad(echo[i])) for i,y in enumerate(ys)])
        np.savez_compressed(dest/'full_data_rma_reference.npz',volume=reference.astype('complex64'))
        for pattern in cfg['sampling']['patterns']:
            folder=dest/pattern;folder.mkdir();volumes={}
            masks=nested_masks((d['nz'],d['nx']),cfg['sampling']['rates'],cfg['sampling']['seed'],pattern)
            for sr,physical_mask in masks.items():
                rec=mask_record(physical_mask,sr,pattern,cfg['sampling']['seed']);mask_records.append(dict(dataset=dataset,**rec))
                tag=f'sr{int(round(sr*100)):03d}';np.save(folder/(tag+'_mask.npy'),physical_mask)
                op0=SliceRM(cfg,ys[0]); mask=op0.pad(physical_mask)
                estimates={method:np.zeros((len(ys),n,n),dtype=np.complex64) for method in methods}
                timings={method:0.0 for method in methods}; consistency={method:[] for method in methods}
                objectives={method:[] for method in methods}
                for i,y in enumerate(ys):
                    op=SliceRM(cfg,y);measurement=op.pad(echo[i])*mask
                    denominator=np.linalg.norm(measurement)
                    for method in methods:
                        start=time.perf_counter()
                        if denominator:
                            image,diag=reconstruct(method,measurement,op,mask,cfg,params)
                        else:
                            image=np.zeros_like(measurement);diag={}
                        timings[method]+=time.perf_counter()-start
                        if not np.isfinite(image).all():raise FloatingPointError(f'{dataset}/{pattern}/{sr}/{method}: nonfinite image')
                        estimates[method][i]=image.astype('complex64')
                        if denominator:
                            value=float(np.linalg.norm(mask*(op.forward(image)-measurement))/denominator)
                            consistency[method].append(dict(matlab_bin=cfg['imaging']['matlab_range_bins'][i],relative_residual=value))
                        else:
                            consistency[method].append(dict(matlab_bin=cfg['imaging']['matlab_range_bins'][i],relative_residual=None))
                        if diag:objectives[method].append(dict(matlab_bin=cfg['imaging']['matlab_range_bins'][i],**diag))
                for method,image in estimates.items():
                    fname=method.lower().replace('-','_')
                    np.savez_compressed(folder/(tag+'_'+fname+'.npz'),volume=image)
                    volumes[(float(sr),method)]=image
                    vals=[v['relative_residual'] for v in consistency[method] if v['relative_residual'] is not None]
                    metrics=image_metrics(image,gt,projected_gt,reference)
                    rows.append(dict(dataset=dataset,pattern=pattern,method=method,requested_sr=sr,actual_sr=rec['actual_rate'],
                        observed_positions=rec['observed_positions'],reconstruction_seconds=timings[method],
                        mean_observed_rm_residual=float(np.mean(vals)) if vals else None,
                        mask_sha256=rec['sha256_uint8'],**metrics))
                    write_json(folder/(tag+'_'+fname+'_diagnostics.json'),dict(
                        observed_rm_residual_by_slice=consistency[method],solver=objectives[method],
                        consistency_model='masked RM slice forward; cross-model diagnostic for geometric BP',
                        rm_ista_residual_policy='historical zero-filled full-domain residual',
                        rmist_generalization='new mask realization at every SR; height masks and SR!=0.5 are OOD'))
                print(f'{dataset}: {pattern} SR={sr:.0%}: {len(methods)} methods complete',flush=True)
            if make_figures:render_pattern(folder,volumes,masks,cfg,pattern,reference)
            del volumes
        statuses[dataset]=dict(status='COMPLETED',matrix_rows=sum(r['dataset']==dataset for r in rows))
        write_json(dest/'status.json',statuses[dataset])
    headers=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=headers or ['dataset','status'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(out/'sampling_manifest.json',mask_records)
    blocked=any(x['status']!='COMPLETED' for x in statuses.values())
    summary=dict(status='INCOMPLETE' if blocked else 'COMPLETED',datasets=statuses,rows=len(rows),
        profile=cfg.get('profile'),configuration_fingerprint=configuration_fingerprint(cfg),
        limitations=['RM-generated point data are model-matched, not a full FMCW simulator.',
          'BP is per-range narrowband spherical matched-filter BP; cross-model comparison with RM.',
          'CV-ISTA definition is classical complex L1 with the true masked adjoint, pending user confirmation.',
          'Accepted ISTA/RMIST preserve zero-filled full-domain residuals.',
          'RMIST weights are fixed; all masks are new, with additional cross-SR/height OOD.',
          'Real RMA reference is a proxy, never voxel GT. CPU runtime excludes loading/plotting.'])
    write_json(out/'run_summary.json',summary)
    lines=['# Baseline experiment report','',f"Status: **{summary['status']}**",'',
        '| Dataset | State | Experiment rows |','|---|---|---|']
    lines += [f"| {name} | {v['status']} | {v.get('matrix_rows',0)} |" for name,v in statuses.items()]
    lines += ['',f"Configuration fingerprint: `{summary['configuration_fingerprint']}`",'',
        '## Interpretation','']+['- '+v for v in summary['limitations']]
    lines += ['','## Output index','','- `summary_metrics.csv`: metrics and CPU timings.',
        '- `sampling_manifest.json`: exact counts and mask hashes.',
        '- `<dataset>/<pattern>/srNNN_<method>.npz`: complex [range,z,x] volume.',
        '- Comparison figures: MIP, representative slice, 3-D point clouds, masks and range energies.',
        '- Each method/SR diagnostics file records per-slice RM residual and CV-ISTA objectives.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
