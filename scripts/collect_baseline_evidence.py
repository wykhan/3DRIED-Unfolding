#!/usr/bin/env python3
"""Build the lightweight review bundle; never copies model weights or volumes."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/review/ISTA_NET'


def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={'1.0':'results/ista_net_height_1.0','0.7':'results/ista_net_height_0.7','0.5':'results/ista_net_h50_v2','0.3':'results/ista_net_height_0.3'}
    training={};index={}
    for rate,folder in paths.items():
        source=ROOT/folder
        summary=json.loads((source/'summary.json').read_text())
        summary['checkpoint_relative_path']=folder+'/best.pt'
        summary['checkpoint_exists_locally']=(source/'best.pt').exists()
        if hashlib.sha256((source/'best.pt').read_bytes()).hexdigest()!=summary['checkpoint_sha256']:
            raise ValueError('checkpoint provenance mismatch')
        training[rate]=summary;index[rate]=folder+'/best.pt'
        shutil.copyfile(source/'history.csv',OUT/('training_history_'+rate+'.csv'))
        shutil.copyfile(source/'config_used.yaml',OUT/('training_config_'+rate+'.yaml'))
        shutil.copyfile(source/'split_manifest.json',OUT/('split_manifest_'+rate+'.json'))
        shutil.copyfile(source/'implementation_hashes.json',OUT/('training_source_hashes_'+rate+'.json'))
    write(OUT/'training_summary.json',training);write(OUT/'checkpoint_index.json',index)
    comparisons={};all_rows=[]
    for name in ['comparison_single_height','comparison_point_sim_height','comparison_hammer_height','comparison_point_sim_random2d','comparison_hammer_random2d']:
        folder=ROOT/'results'/name
        summary=json.loads((folder/'summary.json').read_text())
        comparisons[name]={'relative_path':'results/'+name,'status_counts':{s:sum(row['status']==s for row in summary['rows']) for s in set(row['status'] for row in summary['rows'])}}
        all_rows.extend([dict(run=name,**row) for row in summary['rows']])
        shutil.copyfile(folder/'metrics.csv',OUT/(name+'_metrics.csv'))
        shutil.copyfile(folder/'environment.json',OUT/(name+'_environment.json'))
    keys=list(dict.fromkeys(k for row in all_rows for k in row))
    with (OUT/'all_metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(all_rows)
    write(OUT/'comparison_index.json',comparisons)
    shutil.copyfile(ROOT/'results/ista_validation/tests_final.log',OUT/'tests.txt')
    for name,folder in [('basic_bp_controls','basic_bp_controls'),('resolution','point_resolution')]:
        shutil.copyfile(ROOT/'results'/folder/'summary.json',OUT/(name+'.json'))
    for folder in ['basic_bp_controls','point_resolution','ista_budget_diagnostic']:
        shared=json.loads((ROOT/'results/comparison_hammer_height/environment.json').read_text())
        shared['record_note']='shared execution environment, copied after execution; not an additional timing run'
        write(ROOT/'results'/folder/'shared_environment.json',shared)
        shutil.copyfile(ROOT/'configs/baseline_3dried.yaml',ROOT/'results'/folder/'acquisition_config.yaml')
    write(OUT/'implementation_hashes.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('src/three_dried/**/*.py'))})
    write(OUT/'run_provenance.json',{'local_commit_before_bundle':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'training_weights':'local ignored results, not Git','large_arrays':'local ignored results, not Git',
          'new_mask_rmist':'SKIPPED; frozen M6 mask condition differs','hammer_ground_truth':False,
          'BP_full_FOV':'SKIPPED; literal-loop budget and unresolved raw-phase/range convention; no accelerated replacement',
          'primary_pattern':'height_lines','extension_pattern':'random_2d',
          'expected_test_suite':'63 passed; full CUDA and real raw-data availability',
          'full_matrix_status':'technically reproducible partial matrix; independent scientific review pending'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for rate,folder in paths.items():
        with (ROOT/folder/'history.csv').open() as handle:rows=list(csv.DictReader(handle))
        axes[0].semilogy([int(r['epoch']) for r in rows],[float(r['train_loss']) for r in rows],label=rate)
        axes[1].semilogy([int(r['epoch']) for r in rows],[float(r['validation_complex_nmse']) for r in rows],label=rate)
    axes[0].set_ylabel('normalized reconstruction + symmetry loss');axes[1].set_ylabel('validation Complex NMSE')
    for ax in axes:ax.set_xlabel('epoch');ax.legend(title='height sampling rate');ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(ROOT/'results/ista_validation/training_curves.png',dpi=150);plt.close(fig)
    import numpy as np
    bp=np.load(ROOT/'results/basic_bp_controls/images.npz')
    fig,axes=plt.subplots(2,4,figsize=(16,8));fig.subplots_adjust(hspace=.5,wspace=.4)
    peak=max(float(np.abs(bp[name]).max()) for name in bp.files if name not in ['x_m','z_m'])
    for row,scene in enumerate(['single','two_points']):
        for column,rate in enumerate([1.,.7,.5,.3]):
            db=20*np.log10(np.maximum(np.abs(bp['%s_%s'%(scene,rate)])/peak,1e-12))
            im=axes[row,column].imshow(db,origin='lower',vmin=-40,vmax=0,extent=[bp['x_m'][0]*1000,bp['x_m'][-1]*1000,bp['z_m'][0]*1000,bp['z_m'][-1]*1000],cmap='viridis')
            axes[row,column].set_title('%s / %.0f%%'%(scene,rate*100));axes[row,column].set_xlabel('x (mm)');axes[row,column].set_ylabel('z (mm)')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='dB, one common absolute reference')
    fig.savefig(ROOT/'results/basic_bp_controls/point_controls.png',dpi=140);plt.close(fig)
    print('review bundle',OUT,'rows',len(all_rows))


if __name__=='__main__':main()
