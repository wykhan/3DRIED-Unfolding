#!/usr/bin/env python3
"""Held-out deterministic two-point controls; no threshold/model tuning."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import yaml
from scipy.signal import find_peaks
from three_dried.baseline.config import load_config,coordinates,ranges_m
from three_dried.baseline.operators import SliceRM
from three_dried.imaging.sar_operator import classical_ista
from three_dried.models.ista_net import ISTANet
from three_dried.simulation.ista_net_data import make_operators


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-index',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('new output required')
    args.output.mkdir(parents=True,exist_ok=True)
    settings=yaml.safe_load((ROOT/'configs/ista_net_baseline.yaml').read_text());cfg=load_config(ROOT/settings['acquisition_config'])
    index=json.loads(args.checkpoint_index.read_text());device=settings['device'];torch.set_num_threads(4)
    results=[];profiles={};n=cfg['imaging']['n_fft_space'];iz=255;center=255
    (args.output/'command.txt').write_text(' '.join(sys.argv)+'\n')
    (args.output/'config_used.yaml').write_text(yaml.safe_dump(settings,sort_keys=False))
    for rate_key,path in index.items():
        rate=float(rate_key);op=make_operators(cfg,rate,'height_lines',device)[17]
        state=torch.load(path,map_location=device);m=state['config']['model']
        model=ISTANet(**{k:m[k] for k in ('phases','features','theta_init','initialization')}).to(device).eval();model.load_state_dict(state['model_state_dict'])
        for sep in [2,4,8,16]:
            target=torch.zeros(1,n,n,dtype=torch.complex64,device=device)
            left=center-sep//2;right=center+sep//2;target[0,iz,left]=1;target[0,iz,right]=1
            with torch.no_grad():
                measurement=op(target)
                physical=(measurement[0]/op.gain).cpu().numpy()
                estimates={'RMA':SliceRM(cfg,ranges_m(cfg)[5]).inverse(physical),
                           'ISTA':classical_ista(measurement,op,settings['evaluation']['ista_iterations'],settings['evaluation']['ista_lambda_relative'])[0][0].cpu().numpy(),
                           'ISTA-Net':model(measurement,op)[0].cpu().numpy()}
            for method,image in estimates.items():
                profile=np.abs(image[iz]);peaks=find_peaks(profile)[0]
                chosen=[]
                for truth_index in (left,right):
                    near=peaks[np.abs(peaks-truth_index)<=1]
                    chosen.append(int(near[np.argmax(profile[near])]) if len(near) else None)
                distinct=all(v is not None for v in chosen) and chosen[0]!=chosen[1]
                valley=None
                if distinct:
                    a,b=sorted(chosen);denominator=min(profile[a],profile[b])
                    valley=float(profile[a:b+1].min()/denominator) if denominator>0 else None
                resolved=distinct and valley is not None and valley<=10**(-3/20)
                results.append({'rate':rate,'method':method,'separation_x_m':sep*cfg['aperture']['dx_m'],
                                'resolved_under_fixed_criterion':bool(resolved),'valley_to_weaker_peak':valley,
                                'criterion':'two distinct local peaks within one grid cell of GT and >=3 dB amplitude valley',
                                'scope':'two equal-phase, equal-amplitude noiseless points at bin17; fixed control, not a universal resolution limit'})
                profiles['%s_%s_%s'%(rate,sep,method)]=profile
    (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(args.output/'profiles.npz',**profiles)
    print('completed',len(results),'resolution controls')


if __name__=='__main__':main()
