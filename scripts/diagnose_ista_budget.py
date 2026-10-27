#!/usr/bin/env python3
"""Fixed ISTA iteration-budget diagnostic on validation scenes only."""
import csv
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import yaml
from three_dried.baseline.config import load_config
from three_dried.imaging.sar_operator import complex_shrink
from three_dried.simulation.ista_net_data import make_operators,PointScenes


def main():
    out=ROOT/'results/ista_budget_diagnostic'
    if out.exists():raise FileExistsError('diagnostic already exists')
    out.mkdir()
    cfg=yaml.safe_load((ROOT/'configs/ista_net_baseline.yaml').read_text())
    acquisition=load_config(ROOT/cfg['acquisition_config']);torch.set_num_threads(4)
    dataset=PointScenes(acquisition,cfg['dataset'],'validation');rows=[]
    started=time.monotonic()
    with torch.no_grad():
        for rate in [1.,.7,.5,.3]:
            operators=make_operators(acquisition,rate,'height_lines',cfg['device'])
            y,gt,op,fingerprint=dataset.pair(0,operators);ahy=op.adjoint(y)
            image=ahy.clone();step=op.lipschitz.reciprocal();lam=cfg['evaluation']['ista_lambda_relative']*ahy.abs().max()
            for iteration in range(1,20001):
                image=complex_shrink(image-step*op.adjoint(op(image)-y),step*lam)
                if iteration%100==0:
                    objective=float(.5*(op(image)-y).abs().square().sum()+lam*image.abs().sum())
                    nmse=float((image-gt).abs().square().sum()/gt.abs().square().sum())
                    rows.append({'rate':rate,'iteration':iteration,'objective':objective,'complex_nmse':nmse,'validation_scene_sha256':fingerprint})
                if time.monotonic()-started>600:raise TimeoutError('ten-minute diagnostic cap')
            print(rate,rows[-1],flush=True)
    with (out/'curve.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'scope':'first validation scene only; no parameter search; no test/real tuning','fixed_budgets':[200,2000,20000],'checkpoints':[r for r in rows if r['iteration'] in [200,2000,20000]],'seconds':time.monotonic()-started,'termination':'fixed_budget; not a convergence certificate'},indent=2)+'\n')
    (out/'command.txt').write_text(' '.join(sys.argv)+'\n')


if __name__=='__main__':main()
