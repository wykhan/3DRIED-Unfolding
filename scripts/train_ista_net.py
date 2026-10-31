#!/usr/bin/env python3
"""Bounded original ISTA-Net training; test split opened after selection only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from three_dried.baseline.config import load_config
from three_dried.models.ista_net import ISTANet, ista_net_loss
from three_dried.simulation.ista_net_data import PointScenes, make_operators
from three_dried.imaging.sar_operator import classical_ista


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def evaluate(model, dataset, operators, ista_settings=None):
    rows = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            measurement, target, op, fingerprint = dataset.pair(index, operators)
            prediction = model(measurement, op)
            energy = target.abs().square().sum()
            row = {'index': index, 'scene_sha256': fingerprint,
                   'complex_nmse': float((prediction-target).abs().square().sum()/energy),
                   'observed_relative_residual': float(torch.linalg.vector_norm(op(prediction)-measurement)/torch.linalg.vector_norm(measurement))}
            if ista_settings is not None:
                ista, history = classical_ista(measurement, op, ista_settings['ista_iterations'], ista_settings['ista_lambda_relative'])
                row['ista_complex_nmse'] = float((ista-target).abs().square().sum()/energy)
                row['ista_initial_objective'] = history[0]
                row['ista_final_objective'] = history[-1]
            if not all(np.isfinite(v) for k, v in row.items() if isinstance(v, float)):
                raise FloatingPointError('nonfinite evaluation')
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ista_net_baseline.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rate', type=float)
    parser.add_argument('--pattern', choices=['height_lines', 'random_2d'])
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.rate is not None: cfg['sampling']['rate'] = args.rate
    if args.pattern is not None: cfg['sampling']['pattern'] = args.pattern
    if args.epochs is not None: cfg['training']['epochs'] = args.epochs
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('choose a new output directory; training artifacts are immutable')
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output.resolve()
    if cfg['training']['batch_size'] != 1:
        raise ValueError('this initial implementation explicitly uses batch size one')
    if cfg['training']['max_gpu_hours'] > 6:
        raise ValueError('experiment above six GPU hours is not authorized')
    acquisition = load_config(ROOT/cfg['acquisition_config'])
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); torch.set_num_threads(4)
    if not torch.cuda.is_available() and str(cfg['device']).startswith('cuda'):
        raise RuntimeError('GPU unavailable; do not silently change timed hardware')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = ISTANet(**{k:cfg['model'][k] for k in ('phases','features','theta_init','initialization')}).to(cfg['device'])
    operators = make_operators(acquisition, cfg['sampling']['rate'], cfg['sampling']['pattern'], cfg['device'])
    datasets = {name: PointScenes(acquisition, cfg['dataset'], name) for name in ('train', 'validation', 'test')}
    manifests = {name:[dataset.scene(i)[2] for i in range(len(dataset))] for name,dataset in datasets.items()}
    if any(set(manifests[a]) & set(manifests[b]) for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('scene leakage across splits')
    (out/'config_used.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out/'acquisition_used.yaml').write_text(yaml.safe_dump({k:v for k,v in acquisition.items() if not k.startswith('_')}, sort_keys=False))
    (out/'command.txt').write_text(' '.join(sys.argv)+'\n')
    write_json(out/'split_manifest.json', manifests)
    write_json(out/'environment.json', {'python':sys.version,'executable':sys.executable,'conda':os.getenv('CONDA_DEFAULT_ENV'), 'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':cfg['device'],'gpu':torch.cuda.get_device_name(torch.device(cfg['device'])),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    write_json(out/'implementation_hashes.json', {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/three_dried/models/ista_net.py',ROOT/'src/three_dried/imaging/sar_operator.py',ROOT/'src/three_dried/simulation/ista_net_data.py',Path(__file__).resolve()]})
    initial = evaluate(model, datasets['validation'], operators)
    untrained_nmse = float(np.mean([r['complex_nmse'] for r in initial]))
    write_json(out/'untrained_validation.json',initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    rng = np.random.default_rng(cfg['training']['shuffle_seed'])
    started = time.monotonic(); best = float('inf'); best_epoch = None
    with (out/'history.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['epoch','train_loss','validation_complex_nmse','elapsed_seconds']);writer.writeheader()
        for epoch in range(1,cfg['training']['epochs']+1):
            model.train(); losses=[]
            for index in rng.permutation(len(datasets['train'])):
                measurement,target,op,_ = datasets['train'].pair(int(index),operators)
                optimizer.zero_grad(set_to_none=True)
                prediction,symmetry=model(measurement,op,return_symmetry=True)
                loss=ista_net_loss(prediction,target,symmetry,cfg['training']['symmetry_weight'])
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                loss.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('missing/nonfinite gradient')
                optimizer.step();losses.append(float(loss.detach()))
                if time.monotonic()-started>cfg['training']['max_gpu_hours']*3600:
                    raise TimeoutError('training budget exceeded; incomplete model cannot enter rankings')
            validation=evaluate(model,datasets['validation'],operators)
            nmse=float(np.mean([r['complex_nmse'] for r in validation]))
            elapsed=time.monotonic()-started
            writer.writerow(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_complex_nmse=nmse,elapsed_seconds=elapsed));handle.flush()
            if nmse < best:
                best,best_epoch=nmse,epoch
                torch.save({'model_state_dict':model.state_dict(),'config':cfg,'epoch':epoch,'validation_complex_nmse':nmse},out/'best.pt')
            print('epoch',epoch,'train',float(np.mean(losses)),'val NMSE',nmse,'seconds',elapsed,flush=True)
    state=torch.load(out/'best.pt',map_location=cfg['device'])
    model.load_state_dict(state['model_state_dict'])
    test=evaluate(model,datasets['test'],operators,cfg['evaluation'])
    write_json(out/'independent_test.json',test)
    result={'status':'PASS' if best<untrained_nmse else 'FAIL_NO_VALIDATION_IMPROVEMENT','completed_epochs':cfg['training']['epochs'],'best_epoch':best_epoch,'untrained_validation_nmse':untrained_nmse,'best_validation_nmse':best,'test_complex_nmse':float(np.mean([r['complex_nmse'] for r in test])),'ista_test_complex_nmse':float(np.mean([r['ista_complex_nmse'] for r in test])),'training_seconds':time.monotonic()-started,'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),'rate':cfg['sampling']['rate'],'pattern':cfg['sampling']['pattern'],'scope':'RM-generated noisy points; no real-data tuning; no BP/FMCW validation claim'}
    write_json(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
