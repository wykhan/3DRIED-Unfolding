#!/usr/bin/env python3
"""Run five baselines on 3DRIED-matched synthetic and real acquisitions."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from three_dried.baseline.config import load_config,validate_config
from three_dried.baseline.runner import run_suite

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/baseline_3dried.yaml')
    parser.add_argument('--datasets',nargs='+',choices=['point_sim','real'],default=['point_sim','real'])
    parser.add_argument('--dataset-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--no-figures',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='small numerical check only; not full acquisition validation')
    args=parser.parse_args(); cfg=load_config(args.config)
    if args.smoke:
        cfg=copy.deepcopy(cfg);cfg['profile']='reduced_smoke_not_3dried_acquisition'
        cfg['data'].update(nx=13,nz=12)
        cfg['aperture'].update(horizontal_span_m=0.012,vertical_span_m=0.022)
        cfg['imaging'].update(n_fft_space=16,x_offset_mm=0,z_offset_mm=0,
            matlab_range_bins=[17],representative_matlab_bin=17)
        cfg['simulation']['points']=[dict(matlab_bin=17,x_m=0,z_m=0,amplitude=1.0,phase_rad=0.3)]
        cfg['algorithms']['cv_ista']['iterations']=20
        validate_config(cfg)
    out=args.output_dir or ROOT/cfg['output_dir']
    try:
        summary=run_suite(cfg,ROOT,args.datasets,out,args.dataset_dir,not args.no_figures)
    except (ValueError,FileExistsError) as exc:
        parser.error(str(exc))
    print(summary['status'],out)
    return 2 if summary['status']=='INCOMPLETE' else 0

if __name__=='__main__':raise SystemExit(main())
