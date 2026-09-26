#!/usr/bin/env python3
"""M0 environment and external-dataset readability smoke test."""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy
import scipy
import torch
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/home/superws/dataset/3DRIED/Hammer/Hammer (1)"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mat_path = args.dataset / "output_rawdata.mat"
    with h5py.File(mat_path, "r") as handle:
        root_keys = sorted(handle.keys())
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "matplotlib": matplotlib.__version__,
        "pyyaml": yaml.__version__,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "dataset_readable": os.access(str(args.dataset), os.R_OK),
        "mat_readable": os.access(str(mat_path), os.R_OK),
        "mat_root_keys": root_keys,
        "status": "PASS",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

