# M0 Repository Initialization Report

Date: 2026-09-26 (Asia/Shanghai)  
Status: **PASS**

The clean project structure, packaging metadata, configuration, scripts,
tests, results directory, documentation directory, `.gitignore`, and six root
control documents are present. The project is independent of legacy
TomoSAR/Toe-LISTA code.

## Verification

Command:

```bash
python3 scripts/smoke_test.py --output results/m0_environment.json
```

The check passed with Python 3.8.17, NumPy 1.23.5, SciPy 1.10.1, h5py
3.11.0, PyTorch 2.0.1+cu118, CUDA 11.8, and two NVIDIA GeForce RTX 4090
GPUs available. The Hammer directory and its MATLAB v7.3 file were readable.

The pre-work SHA-256 of the raw MAT file was
`167c5977d75deddac782710c287d5bd49cffefe04a67e0c88fa45ab8e30a7475`.
The dataset was opened read-only and was not changed.

