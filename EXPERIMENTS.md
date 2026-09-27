# EXPERIMENTS.md

## Purpose

This file is the human-readable experiment ledger.

Every nontrivial experiment must have a unique ID.

Recommended format:

```text
EXP-YYYYMMDD-###
```

Example:

```text
EXP-20260926-001
```

---

# Experiment template

## EXP-YYYYMMDD-###

**Milestone:**
**Purpose:**
**Status:** PLANNED / RUNNING / PASS / FAILED / BLOCKED

### Code

```text
git commit:
branch:
```

### Environment

```text
host:
GPU:
Python:
PyTorch:
CUDA:
```

### Data

```text
dataset:
path:
Rx:
raw shape:
processed shape:
```

### Command

```bash
...
```

### Config

```text
...
```

### Outputs

```text
results/...
```

### Quantitative results

```text
runtime:
peak GPU memory:
NaN/Inf:
other metrics:
```

### Qualitative result

```text
...
```

### Interpretation

```text
...
```

### Problems encountered

```text
...
```

### Changes made

```text
...
```

### Next action

```text
...
```

---

# Initial planned experiments

## EXP-M0-001 — Environment smoke test

**Milestone:** M0
**Purpose:** Verify Python/PyTorch/GPU/project paths.
**Status:** PASS

Expected checks:

- import numpy;
- import scipy;
- import torch;
- CUDA availability;
- GPU name;
- read access to Hammer path.

Command: `python3 scripts/smoke_test.py --output results/m0_environment.json`

Result: Python 3.8.17, PyTorch 2.0.1+cu118, CUDA 11.8, two RTX 4090 GPUs,
and read-only HDF5 access verified. Output: `results/m0_environment.json`.

---

## EXP-M1-001 — Hammer raw data inspection

**Milestone:** M1
**Purpose:** Identify actual file structure, array shape, dtype, axes, metadata, and Rx layout.
**Status:** PASS

Dataset:

```text
/home/superws/dataset/3DRIED/Hammer/Hammer (1)
```

Important:

```text
single Rx only
```

No raw data modification is permitted.

Command: `python3 scripts/inspect_hammer.py --hash-files --output results/hammer_audit.json`

Result: 82,278 complex frames of MATLAB shape `[1,4,256]`; aperture is
`[z=200,x=407,fast-time=256]` after reference motion correction. MATLAB Rx1
(Python 0) is selected because the bundled RMA explicitly uses it. Output:
`results/hammer_audit.json`. Raw MAT SHA-256:
`167c5977d75deddac782710c287d5bd49cffefe04a67e0c88fa45ab8e30a7475`.

---

## EXP-M2-001 — Minimal Hammer RMA smoke reconstruction

**Milestone:** M2
**Purpose:** Run the smallest useful RMA path on a restricted range interval or reduced debugging subset.
**Status:** PASS

Goal:

- verify preprocessing;
- verify phase convention;
- verify target energy appears at plausible range.

This is a debugging experiment, not the final baseline.

Command: `python3 scripts/run_rma.py --config configs/hammer_smoke.yaml`

Config: Rx1 only; MATLAB bins 15--19; 407×200 full aperture; 512×512
spatial FFT. Output: `results/rma_hammer_smoke/`.

Result: complex64 volume `[5,512,512]`, 0 NaN/Inf, 13.74 s. Peak at
MATLAB bin 17 / 614.310 mm. The head and diagonal handle were visibly
focused, so the same reference phase convention proceeded to the full run.

---

## EXP-M2-002 — Full Hammer single-Rx RMA baseline

**Milestone:** M2
**Purpose:** Produce the reproducible conventional RMA reference reconstruction.
**Status:** PASS

Required outputs:

```text
results/rma_hammer/
```

Required evidence:

- recognizable Hammer;
- plausible range;
- plausible orientation;
- saved 3-D volume;
- saved MIP;
- saved representative slices;
- saved configuration;
- saved run log.

After this experiment:

```text
STOP FOR HUMAN REVIEW
```

### Code

Initial implementation commit: `ec78736` on `master`. Final milestone commit
is recorded in the Git history containing this ledger update.

### Environment

Python 3.8.17; NumPy 1.23.5; PyTorch 2.0.1+cu118; CUDA 11.8 available;
two NVIDIA GeForce RTX 4090 GPUs. RMA execution used CPU NumPy FFTs.

### Data and command

Dataset: `/home/superws/dataset/3DRIED/Hammer/Hammer (1)`
Rx: MATLAB 1 / Python 0 only
Raw: 82,278 frames, each `[1,4,256]` complex single
Corrected single-Rx echo: `[200,407,256]` complex64

```bash
/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rma.py --config configs/hammer.yaml
```

### Outputs and quantitative results

Output: `results/rma_hammer/`
Volume: `[13,512,512]` complex64, 27,263,104 bytes
Bins/range: MATLAB 12--24 / 0.406069--0.905848 m
Peak: bin 17, 0.614310 m, x=42.5 mm, z=141 mm
NaN/Inf: 0/0
Runtime: approximately 14.3 s
Repeatability: identical volume SHA-256
`257bc8c26a38af67ba4b2e5c10242930335031201cb16e0d10cecab4c6c4a40a`
across two consecutive runs.

### Qualitative result and interpretation

PASS. The 500 mm ROI clearly resolves a claw-like head and long straight
diagonal handle. Strong returns lie at 0.573--0.656 m and agree with the
supplied reference depth. The horizontal slope is mirrored relative to the
JPEG; this absolute x-handedness uncertainty is documented and deliberately
not corrected without human review.

### Problems and changes

`pytest` was unavailable, so tests were expressed with the standard-library
`unittest` runner (3/3 pass). The first full diagnostic plotted the entire
periodic 512 grid; applying the already-configured 500 mm visualization ROI
removed wraparound replicas while retaining the full saved complex volume.

### Next action

STOP for human review. M3 remains locked.

# RMIST reproduction series

The experiments below belong to the strict RMIST reproduction baseline.
Earlier generic M3/M4 experiments remain historical and are not acceptance evidence for M3-R/M4-R.

## EXP-RMIST-M3-001 — Strict RMIST kernel validation

**Milestone:** M3-R
**Purpose:** Validate RMIST Eqs. (13)--(15), physical padding, numerical
support handling, full-domain inversion, and Hammer M2 consistency.
**Status:** PASS

### Code and environment

Branch: `milestone/rmist-m3-m4`. Python 3.8.17, PyTorch 2.0.1+cu118,
CUDA 11.8, two RTX 4090 GPUs. CPU was used for the reference metrics and
one GPU for the CUDA smoke test.

### Data and command

Hammer `output_rawdata.mat`, MATLAB Rx1 / Python index 0 only. Corrected
physical echo `[200,407,256]`; RM domain `[512,512]`.

```bash
/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rmist_kernel_validation.py --config configs/rmist_kernel.yaml
```

### Results

Output: `results/m3_rmist_kernel/`. Runtime: 13.49 s. Reciprocal max/RMS
error: `4.4755e-16` / `1.1609e-16`. Synthetic full-domain relative error:
`4.1301e-7` complex64 and `7.2185e-16` complex128. Hammer M2 complex
relative difference: `2.2378e-7` (target `<1e-5`). Padding/crop and CUDA
smoke checks passed. No NaN or Inf occurred.

### Interpretation and next action

PASS. The strict reciprocal RMIST pair is compatible with approved M2
without changing its `1/ky` / `ky` structure. Proceed to M4-R conventional
RM-kernel ISTA only.
