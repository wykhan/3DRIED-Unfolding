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
**Status:** PLANNED

Goal:

- verify preprocessing;
- verify phase convention;
- verify target energy appears at plausible range.

This is a debugging experiment, not the final baseline.

---

## EXP-M2-002 — Full Hammer single-Rx RMA baseline

**Milestone:** M2  
**Purpose:** Produce the reproducible conventional RMA reference reconstruction.  
**Status:** PLANNED

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
