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

---

## EXP-M3-001 — Holography operator validation

**Milestone:** M3
**Purpose:** Validate the generic forward/adjoint pair and its consistency
with the approved M2 reconstruction.
**Status:** PASS (Codex self-review)

### Command

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/validate_operator.py --config configs/m3_operator.yaml
```

### Configuration and data

Seed set: 7, 23, 101; sensitivity seed 313. Tests use complex64 and
complex128 on CPU plus complex64 on CUDA. Hammer consistency uses the same
Rx1 corrected `[200,407,256]` echo, MATLAB bins 12--24, 512×512 image grid,
and M2 physical parameters.

### Quantitative results

- seven complete unit tests passed;
- maximum complex64 adjoint relative error: `1.4088693e-6`;
- maximum CPU complex128 error: `1.6636601e-15`;
- CUDA complex64 error: `3.2709124e-7`;
- intentional wrong-phase control error: `1.1021071`;
- M2/M3 unaligned complex relative difference: `2.2377741e-7`;
- M2/M3 magnitude relative difference: `1.8396511e-7`;
- magnitude Pearson correlation: `0.9999999999999872`;
- least-squares comparison scalar: `1.000000119 - 1.375e-9j`;
- Hammer consistency runtime: 13.52 s;
- NaN/Inf: none.

The initial implementation built the complex64 wavenumber grid in float32,
producing an `8.39e-5` M2 complex discrepancy despite exact adjoint tests.
Matching M2's float64 grid construction reduced the discrepancy to
`2.24e-7`. No physical parameter or coordinate convention changed.

### Outputs

`results/m3_operator/metrics.json`, `m2_consistency.png`, saved config,
command, and environment. Report: `docs/M3_OPERATOR_REPORT.md`.

### Next action

Commit and push M3, then proceed to M4 because every blocking M3 self-review
item passes.
