# EXPERIMENTS.md

## 2026-10-27 — original ISTA-Net baseline extension

Branch: `baseline/ista-net-validation`. Environment: `tmux 3dried`,
ComplexValue Python 3.8.17, PyTorch 2.0.1+cu118, RTX 4090.
New+legacy unit suite: 61 tests PASS; later additions recorded separately.
Command: `CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/train_ista_net.py --config configs/ista_net_baseline.yaml --rate 0.5 --output results/ista_net_h50_v2`.
Fixed protocol: 128/32/32 disjoint full-size RM point scenes, 20 dB AWGN,
20 epochs, Adam 1e-4, batch one, 9 phases / 32 features, height-row mask.
50% completed in about 186 s, best validation epoch 18, validation Complex
NMSE 0.01049538, independent test 0.01336434; conservative 200-step ISTA
0.93952807 is a fixed-budget comparator, not a converged optimum.
Checkpoint SHA-256 6707938295db9da6cd83bad983c79f7c109d08cfa1aea15301b076c73f5b1ec5.
Initial invocation failed before training on relative-path provenance handling;
the corrected rerun uses a fresh immutable output directory and retains logs.
Other rates use the same fixed training protocol, separately trained weights.

## 2026-10-24 — Unified imaging baseline

Command: `python scripts/run_baseline_suite.py --no-figures --output-dir results/baseline_20261024_final`.
The earlier `baseline_20261024_verified` run generated the saved figures from
identical complex arrays.
Configuration: `configs/baseline_3dried.yaml`; full 200×407 aperture,
512×512 image grid, 13 bins, fixed M6 scalars. Five methods × four SRs ×
two patterns: 40 point-simulation rows; real 40 rows BLOCKED_MISSING_DATA.
CPU NumPy/SciPy; no GPU result claimed. Output summary intentionally says
INCOMPLETE when real data is requested. See implementation report and
`results/review/BASELINE/summary_metrics.csv`. Tests: 46 PASS, 3 skips.


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

## EXP-20260927-008 — M8 fixed-model sim-to-real diagnostic

**Milestone:** M8

**Purpose:** Explain why accepted synthetic RMIST gain collapses on Hammer.

**Status:** PASS — SCALE_MISMATCH_SUPPORTED — AWAITING REVIEW

### Code

```text
branch: milestone/rmist-m8
```

### Environment

```text
GPU: NVIDIA GeForce RTX 4090
Python: 3.11.4
PyTorch: 2.2.2+cu121
CUDA: 12.1
```

### Data

```text
synthetic: first 32 accepted M6 validation samples
real: accepted Rx1 Hammer bins 12-24; primary bin 17
model: accepted M6 epoch 99, unchanged
```

### Command

```bash
PYTHONPATH=src python3 scripts/diagnose_rmist_sim2real.py --config configs/rmist_sim2real_diagnostic.yaml
```

### Outputs

```text
results/review/M8/
```

### Quantitative results

```text
runtime: 22.12 s
synthetic alpha0 RMS median: 0.01376872018
Hammer bin17 alpha0 RMS: 31810917.7538
real/synthetic RMS ratio: 2.310375789e9
c_align: 4.328300204e-10
raw final E_RMA: 1.141222e-6
aligned final E_RMA: 0.7151915
trace vs ordinary output: exact
```

### Qualitative result

At raw scale, Hammer thresholds and shrinkage are effectively inactive. Scale
alignment restores phase activity comparable to the synthetic control.

### Interpretation

`SCALE_MISMATCH_SUPPORTED`. This is diagnostic evidence only; broader
distribution mismatch is not excluded and no production normalization is authorized.

### Problems encountered

None. An initial root-category rule conflated downstream correction activity
with an independent residual cause; phase-1 evidence clarified that residual
and correction ratios are initially comparable while shrinkage differs by
about `2.9e10`-fold.

### Changes made

Added external tracing, metric/statistic helpers, fixed-control diagnostic,
counterfactual scale sweep, tests, plots, and report. No model state changed.

### Next action

STOP for independent review.

## EXP-20260927-007 — Strict M7-R Hammer inference

**Milestone:** M7-R

**Purpose:** Apply accepted M6 RMIST parameters slice-wise to real Hammer data.

**Status:** TECHNICAL PASS / REAL-DATA SUCCESS — AWAITING REVIEW

### Code

```text
branch: milestone/rmist-m7
```

### Environment

```text
GPU: NVIDIA GeForce RTX 4090
Python: 3.11.4
PyTorch: 2.2.2+cu121
CUDA: 12.1
```

### Data

```text
dataset: 3DRIED Hammer
path: /home/superws/dataset/3DRIED/Hammer/Hammer (1)
Rx: MATLAB Rx1 / Python index 0
raw selected shape: [200,407,256]
output shape: [13,512,512]
```

### Command

```bash
PYTHONPATH=src python3 scripts/run_rmist_hammer.py --config configs/rmist_hammer_inference.yaml
```

### Outputs

```text
results/review/M7/
results/m7_rmist_hammer/  # large ignored volumes and 3-D figure
```

### Quantitative results

```text
runtime: 22.17 s
NaN/Inf: 0
effective physical SR: 0.4997788698
mean E_obs RMA50: 0.0849890322
mean E_obs Ini-ISTA50: 0.0849890328
mean E_obs RMIST50: 0.0849889525
```

### Qualitative result

The head and continuous diagonal handle remain recognizable around bin 17
(614.310 mm). Orientation matches accepted M2. RMIST50 is visually nearly
identical to both sparse baselines.

### Interpretation

Eq. (32) is technically reproduced without catastrophic sim-to-real failure,
but this model does not demonstrate a material learned improvement on Hammer.

### Problems encountered

None. The lack of meaningful separation from sparse baselines is reported as
a non-blocking scientific limitation, not repaired through unauthorized adaptation.

### Changes made

Added inference-only implementation, configuration, tests, review artifacts,
and milestone report. No learned values or raw data changed.

### Next action

STOP for independent review.

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

## EXP-RMIST-M4-001 — Strict RM-kernel ISTA synthetic validation

**Milestone:** M4-R
**Purpose:** Reproduce RMIST Eq. (16)--(19) with Ini-ISTA parameters, then
separately evaluate a documented tuned diagnostic and 50% mask extension.
**Status:** PASS

### Code and environment

Branch: `milestone/rmist-m3-m4`. Python 3.8.17 and PyTorch 2.0.1+cu118.
The 128×128 validation ran on CPU; CUDA remained available.

### Command and data

```bash
/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rmist_ista.py --config configs/rmist_ista.yaml
```

Twelve unit complex point scatterers with uniform random phase, strict RM
echo generation, `Mask=1` noise-free first, separate 20 dB complex AWGN,
and a separate 50% Bernoulli-mask extension. Seed: 20260927.

### Results

Output: `results/m4_rmist_ista/`. Total runtime: 1.88 s. Ini-ISTA
(`tau=1e-4`, `theta=1e-3`, 9 iterations) NRMSE was `0.037414` noise-free
and `0.045532` at 20 dB. A 24-point, nine-iteration grid search on the fixed
20 dB full-data case selected diagnostic `tau=1.0`, `theta=0.01`, yielding
NRMSE `0.010156` noise-free and `0.010271` at 20 dB. At 20 dB with a
realized 0.50549 sampling rate, Ini/tuned NRMSE was `0.695072`/`0.671940`.
All outputs were finite.

### Interpretation and next action

PASS. Eq. (19), complex shrinkage, provenance labeling, and synthetic tests
meet M4-R gates. The paper-style 512×512 Monte Carlo benchmark is explicitly
deferred. Stop for independent review; M5-R remains locked.

## REVIEW-GATE-RMIST-M3-M4-001 — Independent external review

**Scope:** M3-R and M4-R
**Status:** ACCEPTED

Independent external review concluded that M3-R passes and M4-R passes with
non-blocking observations. The documented non-monotonic Ini-ISTA residual and
weak 50% sampling diagnostic do not change Eq. (19) and do not block future
M5-R architecture work. No new experiment was performed for this review-gate
record. M5-R remains locked until explicit human authorization.

## EXP-RMIST-M5-001 — Strict RMIST-Net architecture validation

**Milestone:** M5-R
**Purpose:** Validate the literal nine-phase RMIST Eq. (20)--(26)
architecture without optimization or training.
**Status:** PASS — AWAITING INDEPENDENT REVIEW

### Code and environment

Branch: `milestone/rmist-m5`. Python 3.8.17, PyTorch 2.0.1+cu118, CUDA
11.8, NVIDIA GeForce RTX 4090. Deterministic seed: 20260927.

### Command

```bash
PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  scripts/run_rmist_net_validation.py --output results/review/M5

PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  -m unittest discover -s tests -v
```

### Results

The model executes exactly nine phases and has exactly 18 trainable scalars:
`tau[9]` initialized to `1e-4` and `theta[9]` initialized to `1e-3`.
Initialization relative difference against direct `RM_dagger(S)` is zero.
M4-to-M5 relative difference is `0.0` for complex64 and
`1.21362588613636e-13` for complex128, below both blocking tolerances. All
gradient entries are finite and nonzero under the strict RMIST kernel. Single,
batch `[2,32,32]`, CPU complex64/complex128, and CUDA complex64 checks pass.
The phase trace has exactly nine entries. The full repository suite passes
22/22 tests.

### Outputs and interpretation

Lightweight evidence is in `results/review/M5/`. PyTorch complex internal
representation is a paper-unspecified implementation choice recorded in
D027. No optimizer was created, no optimizer step was taken, no training
dataset was generated, and no real Hammer inference was performed. M5-R
passes self-validation and stops for independent review; M6-R remains locked.

## EXP-RMIST-M6-001 — Strict RMIST synthetic training

**Milestone:** M6-R
**Purpose:** Reproduce RMIST Eq. (27)--(31) synthetic training using the
accepted nine-phase architecture.
**Status:** PASS — AWAITING INDEPENDENT REVIEW

### Code, environment, and command

Branch: `milestone/rmist-m6`. Python 3.11.4, PyTorch 2.2.2+cu121, CUDA 12.1,
NVIDIA GeForce RTX 4090. Seed: 20260927.

```bash
PYTHONPATH=src python3 scripts/train_rmist.py --config configs/rmist_train.yaml
```

### Data and strict training settings

Exactly 1000 procedural training pairs and 200 separately seeded validation
pairs; 512×512 grid; 100 unique complex point scatterers; amplitude
`Uniform(0.5,1.0)`; phase `Uniform(0,2*pi)`; range 0.5 m; one fixed binary
50% mask; strict RM echoes; 20 dB circular complex AWGN. Adam used learning
rate `1e-4`, batch size 8, nine phases, and exactly 100 epochs. Training loss
was the displayed Eq. (27) mean per-sample complex L2 norm.

### Results

Preflight passed determinism, realized mask rate `0.4999198914`, achieved SNR
`20.0` dB, all-18-gradient finiteness, exact kernel-cache equivalence, and a
tiny-overfit reduction from `4.732818` to `4.120745`. Full training completed
in `228.535` s with peak allocated GPU memory `629.004` MiB. Train loss was
`4.115710` at epoch 1 and `4.048267` at epoch 100. Validation loss was
`4.688689` untrained, best `4.043289` at epoch 99, and `4.043440` final.

Held-out untrained/Ini-ISTA NRMSE was `0.614831`; the best trained model
reached `0.530265`. Trained PSNR/TBR were `42.00497`/`17.77521` dB versus
`40.72047`/`3.28918` dB untrained. All 27 repository tests pass.

### Artifacts and next action

Lightweight artifacts are under `results/review/M6/`. Best-validation and
epoch-100 checkpoints remain local under ignored `results/m6_rmist_training/`.
No full dataset was stored or committed, and no Hammer inference occurred.
Stop for independent review; M7-R remains locked.

## 2026-10-27 — authorized BP acceleration / single-slice controls

After the user retracts mandatory unoptimized BP, preserve the phase-only
spherical sum while accelerating it by exact linear FFT convolution,
kernel-spectrum caching and CUDA. Command in tmux 3dried / ComplexValue:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/benchmark_bp_slice.py --output results/bp_fft_slice_20261027/run`.
Independent analytic single/five-point echoes use the full 200x407 aperture
and 512x512 output, depth 0.6143103952815643 m, 78.8 GHz, height masks at
100/70/50/20%, seed 20261024. All eight controls have zero grid localization
error; direct-sum spot relative error <=1.476e-12. Complex128 CPU/CUDA relative
error <=8.70e-16. Two warmups/ten repeats, one CPU FFT worker: CPU uncached
203.84--206.59 ms, cached 111.29--113.42 ms; RTX 4090 cached with transfers
2.036--2.125 ms, device-resident 1.385--1.388 ms. Kernel setup separately
127.46 ms CPU /79.67 ms GPU, excluding GPU context creation. All 68 tests pass.
Unified-entry 20% CUDA full-volume smoke completes as cross-model DIAGNOSTIC.
Saved configs, masks, targets, phases, timing samples, provenance and logs:
`results/bp_fft_slice_20261027/`; lightweight evidence: `results/review/BP_FFT/`.
No Hammer run, RMIST change, raw-data modification or full FMCW BP claim.

## 2026-10-27 — completed matrix and diagnostics

Four height-rate models each completed 20 epochs. Independent test NMSE at 100/70/50/30% is 0.01287696/0.01270838/0.01336434/0.01483224. Five entry-point runs produced 100 rows (54 completed, 46 SKIPPED), with two warmups/five repeats and preprocessing excluded from times. Lightweight configs/seeds/hashes/metrics are in results/review/ISTA_NET; volumes/weights remain ignored. BP local controls use full aperture and literal loops; full-FOV budget calls return no partial image. A validation-only 20000-step ISTA diagnostic confirms underconvergence at 200. Hammer50 residual: ISTA-Net 1.57259, RMA 0.05518. No tuning or M9 normalization follows this observation. BASELINE_REPORT.md contains exact commands and limitations.
