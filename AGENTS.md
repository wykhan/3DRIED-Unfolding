# AGENTS.md

## Full FMCW BP authorization — 2026-10-27 (latest)

The user explicitly chooses complete FMCW range-migration BP for 3DRIED and
requests a fresh optimization and single-slice performance assessment.
Implement and validate a separate range-profile/interpolation BP, including
the declared dechirp/FFT phase convention, then benchmark the full physical
aperture and 512x512 output. GPU parallelism, kernel fusion, bounded caching
and precision experiments with numerical evidence are authorized. Do not
reuse the single-frequency FFT timing as a wideband result. Keep legacy
operators/coordinates and frozen RMIST unchanged; unresolved Hammer phase
or range calibration remains an explicit gate for real-data acceptance.

## BP acceleration authorization — 2026-10-27 (latest)

The user explicitly retracts the literal unoptimized BP requirement and asks
to adopt optimized BP with documented changes and measured 3DRIED slice
timings. Exact FFT linear convolution, fixed-kernel caching and GPU execution
are now authorized. Preserve the phase-only spherical sum when assessing
acceleration; the legacy Green 1/R weight and sensitivity normalization are
separate model/gain choices. Report the narrowband fixed-depth scope and do
not promote it to validated full FMCW/Hammer range-migration BP.
The no-acceleration statements below are retained as historical scope.

## Current user authorization — 2026-10-27

The user explicitly supersedes the old suite's CV-ISTA display name and the
old no-training lock for a new independent ISTA-Net baseline. The five methods
are literal unoptimized phase-only BP, reference RMA, true-A/A^H ISTA,
original ISTA-Net (not plus), and frozen accepted RMIST-Net. Train ISTA-Net
first at 50% height-row sampling with disjoint train/validation/test scenes.
No BP FFT-convolution acceleration, adaptive weighting, or image enhancement.
Necessary range sampling/interpolation and correctness checks remain required.
Keep strict RMIST RM/RM_dagger and weights unchanged; M8/M9 do not establish a
beneficial Hammer normalization. Unverified rows must be SKIPPED or explicitly
diagnostic. Do not merge main or overwrite other worktrees. The historical
authorization below is retained as history, not a veto on this user request.

## Baseline branch authorization — 2026-10-24

On `baseline/3dried-imaging-suite`, the user's current request additionally
authorizes BP/RMA/ISTA/CV-ISTA/RMIST, point simulation and real echoes, the four
SRs 1.0/0.7/0.5/0.3, and explicit shared acquisition parameters. Read
`BASELINE_SPEC.md`, `configs/baseline_3dried.yaml`, and
`docs/BASELINE_PARAMETERS.md`. This is an additive baseline extension, not a
replacement of the accepted strict RMIST operator or a new training milestone.
The historical scope and locks below describe the reproduction mainline;
they do not block the explicitly requested new baseline algorithms/masks.
Keep single-Rx, original data read-only, accepted coordinates, and fixed RMIST
weights. Report external-data/GPU absence honestly. Do not merge this branch
or M9 into main as part of implementing this suite.

## 1. Project role

This repository implements a reproducible learning-based imaging baseline for the 3DRIED dataset.

The current project goal is:

> Reproduce the physics-based RMIST-Net method for 3DRIED 3-D reconstruction,
> starting from the Hammer target.

Codex acts as the implementation and experiment agent. It may autonomously inspect data, write code, run tests, debug ordinary implementation problems, execute approved experiments, update project records, and prepare milestone reports.

Codex must not silently change the scientific problem, physical model, data interpretation, or milestone acceptance criteria.

---

## 2. Current authorization scope

Milestone state:

- M0-M2: complete and human-reviewed;
- M3-R: strict `RM` / `RM_dagger`, externally reviewed PASS;
- M4-R: strict RM-kernel ISTA Eq. (16)-(19), externally reviewed and accepted;
- M5-R: RMIST-Net Eq. (20)-(26), independently reviewed PASS;
- M6-R: synthetic training Eq. (27)-(31), independently reviewed PASS with non-blocking observations;
- M7-R: INDEPENDENTLY REVIEWED — TECHNICAL PASS, with important scientific limitation;
- M8: INDEPENDENTLY REVIEWED PASS — `SCALE_MISMATCH_SUPPORTED`;
- M9: scale-normalized inference-quality evaluation, LOCKED.

The previous generic `milestone/m3-m4` implementation was independently reviewed and is not the accepted physics basis for learning.

Algorithmic source of truth:

1. RMIST-Net paper;
2. `docs/RMIST_REPRO_SPEC.md`;
3. human-approved 3DRIED M0-M2 geometry and preprocessing.

M5-R architecture implementation and validation are complete and independently
accepted. M6-R synthetic training is independently accepted. M7-R Eq. (32)
inference is independently accepted as a technical PASS: the Hammer remains
recognizable with preserved range/orientation, but learned improvement over
RMA50/Ini-ISTA50 is not demonstrated. M8 is independently accepted: scale
mismatch is the dominant demonstrated mechanism of Hammer inactivity, but
normalization has not been shown to improve real-data image quality. No
training, fine-tuning, production normalization, domain adaptation,
calibration learning, parameter changes, or M9 work are authorized.

Codex must not replace RMIST's `RM` / `RM_dagger` inverse-kernel structure with a Hilbert-adjoint pair, FISTA, LISTA, Gamma-Net, Toe-LISTA, CNN proximal blocks, or another architecture.

---

## 3. Dataset

Primary real dataset path:

```text
/home/superws/dataset/3DRIED/Hammer/Hammer (1)
```

Important fixed decision:

- Use only ONE Rx channel for the baseline.
- Do not fuse, average, concatenate, calibrate, or jointly process the other Rx channels.
- If multiple Rx channels are present, select one channel and document exactly which channel is used.
- Prefer the first Rx channel unless the dataset metadata or existing reference code clearly identifies another channel as the one used in the Wang Mou series implementation.
- If this cannot be determined reliably, stop and record the ambiguity in STATUS.md.

Raw dataset files are read-only.

Never overwrite, rename, move, or modify source data.

---

## 4. Scientific constraints

The baseline must follow this high-level pipeline:

```text
3DRIED raw echo
    ↓
single-Rx selection
    ↓
range preprocessing / range FFT
    ↓
conventional RMA reconstruction
    ↓
strict RM / RM_dagger kernels
    ↓
strict RM-kernel ISTA Eq. (16)-(19)
    ↓
RMIST-Net Eq. (20)-(26), when explicitly unlocked
    ↓
synthetic RMIST training Eq. (27)-(31), when explicitly unlocked
    ↓
real Hammer slice-wise inference Eq. (32), when explicitly unlocked
```

The pipeline is implemented and accepted through strict RM-kernel ISTA.
Later stages shown above remain locked.

Do not introduce:

- Gamma-Net
- Toe-LISTA
- CNN post-processing
- learned forward operators
- GANs
- diffusion models
- self-supervised losses
- sparse-aperture experiments
- multi-Rx fusion

unless explicitly authorized later.

---

## 5. Coding rules

### 5.1 Reproducibility

Every executable experiment must have:

- explicit config or command-line arguments;
- fixed random seed where randomness exists;
- saved command line;
- saved environment information;
- saved outputs under `results/`;
- a short experiment record in `EXPERIMENTS.md`.

### 5.2 File organization

Prefer:

```text
src/
  data/
  imaging/
  solvers/
  models/
  simulation/

scripts/
tests/
configs/
results/
docs/
```

Do not place new implementation code into legacy Toe-LISTA files.

### 5.3 Numerical conventions

For new PyTorch FFT-based code, prefer:

```python
torch.fft.fft2(..., norm="ortho")
torch.fft.ifft2(..., norm="ortho")
```

unless reproduction of an official/reference RMA implementation requires another convention.

If another convention is used, document it.

### 5.4 Complex data

Preserve complex-valued radar data as complex values whenever possible.

Do not discard phase information before final visualization.

### 5.5 Visualization

At minimum, save:

- representative range slices;
- x-z maximum-intensity projection;
- one 3-D visualization or point-cloud-style thresholded visualization;
- normalized dB visualization.

Do not use visualization success as proof of mathematical correctness.

---

## 6. Autonomous actions allowed

Codex may autonomously:

- inspect repository structure;
- inspect dataset files in read-only mode;
- identify shapes, dtypes, keys, metadata;
- write loaders;
- write RMA code;
- write unit tests;
- fix Python errors;
- fix tensor shapes;
- fix device placement;
- choose CPU/GPU execution;
- reduce batch/chunk size after OOM;
- add logging;
- create configs;
- add comments and documentation;
- run smoke tests;
- rerun failed tests after code fixes;
- generate figures;
- create commits;
- update STATUS.md;
- update EXPERIMENTS.md.

---

## 7. Actions requiring human approval

Stop before doing any of the following:

1. changing the physical imaging model;
2. changing the coordinate convention after it has been established;
3. combining multiple Rx channels;
4. replacing the single-Rx rule;
5. using an RMA image as supervised ground truth;
6. skipping a milestone acceptance test;
7. changing the core algorithm from strict RMIST reproduction;
8. modifying raw dataset files;
9. deleting non-generated project files;
10. starting any locked milestone without explicit human authorization;
11. running an experiment estimated to require more than 6 GPU-hours;
12. using an undocumented assumption to resolve an ambiguous radar parameter;
13. changing the RMIST `1/ky` forward and `ky` inverse-kernel structure;
14. changing the RMIST 9-phase network structure;
15. changing RMIST-reported training hyperparameters;
16. substituting a mathematically different operator because it has better adjoint properties;
17. treating an unspecified paper parameter as if it were reported.

---

## 8. Error-handling policy

Ordinary implementation errors are not stop conditions.

For:

- import errors;
- path errors;
- dtype errors;
- tensor shape errors;
- CUDA OOM;
- NaN caused by obvious numerical instability;
- plotting errors;
- checkpoint errors;

Codex should diagnose, fix, rerun, and continue.

If three materially different repair attempts fail on the same scientific issue, summarize the attempts in STATUS.md and stop.

---

## 9. Milestone reporting

At the end of every milestone:

1. run all relevant tests;
2. save outputs;
3. update STATUS.md;
4. update EXPERIMENTS.md;
5. record nontrivial choices in DECISIONS.md;
6. create a milestone summary under `docs/`;
7. commit changes.

Milestone-specific stop and review gates are defined in `PLAN.md` and
`docs/RMIST_REPRO_SPEC.md`.

---

## 10. Definition of "PASS"

A milestone is not PASS because code runs.

PASS requires the acceptance criteria defined in PLAN.md.

If criteria are not met, mark the milestone:

```text
BLOCKED
```

or

```text
FAILED
```

with evidence and next diagnostic steps.
