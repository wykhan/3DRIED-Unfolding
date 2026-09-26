# AGENTS.md

## 1. Project role

This repository implements a reproducible learning-based imaging baseline for the 3DRIED dataset.

The current project goal is:

> Build a physics-based ISTA unfolding baseline for 3DRIED 3-D reconstruction, starting from the Hammer target.

Codex acts as the implementation and experiment agent. It may autonomously inspect data, write code, run tests, debug ordinary implementation problems, execute approved experiments, update project records, and prepare milestone reports.

Codex must not silently change the scientific problem, physical model, data interpretation, or milestone acceptance criteria.

---

## 2. Current authorization scope

M0--M2 are complete and human-reviewed. The currently authorized autonomous
scope is:

- M3: forward / adjoint operator validation
- M4: conventional Kernel-ISTA

After M4 passes, STOP and request independent review.

Do not begin M5 or later milestones until PLAN.md explicitly marks them as unlocked.

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
validated forward / adjoint operator
    ↓
Kernel-ISTA
    ↓
ISTA-Unfolding
    ↓
Hammer inference
```

For the currently authorized M3-M4 phase, implement only the validated
forward/adjoint operator and conventional Kernel-ISTA stages. M0--M2 are
already complete; M5 and later stages remain locked.

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
7. changing the core algorithm from ISTA unfolding;
8. modifying raw dataset files;
9. deleting non-generated project files;
10. starting M3 or later before M2 review;
11. running an experiment estimated to require more than 6 GPU-hours;
12. using an undocumented assumption to resolve an ambiguous radar parameter.

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

For M2 specifically, stop after producing the RMA baseline report and request human review.

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
