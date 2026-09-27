# PLAN.md

## Project state

Current execution authorization:

```text
COMPLETED + HUMAN APPROVED: M0, M1, M2
COMPLETED + EXTERNALLY REVIEWED: M3-R (PASS), M4-R (PASS)
HISTORICAL / NOT MERGED: previous generic M3-M4 branch
LOCKED: M5-R, M6-R, M7-R
```

Define milestones:

```markdown
M3-R — Strict RM / RM_dagger implementation
M4-R — RM-kernel ISTA Eq. (16)-(19)
M5-R — RMIST-Net Eq. (20)-(26)
M6-R — synthetic training Eq. (27)-(31)
EXTERNAL REVIEW
M7-R — real Hammer slice-wise inference Eq. (32)
```

The RMIST paper and `docs/RMIST_REPRO_SPEC.md` are the algorithmic authority
for M3-R onward. Merge completion does not unlock M5-R.
---

# M0 — Repository Initialization

## Goal

Create a clean 3DRIED unfolding project without modifying the existing TomoSAR-Toe-LISTA codebase.

## Tasks

- [x] create repository/project directory structure;
- [x] initialize Git if needed;
- [x] record Python, PyTorch, CUDA, GPU, OS versions;
- [x] create `configs/`, `src/`, `scripts/`, `tests/`, `results/`, `docs/`;
- [x] place the six AI-native control files at repository root;
- [x] add `.gitignore`;
- [x] create minimal environment/smoke test;
- [x] verify dataset path is readable;
- [x] make no changes to raw dataset.

## Acceptance criteria

PASS only if:

- [x] repository runs a minimal Python smoke test;
- [x] GPU availability is recorded;
- [x] Hammer dataset path is readable;
- [x] raw dataset remains untouched;
- [x] working tree is clean after initial commit.

## Deliverable

```text
docs/M0_REPOSITORY_REPORT.md
```

---

# M1 — Hammer Dataset Audit

## Goal

Determine the actual data structure of:

```text
/home/superws/dataset/3DRIED/Hammer/Hammer (1)
```

using read-only inspection.

## Required questions

- [x] What files exist?
- [x] What are their formats?
- [x] What variables/keys are stored?
- [x] What is the actual shape?
- [x] What is the dtype?
- [x] Is the echo already complex-valued?
- [x] Which axis is ADC/range-fast-time?
- [x] Which axis is horizontal aperture?
- [x] Which axis is vertical aperture?
- [x] Which axis is Rx?
- [x] What acquisition/system metadata is present?
- [x] Are reference images included?
- [x] Is there preprocessing or reference MATLAB/Python code nearby?

## Single-Rx rule

- [x] identify available Rx channels;
- [x] select ONE Rx only;
- [x] document the selected Rx index;
- [x] do not average or fuse Rx channels.

## Required script

```text
scripts/inspect_hammer.py
```

It must be rerunnable and produce a concise machine-readable summary.

## Deliverable

```text
docs/HAMMER_DATA_AUDIT.md
```

## Stop conditions

STOP if:

- data dimensions cannot be interpreted reliably;
- the Rx axis cannot be identified;
- the selected Rx cannot be justified;
- essential radar parameters required for RMA are missing;
- file contents conflict materially with known 3DRIED descriptions.

## Acceptance criteria

PASS only if the raw-to-range-processing input layout is unambiguous enough to implement M2.

---

# M2 — Conventional RMA Baseline

## Goal

Reconstruct a recognizable 3-D Hammer image from the selected single-Rx raw echo.

## Required implementation

Suggested files:

```text
src/data/three_dried.py
src/imaging/rma.py
scripts/run_rma.py
configs/hammer.yaml
```

## Required pipeline

```text
raw Hammer echo
    ↓
single-Rx extraction
    ↓
range preprocessing / range FFT
    ↓
range-focused echo cube
    ↓
2-D phase-compensation reconstruction
    ↓
stack range slices
    ↓
3-D complex volume
    ↓
magnitude / dB visualization
```

## Important constraints

- Do not use deep learning.
- Do not use multiple Rx.
- Do not optimize for speed yet.
- Do not invent missing radar parameters.
- Prefer reproduction of official/reference processing when available.
- Save intermediate array shapes and physical coordinate ranges.

## Required outputs

```text
results/rma_hammer/
    volume.npy
    mip_xz.png
    representative_slices/
    volume_3d.png
    run.log
    config_used.yaml
    environment.txt
    metrics_or_stats.json
```

## Required report

```text
docs/M2_RMA_HAMMER_REPORT.md
```

The report must answer:

1. Which Rx was used?
2. What was the raw shape?
3. What preprocessing was applied?
4. What range bins were reconstructed?
5. What physical x/y/z ranges were used?
6. Is the Hammer recognizable?
7. Is its location/orientation plausible?
8. How does the result compare qualitatively with available 3DRIED reference imagery?
9. What uncertainties remain?

## Acceptance criteria

PASS only if:

- [x] RMA code completes without NaN/Inf;
- [x] reconstructed target is recognizable as Hammer;
- [x] target is located at a physically plausible range;
- [x] target orientation is plausible;
- [x] outputs are reproducible from config;
- [x] all important assumptions are documented.

## Human checkpoint — completed

After M2:

```text
M2 was human-reviewed and approved before M3-R began.
```

---

# M3-R — Strict RM / RM_dagger

Status:

```text
COMPLETED — EXTERNALLY REVIEWED PASS
```

Accepted strict reciprocal kernels and M2 consistency evidence are documented
in `docs/M3_RMIST_KERNEL_REPORT.md`.

---

# M4-R — Strict RM-Kernel ISTA Eq. (16)-(19)

Status:

```text
COMPLETED — EXTERNALLY REVIEWED PASS
```

Accepted Eq. (19), complex shrinkage, and synthetic evidence are documented
in `docs/M4_RMIST_ISTA_REPORT.md`. The independently reviewed non-blocking
observations are retained in `REVIEW_CHECKLIST.md`.

---

# M5-R — RMIST-Net Architecture Eq. (20)-(26)

Status:

```text
LOCKED
```

Do not begin without explicit human authorization. When unlocked, reproduce
the 9-phase RMIST-Net with only phase-wise `tau_t` and `theta_t` trainable.

---

# M6-R — RMIST Synthetic Training Eq. (27)-(31)

Status:

```text
LOCKED
```

After M5-R is completed and authorized, reproduce the RMIST synthetic
training protocol. Then stop for external review.

---

# M7-R — Real Hammer Slice-Wise Inference Eq. (32)

Status:

```text
LOCKED
```

Apply an accepted, reviewed RMIST model independently by range slice to the
single-Rx Hammer data. Hammer RMA must not be used as supervised ground truth.
