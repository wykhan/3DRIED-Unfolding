# PLAN.md

## Project state

Current execution authorization:

```text
UNLOCKED: M0, M1, M2
UNLOCKED: M3, M4
LOCKED:   M5, M6, M7
```

M0--M2 have passed human review. Codex must stop after M4 for external review.

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

## Human checkpoint

After M2:

```text
STOP.
Do not begin M3.
Request human review.
```

---

# M3 — Forward / Adjoint Operator Validation

Status:

```text
PASS (Codex self-review; external review pending)
```

Planned after M2 review.

Goals:

- implement `HolographyOperator`;
- pass adjoint test;
- verify adjoint reconstruction is consistent with M2 RMA.

Target:

\[
e_{\rm adj}<10^{-5}.
\]

---

# M4 — Kernel-ISTA

Status:

```text
UNLOCKED
```

Planned goals:

- implement complex soft threshold;
- reconstruct synthetic sparse targets;
- verify residual decreases;
- compare adjoint initialization vs Kernel-ISTA.

---

# M5 — ISTA-Unfolding

Status:

```text
LOCKED
```

Planned goals:

- 5-layer unfolding baseline;
- learn only per-layer step size and threshold;
- train on synthetic oracle data;
- compare with multi-iteration Kernel-ISTA.

---

# M6 — Hammer Inference

Status:

```text
LOCKED
```

Planned goals:

- infer on real Hammer;
- compare RMA / Kernel-ISTA / ISTA-Unfolding;
- compute no-reference and measurement-consistency metrics.

---

# M7 — Baseline Freeze

Status:

```text
LOCKED
```

Planned deliverables:

- final reproducible config;
- checkpoint;
- code tag/commit;
- baseline report;
- experiment summary;
- known limitations;
- next-step recommendations.
