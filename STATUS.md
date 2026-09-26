# STATUS.md

## Current overall status

M0 PASS — human reviewed
M1 PASS — human reviewed
M2 PASS — human reviewed
M3 PASS — Codex self-review; external review pending
M4 PASS — Codex self-review; external review pending

Current authorized execution:
M3 → M4 → EXTERNAL REVIEW

M5 and later milestones are locked.


## history below

## Current overall status

```text
M0 PASS / M1 PASS / M2 PASS / M3 PASS / M4 PASS
```

Current authorized execution:

```text
M3 → M4 → EXTERNAL REVIEW
```

M5 and later milestones are locked.

---

## Current objective

Validate the M2-consistent physics operator and conventional non-learned
Kernel-ISTA baseline, then stop before M5 for independent review.

---

## Dataset

Target:

```text
/home/superws/dataset/3DRIED/Hammer/Hammer (1)
```

Fixed decision:

```text
single Rx only
```

Other Rx channels must not be fused or averaged.

---

## Milestones

- [x] M0 Repository Initialization
- [x] M1 Hammer Dataset Audit
- [x] M2 Conventional RMA Baseline

### Human checkpoint

- [x] Review M2 RMA reconstruction

### Current authorized milestones

- [x] M3 Forward / Adjoint Operator Validation (self-review PASS)
- [x] M4 Kernel-ISTA (self-review PASS)

### Locked

- [ ] M5 ISTA-Unfolding
- [ ] M6 Hammer Inference
- [ ] M7 Baseline Freeze

---

## Current task

Stop after M4 and request independent review. Do not begin M5.

---

## Known fixed technical decisions

1. Clean 3DRIED-specific repository.
2. Do not use Gamma-Net as implementation base.
3. Do not use Toe-LISTA as implementation base.
4. Use one Rx only.
5. First reproduce conventional RMA.
6. Later use physics-based FFT holographic forward/adjoint operator.
7. Later use minimal ISTA unfolding.
8. Synthetic oracle training; real RMA is not supervised GT.

---

## Current blockers

None. M3 and M4 pass every blocking Codex self-review gate.

---

## Decisions required from human

No implementation blocker. External review is required before M5.

---

## Last completed milestone

M4 Kernel-ISTA (Codex self-review).

---

## Next action

Push the completed M3-M4 branch and request independent review. M5 remains locked.

---

## Update format for Codex

After each significant run, update this section:

```text
Timestamp:
Milestone:
Task:
Status: RUNNING / PASS / FAILED / BLOCKED
Command:
Result:
Evidence:
Next action:
Human decision required: YES / NO
```

---

Timestamp: 2026-09-26 Asia/Shanghai
Milestone: M0
Task: Repository initialization and environment smoke test
Status: PASS
Command: `python3 scripts/smoke_test.py --output results/m0_environment.json`
Result: Python environment, two RTX 4090 GPUs, and Hammer HDF5 readability verified.
Evidence: `results/m0_environment.json`, `docs/M0_REPOSITORY_REPORT.md`
Next action: M1 dataset audit
Human decision required: NO

Timestamp: 2026-09-26 Asia/Shanghai
Milestone: M1
Task: Read-only Hammer audit
Status: PASS
Command: `python3 scripts/inspect_hammer.py --hash-files --output results/hammer_audit.json`
Result: Complex `[frame=82278, chirp=1, Rx=4, ADC=256]` capture interpreted; Rx1 selected and 407×200 aperture conventions established from bundled reference code.
Evidence: `results/hammer_audit.json`, `docs/HAMMER_DATA_AUDIT.md`
Next action: M2 conventional RMA
Human decision required: NO

Timestamp: 2026-09-26 Asia/Shanghai
Milestone: M2
Task: Minimal diagnostic and full single-Rx conventional RMA
Status: PASS
Command: `/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rma.py --config configs/hammer.yaml`
Result: Recognizable complex Hammer volume, peak at 614.310 mm, 0 NaN/Inf, deterministic volume across repeated runs.
Evidence: `results/rma_hammer/`, `docs/M2_RMA_HAMMER_REPORT.md`
Next action: STOP and obtain human review before M3
Human decision required: YES

Timestamp: 2026-09-26 Asia/Shanghai
Milestone: M3
Task: Reusable holography forward/adjoint implementation and validation
Status: PASS (Codex self-review; external review pending)
Command: `python3 scripts/validate_operator.py --config configs/m3_operator.yaml`
Result: worst adjoint error 1.409e-6; complex128 error <=1.664e-15; intentional wrong-phase error 1.102; M2 complex relative difference 2.238e-7.
Evidence: `results/m3_operator/`, `docs/M3_OPERATOR_REPORT.md`
M3 implementation commit: `f77f61ab69b615d3ebc7bd723f9b41876ef0a42a`
Next action: commit/push M3, then execute M4
Human decision required: NO

Timestamp: 2026-09-26 Asia/Shanghai
Milestone: M4
Task: Non-learned Kernel-ISTA on a noise-free complex sparse slice
Status: PASS (Codex self-review; external review pending)
Command: `python3 scripts/run_ista.py --config configs/kernel_ista_baseline.yaml`
Result: NMSE improved from 0.648365 to 5.88191e-6 in 250 iterations; objective was monotone within float32 tolerance; output was bitwise reproducible.
Evidence: `results/m4_kernel_ista/`, `docs/M4_KERNEL_ISTA_REPORT.md`
Commit: pending M4 milestone commit
Next action: commit/push M4, then STOP for independent review
Human decision required: YES
