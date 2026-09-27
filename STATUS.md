# STATUS.md

## Current overall status

```markdown
M0 PASS — human reviewed
M1 PASS — human reviewed
M2 PASS — human reviewed
M3-R PASS — independently reviewed
M4-R PASS — independently reviewed
M5-R UNLOCKED — architecture validation only
M6-R LOCKED
M7-R LOCKED

Previous generic M3/M4:
Historical only
Preserved on `milestone/m3-m4`
Not merged to `main`

Active work branch:
`milestone/rmist-m5`

Current task:
Implement and validate strict 9-phase RMIST-Net Eq. (20)-(26).
Stop after M5-R for independent review.

M6-R and M7-R remain locked. No training or real Hammer inference is
authorized.
```

M3-R PASS (2026-09-27). Strict RMIST kernels reproduce M2 with complex
relative difference `2.2378e-7`; reciprocal and full-domain round-trip tests
pass.

M4-R PASS (2026-09-27). Strict Eq. (16)--(19) Ini-ISTA and separately tuned
synthetic diagnostics were independently accepted with documented
non-blocking observations. M5-R and all training remain locked.

The remainder of this file is an immutable chronological history. Statements
about then-current locks or objectives below are historical, not active
governance.


## history below

## Current overall status

```text
M0 PASS / M1 PASS / M2 PASS — AWAITING HUMAN REVIEW
```

Current authorized execution:

```text
M0 → M1 → M2 → HUMAN REVIEW
```

M3 and later milestones are locked.

---

## Current objective

Build a conventional single-Rx RMA baseline for the 3DRIED Hammer target before implementing any learning-based reconstruction.

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

- [ ] Review M2 RMA reconstruction

### Locked

- [ ] M3 Forward / Adjoint Operator Validation
- [ ] M4 Kernel-ISTA
- [ ] M5 ISTA-Unfolding
- [ ] M6 Hammer Inference
- [ ] M7 Baseline Freeze

---

## Current task

Stop after M2 and request human review. Do not begin M3.

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

None. M1 found no blocking ambiguity: bundled reference scripts identify
Rx1, aperture axes, motion correction, and all required radar parameters.

---

## Decisions required from human

Human review of the M2 image and the documented horizontal-handedness
uncertainty is required before unlocking M3.

---

## Last completed milestone

M2 Conventional RMA Baseline.

---

## Next action

Human inspection of `results/rma_hammer/` and
`docs/M2_RMA_HAMMER_REPORT.md`; M3 remains locked.

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

Timestamp: 2026-09-27 Asia/Shanghai
Milestone: M3-R
Task: Strict RMIST RM / RM_dagger kernel reproduction and M2 validation
Status: PASS
Command: `/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rmist_kernel_validation.py --config configs/rmist_kernel.yaml`
Result: Reciprocal max error `4.4755e-16`; complex64/128 round trips `4.1301e-7`/`7.2185e-16`; M2 difference `2.2378e-7`; CUDA passed.
Evidence: `results/m3_rmist_kernel/`, `docs/M3_RMIST_KERNEL_REPORT.md`
Next action: M4-R strict conventional RM-kernel ISTA
Human decision required: NO

Timestamp: 2026-09-27 Asia/Shanghai
Milestone: M4-R
Task: Strict RM-kernel ISTA Eq. (16)--(19) synthetic validation
Status: PASS
Command: `/home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rmist_ista.py --config configs/rmist_ista.yaml`
Result: Ini-ISTA NRMSE `0.037414` noise-free and `0.045532` at 20 dB; separate tuned diagnostic NRMSE `0.010156` and `0.010271`; all finite.
Evidence: `results/m4_rmist_ista/`, `docs/M4_RMIST_ISTA_REPORT.md`
Next action: STOP for independent review; do not begin M5-R
Human decision required: YES
