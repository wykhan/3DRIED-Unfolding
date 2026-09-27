# STATUS.md

## Current overall status

```markdown
M0 PASS — human reviewed
M1 PASS — human reviewed
M2 PASS — human reviewed
M3-R PASS — independently reviewed
M4-R PASS — independently reviewed
M5-R PASS — independently reviewed
M6-R PASS — independently reviewed with non-blocking observations
M7-R INDEPENDENTLY REVIEWED — TECHNICAL PASS WITH IMPORTANT SCIENTIFIC LIMITATION
M8 INDEPENDENTLY REVIEWED PASS — SCALE_MISMATCH_SUPPORTED
M9 SCALE-NORMALIZED INFERENCE QUALITY — LOCKED

Previous generic M3/M4:
Historical only
Preserved on `milestone/m3-m4`
Not merged to `main`

Active work branch:
`milestone/rmist-m8`

Current task:
M8 is accepted. Stop; M9 remains locked.

M7-R is independently accepted as a technical PASS. Hammer remains recognizable
and range/orientation are preserved. RMIST50 is numerically almost identical to
RMA50/Ini-ISTA50, so learned improvement on real Hammer is NOT DEMONSTRATED.
M8 independently establishes `SCALE_MISMATCH_SUPPORTED`: the `alpha0` RMS
ratio is approximately `2.3104e9`, raw Hammer phase-1 residual/correction
activity is comparable to synthetic, raw threshold activity is effectively
zero, and counterfactual alignment restores strong phase dynamics. It does
not establish improved Hammer image quality. M9, training, fine-tuning,
production normalization, and parameter changes remain prohibited.
```

M3-R PASS (2026-09-27). Strict RMIST kernels reproduce M2 with complex
relative difference `2.2378e-7`; reciprocal and full-domain round-trip tests
pass.

M4-R PASS (2026-09-27). Strict Eq. (16)--(19) Ini-ISTA and separately tuned
synthetic diagnostics were independently accepted with documented
non-blocking observations. At M4-R completion, M5-R and all training were
still locked.

M5-R PASS (2026-09-27). The strict nine-phase Eq. (20)--(26) architecture has
exactly 18 trainable scalars (`tau[9]`, `theta[9]`), exact `RM_dagger(S)`
initialization, and literal complex shrinkage. M4-to-M5 relative differences
are `0.0` (complex64) and `1.21362588613636e-13` (complex128). All 22
repository tests pass, including CUDA, batch, representation, and all-phase
gradient checks. No optimizer/training or Hammer inference was started. Stop
for independent review; M6-R/M7-R remain locked. Independent review
subsequently accepted M5-R with no blocking scientific issues.

M6-R PASS (2026-09-27). Exactly 1000 training pairs and 200 disjoint
validation pairs were generated procedurally at 512×512 with a fixed realized
sampling rate of `0.4999198914` and achieved 20 dB complex AWGN. Adam at
`1e-4` completed all 100 epochs with batch size 8. Best validation epoch was
99; NRMSE improved from `0.6148313` untrained to `0.5302653`, PSNR improved
from `40.7205` to `42.0050` dB, and TBR improved from `3.2892` to `17.7752`
dB. All 27 tests pass. No Hammer inference occurred. Stop for independent
review; M7-R remains locked.

Independent review accepted M6-R with no blocking scientific issues. The
NRMSE improvement (`0.6148313` to `0.5302653`) is a modest baseline result,
not reproduction of the paper's exact tables. M6 ran under Python 3.11.4,
PyTorch 2.2.2, and CUDA 12.1; all regressions pass. Best-checkpoint provenance
is recorded for future inference-only use.

M7-R TECHNICAL PASS / REAL-DATA SUCCESS (2026-09-27). Strict Eq. (32)
inference used Rx1, bins 12--24, the exact accepted M6 mask and epoch-99
parameters, and independent range-dependent kernels. The Hammer remains
recognizable at the accepted range/orientation, but trained RMIST is nearly
identical to the sparse baselines and demonstrates no material learned
improvement. All 31 tests pass. Stop for independent review.

Independent review accepted M7-R as a technical PASS with an important
scientific limitation. The Hammer is recognizable with preserved
range/orientation, but RMIST50 differs from RMA50 by only about `1.1364e-6`
in complex relative norm and observed-sample consistency is essentially
identical across sparse methods. Learned real-data improvement is not
demonstrated. M8 sim-to-real diagnostics remain locked.

M8 PASS (2026-09-27). A non-invasive nine-phase trace matches ordinary RMIST
output exactly and leaves parameters unchanged. The accepted 32-sample M6
control has median alpha0 RMS `0.0137687`, while Hammer bin 17 has RMS
`3.18109e7` (`2.3104e9` times larger). Unscaled Hammer shrinkage is effectively
zero; counterfactual scale alignment restores synthetic-order phase activity.
Classification: `SCALE_MISMATCH_SUPPORTED`. Stop for independent review.

Independent review accepted M8 with no blocking implementation issues.
Scale mismatch is the dominant demonstrated mechanism behind M7 inactivity;
residual degeneracy is not the dominant phase-1 explanation. Normalization as
an image-improvement method remains unvalidated. M9 may evaluate it without
retraining only after separate authorization and is currently locked.

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

Timestamp: 2026-09-27 Asia/Shanghai
Milestone: M5-R
Task: Strict nine-phase RMIST-Net Eq. (20)--(26) architecture validation
Status: PASS — AWAITING INDEPENDENT REVIEW
Command: `PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 scripts/run_rmist_net_validation.py --output results/review/M5`
Result: Exact 18-scalar architecture; alpha0 difference 0; M4-to-M5 differences 0 (complex64) and 1.21362588613636e-13 (complex128); finite nonzero gradients in every phase; 22/22 tests pass.
Evidence: `results/review/M5/`, `docs/M5_RMIST_NET_REPORT.md`
Next action: STOP for independent review; do not begin M6-R.
Human decision required: YES

Timestamp: 2026-09-27 Asia/Shanghai
Milestone: M6-R
Task: Strict RMIST synthetic training Eq. (27)--(31)
Status: PASS — AWAITING INDEPENDENT REVIEW
Command: `PYTHONPATH=src python3 scripts/train_rmist.py --config configs/rmist_train.yaml`
Result: 100/100 epochs on 1000 training pairs; best validation epoch 99; held-out NRMSE improved from 0.6148313 to 0.5302653; all 27 tests pass.
Evidence: `results/review/M6/`, `docs/M6_RMIST_TRAINING_REPORT.md`
Next action: STOP for independent review; do not begin M7-R.
Human decision required: YES
