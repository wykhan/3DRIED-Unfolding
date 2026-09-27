# RMIST_M7_START_PROMPT.md

# Codex task: M7-R real Hammer slice-wise RMIST inference

The human explicitly authorizes M7-R after accepted M6-R is merged into main.

M7-R is inference/evaluation only.

Do NOT train, fine-tune, adapt, calibrate, or modify learned parameters.
Do NOT create an optimizer.
Do NOT change the M6 training distribution.

## 1. Verify branch

The human should create:

```text
milestone/rmist-m7
```

from accepted `main` and commit M7 specification files before launching.

Run:

```bash
git status
git branch --show-current
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git log --oneline --decorate -12
```

Expected:

```text
current branch = milestone/rmist-m7
```

If unexpected, STOP.

## 2. Read source-of-truth

Read completely:

- `AGENTS.md`
- `RESEARCH_SPEC.md`
- `PLAN.md`
- `STATUS.md`
- `DECISIONS.md`
- `docs/RMIST_REPRO_SPEC.md`
- `docs/M6_RMIST_TRAINING_REPORT.md`
- `docs/M7_RMIST_HAMMER_SPEC.md`
- `docs/M7_REVIEW_CHECKLIST.md`
- `docs/M2_RMA_HAMMER_REPORT.md`

Inspect accepted M2-M6 implementation.

## 3. Unlock only M7-R

Update governance:

```text
M6-R: independently reviewed PASS
M7-R: UNLOCKED — real inference only
```

Commit governance unlock before implementation.

## 4. Resolve accepted M6 model

Preferred:

```text
results/m6_rmist_training/checkpoints/best_validation.pt
```

Verify best epoch 99 and parameter equality with committed M6 summary.

If checkpoint is absent, reconstruct model exactly from:

```text
results/review/M6/summary_metrics.json -> best_parameters
```

Record provenance.

Never use different tau/theta.

## 5. Recreate exact M6 mask

Regenerate 512x512 mask with seed 20260928.

Verify SHA-256:

```text
43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2
```

If mismatch, STOP.

Map it to the accepted physical Hammer support and record realized effective SR.

## 6. Load Hammer with accepted M2 preprocessing

Use:

```text
dataset path:
/home/superws/dataset/3DRIED/Hammer/Hammer (1)

Rx:
MATLAB Rx1 / Python index 0

range bins:
12..24
```

Use accepted motion/serpentine/range-FFT conventions unchanged.

## 7. Preflight one slice

For one representative slice:

- pad physical echo;
- apply effective M6 mask;
- run RMA50;
- run Ini-ISTA50;
- run trained RMIST50;
- verify all outputs finite;
- verify observed-sample residual computation finite.

Run all repository tests before full cube.

## 8. Full slice-wise Eq. (32)

For each range bin:

1. compute exact `range_m`;
2. build/use corresponding RM kernels;
3. use same accepted learned tau/theta;
4. reconstruct independently.

Do not process multiple different ranges with one scalar range value.

Stack volume `[range,z,x]`.

## 9. Comparison set

Produce:

- full-data M2 RMA reference;
- 50% masked RMA;
- 50% masked 9-step Ini-ISTA;
- 50% masked trained RMIST.

Do not use M4 tuned diagnostic as formal Hammer baseline.

## 10. Metrics

For sparse methods compute observed-sample consistency:

\[
E_{obs}
=
\frac{\|Mask_{eff}\odot(RM(\hat\alpha)-P(S_r))\|_2}
{\|Mask_{eff}\odot P(S_r)\|_2+\epsilon}.
\]

Save per-slice CSV and aggregate summary.

Do not compute GT-dependent real-data NRMSE/PSNR/TBR.

## 11. Visual outputs

Use identical ROI and normalization across compared methods.

Create:

```text
results/review/M7/comparison_mip.png
results/review/M7/representative_slices.png
```

Also save large volumes/3-D visualizations under ignored result paths.

Assess Hammer head, handle, artifacts, range, orientation, and sim-to-real behavior.

## 12. Classification

Do not force success.

Report separately:

```text
technical_pipeline: PASS/FAIL
real_data_outcome: SUCCESS / PARTIAL_DEGRADED / FAILURE
```

If real-data quality is poor but pipeline is correct, report `FAILURE` or
`PARTIAL_DEGRADED` and STOP. Do not retrain.

## 13. Required files

Create:

```text
docs/M7_RMIST_HAMMER_REPORT.md
configs/rmist_hammer_inference.yaml
scripts/run_rmist_hammer.py
tests/test_rmist_hammer.py
results/review/M7/
```

Commit lightweight review artifacts only.

## 14. Final tests and stop

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Push `milestone/rmist-m7`.

Do not merge.

Return:

```text
M7-R technical pipeline:
real-data outcome:
commit:

model source:
best epoch:
checkpoint hash / fallback:

mask hash:
effective physical SR:

range bins:
range span:

aggregate observed residual:
RMA50:
Ini-ISTA50:
RMIST50:

Hammer recognizability:
artifact/clutter assessment:
range/orientation assessment:

tests:
unresolved issues:

STOP.
```
