# RMIST_M8_START_PROMPT.md

# Codex task: diagnose why RMIST learned gain vanishes on real Hammer

The human explicitly authorizes M8 after accepted M7 is merged into main.

M8 is diagnostic only.

Do NOT train.
Do NOT fine-tune.
Do NOT change tau/theta.
Do NOT add normalization to the production pipeline.
Do NOT modify M6/M7 scientific outputs.
Do NOT start a new model.

## 1. Verify branch

The human should create:

```text
milestone/rmist-m8
```

from accepted `main` and commit M8 specification files before launching.

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
current branch = milestone/rmist-m8
```

If unexpected, STOP.

## 2. Read source-of-truth

Read completely:

- `AGENTS.md`
- `RESEARCH_SPEC.md`
- `PLAN.md`
- `STATUS.md`
- `DECISIONS.md`
- `docs/M6_RMIST_TRAINING_REPORT.md`
- `docs/M7_RMIST_HAMMER_REPORT.md`
- `docs/M8_SIM2REAL_DIAGNOSTIC_SPEC.md`
- `docs/M8_REVIEW_CHECKLIST.md`

Inspect accepted M3-M7 source.

## 3. Unlock M8 diagnostic only

Update governance:

```text
M7-R: independently reviewed technical PASS
M8: UNLOCKED — diagnostic only
```

State explicitly:

```text
No training, fine-tuning, normalization fix, domain adaptation, or parameter updates authorized.
```

Commit governance unlock first.

## 4. Build non-invasive tracing instrumentation

Implement a diagnostic/tracing path that records every RMIST phase without
changing the accepted `RMISTNet.forward()` result.

For each phase record:

```text
alpha_before
residual v_t
gradient correction tau_t * RM_dagger(v_t)
rough r_t
alpha_after
```

Prefer hooks/helper functions outside the accepted model rather than changing
production semantics.

Blocking test:

```text
traced_output == ordinary_RMIST_output
```

within accepted numerical tolerance.

## 5. Synthetic diagnostic control

Use accepted M6 validation generator and accepted best epoch-99 parameters.

Recommended deterministic subset:

```text
32 samples
```

Use exactly the accepted M6:

```text
512x512
range 0.5 m
SR=0.5
mask seed 20260928
20 dB AWGN
```

Compute scale and phase metrics from the M8 spec.

## 6. Hammer diagnostics

Use accepted M7 pipeline.

Primary detailed slice:

```text
MATLAB bin 17
range 0.6143103953 m
```

Also compute aggregate scale/phase statistics over all bins 12-24.

No parameter changes.

## 7. Required per-phase metrics

For each of 9 phases compute:

```text
E_res    = ||v_t|| / ||S||
E_grad   = ||tau_t RM_dagger(v_t)|| / ||alpha_t||
E_shrink = ||alpha_{t+1} - r_t|| / ||r_t||
E_update = ||alpha_{t+1} - alpha_t|| / ||alpha_t||
E_RMA    = ||alpha_t - alpha_0|| / ||alpha_0||
```

Also compute:

```text
theta / p50(|r_t|)
theta / p90(|r_t|)
theta / p99(|r_t|)
fraction |r_t| <= theta
fraction |r_t| <= 2 theta
fraction |r_t| <= 5 theta
```

Use epsilon only for numerical safety and document it.

## 8. Scale statistics

For masked measurement and alpha0, synthetic vs real, report:

```text
RMS
mean
median
p90
p95
p99
max
```

Compute real/synthetic ratios.

Do not conclude scale mismatch merely from max values.

## 9. Counterfactual scale sweep

Compute:

```text
c_align = median synthetic alpha0 RMS / real Hammer bin17 alpha0 RMS
```

Evaluate fixed trained model on Hammer bin17 for:

```text
c=1
0.25*c_align
0.5*c_align
1.0*c_align
2.0*c_align
4.0*c_align
```

For each run:

- input is `c*S_real`;
- output is divided by `c` before comparison;
- record phase activity;
- record relative difference from RMA initialization.

This is diagnostic only.

Do not select a "best" scale for deployment.

## 10. Root-cause classification

Use the M8 specification categories:

```text
SCALE_MISMATCH_SUPPORTED
SCALE_MISMATCH_NOT_SUFFICIENT / BROADER_DISTRIBUTION_MISMATCH_SUPPORTED
RESIDUAL_DEGENERACY_DOMINANT
MULTIPLE_FACTORS_SUPPORTED
INCONCLUSIVE
```

Support the selected classification with quantitative evidence.

Do not overclaim causality.

## 11. Required files

Create:

```text
docs/M8_SIM2REAL_DIAGNOSTIC_REPORT.md
configs/rmist_sim2real_diagnostic.yaml
scripts/diagnose_rmist_sim2real.py
tests/test_rmist_sim2real_diagnostic.py
results/review/M8/
```

Commit lightweight artifacts:

```text
summary_metrics.json
scale_statistics.json
phase_metrics_synthetic.csv
phase_metrics_hammer.csv
scale_sweep.csv
scale_comparison.png
phase_activity.png
threshold_activity.png
```

Do not commit large tensors.

## 12. Final validation

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Apply all M8 review gates.

Push `milestone/rmist-m8`.

Do not merge.

Return:

```text
M8 diagnostic: PASS/FAIL
commit:

root-cause classification:

synthetic alpha0 RMS median:
Hammer bin17 alpha0 RMS:
real/synthetic RMS ratio:
c_align:

Hammer phase activity summary:
max E_grad:
max E_shrink:
max E_update:
final E_RMA:

Synthetic phase activity summary:
max/median E_grad:
max/median E_shrink:
max/median E_update:
final E_RMA:

scale sweep finding:

instrumentation-output regression:
tests:

scientific interpretation:
unresolved issues:

STOP.
```
