# M8_REVIEW_CHECKLIST.md

# M8 Sim-to-Real Diagnostic Review

Independent review verdict: **PASS**. Root cause supported:
`SCALE_MISMATCH_SUPPORTED`. Counterfactual alignment restores internal phase
dynamics but does not establish improved real-data reconstruction quality.

## Governance

- [x] Branch is `milestone/rmist-m8`.
- [x] Branch starts from accepted M7 main.
- [x] M8 contains no training/fine-tuning.
- [x] Learned tau/theta unchanged.
- [x] M6 mask/operator conventions unchanged.

## Synthetic control

- [x] Uses accepted M6 validation generator/config.
- [x] Uses deterministic 32-sample subset.
- [x] Trained model parameters exactly match M6 best epoch 99.

## Real control

- [x] Uses accepted M7 Hammer preprocessing.
- [x] Rx1 only.
- [x] Same accepted M6 mask.
- [x] Bin 17 primary trace plus aggregate 13-slice statistics.

## Scale diagnostics

- [x] Measurement magnitude stats reported.
- [x] alpha0 magnitude stats reported.
- [x] RMS/mean/median/p90/p95/p99/max reported.
- [x] Real/synthetic scale ratios reported.

## Per-phase diagnostics

For all 9 phases:

- [x] relative residual `E_res`.
- [x] relative gradient correction `E_grad`.
- [x] relative shrinkage `E_shrink`.
- [x] total update `E_update`.
- [x] cumulative divergence from RMA `E_RMA`.
- [x] theta/p50 ratio.
- [x] theta/p90 ratio.
- [x] theta/p99 ratio.
- [x] fractions below theta, 2theta, 5theta.

## Counterfactual scale sweep

- [x] `c_align` computed from data.
- [x] original `c=1` included.
- [x] sweep around `c_align` completed.
- [x] outputs rescaled back for fair comparison.
- [x] no retraining/parameter change.
- [x] interpretation is diagnostic only.

## Instrumentation correctness

- [x] traced forward output matches ordinary RMIST output exactly.
- [x] phase decomposition numerically consistent.
- [x] no optimizer/backward/train calls.
- [x] all M3-M7 regressions pass.

## Root-cause conclusion

One classification selected:

```text
[x] SCALE_MISMATCH_SUPPORTED
[ ] SCALE_MISMATCH_NOT_SUFFICIENT / BROADER_DISTRIBUTION_MISMATCH_SUPPORTED
[ ] RESIDUAL_DEGENERACY_DOMINANT
[ ] MULTIPLE_FACTORS_SUPPORTED
[ ] INCONCLUSIVE
```

- [x] classification supported quantitatively.
- [x] no untested causal claim presented as fact.

## Verdict

```text
[x] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

M8 is independently accepted. M9 scale-normalized inference-quality
evaluation remains locked pending separate explicit authorization.
