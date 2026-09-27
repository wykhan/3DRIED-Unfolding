# M8_REVIEW_CHECKLIST.md

# M8 Sim-to-Real Diagnostic Review

## Governance

- [x] Branch is `milestone/rmist-m8`.
- [x] Branch starts from accepted M7 main.
- [ ] M8 contains no training/fine-tuning.
- [ ] Learned tau/theta unchanged.
- [ ] M6 mask/operator conventions unchanged.

## Synthetic control

- [ ] Uses accepted M6 validation generator/config.
- [ ] Uses deterministic sample subset.
- [ ] Trained model parameters exactly match M6 best epoch 99.

## Real control

- [ ] Uses accepted M7 Hammer preprocessing.
- [ ] Rx1 only.
- [ ] Same accepted M6 mask.
- [ ] Bin 17 primary trace plus aggregate 13-slice statistics.

## Scale diagnostics

- [ ] Measurement magnitude stats reported.
- [ ] alpha0 magnitude stats reported.
- [ ] RMS/mean/median/p90/p95/p99/max reported.
- [ ] Real/synthetic scale ratios reported.

## Per-phase diagnostics

For all 9 phases:

- [ ] relative residual `E_res`.
- [ ] relative gradient correction `E_grad`.
- [ ] relative shrinkage `E_shrink`.
- [ ] total update `E_update`.
- [ ] cumulative divergence from RMA `E_RMA`.
- [ ] theta/p50 ratio.
- [ ] theta/p90 ratio.
- [ ] theta/p99 ratio.
- [ ] fractions below theta, 2theta, 5theta.

## Counterfactual scale sweep

- [ ] `c_align` computed from data.
- [ ] original `c=1` included.
- [ ] sweep around `c_align` completed.
- [ ] outputs rescaled back for fair comparison.
- [ ] no retraining/parameter change.
- [ ] interpretation is diagnostic only.

## Instrumentation correctness

- [ ] traced forward output matches ordinary RMIST output.
- [ ] phase decomposition numerically consistent.
- [ ] no optimizer/backward/train calls.
- [ ] all M3-M7 regressions pass.

## Root-cause conclusion

One classification selected:

```text
[ ] SCALE_MISMATCH_SUPPORTED
[ ] SCALE_MISMATCH_NOT_SUFFICIENT / BROADER_DISTRIBUTION_MISMATCH_SUPPORTED
[ ] RESIDUAL_DEGENERACY_DOMINANT
[ ] MULTIPLE_FACTORS_SUPPORTED
[ ] INCONCLUSIVE
```

- [ ] classification supported quantitatively.
- [ ] no untested causal claim presented as fact.

## Verdict

```text
[ ] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

After M8, STOP for independent review.
