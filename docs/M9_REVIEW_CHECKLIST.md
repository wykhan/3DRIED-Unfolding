# M9_REVIEW_CHECKLIST.md

# M9 Scale-Normalized RMIST Review

## Governance

- [x] Branch is `milestone/rmist-m9`.
- [x] Branch starts from accepted M8 main.
- [ ] No training/fine-tuning occurs.
- [ ] M6 best tau/theta remain unchanged.
- [ ] M6 mask and M7 preprocessing remain unchanged.
- [ ] Normalization variants are fixed before final comparison.

## Synthetic references

- [ ] Uses accepted M6 validation generator.
- [ ] Uses same deterministic 32-sample control as M8.
- [ ] Synthetic alpha0 RMS reference regenerated.
- [ ] Synthetic alpha0 p95 reference regenerated.
- [ ] Synthetic observed-measurement RMS reference regenerated.
- [ ] Regenerated alpha0 RMS matches M8 within tolerance.

## Variants

- [ ] N0 raw/unscaled RMIST.
- [ ] N1 global alpha0-RMS matching.
- [ ] N2 per-slice alpha0-RMS matching.
- [ ] N3 per-slice alpha0-p95 matching.
- [ ] N4 per-slice observed-measurement RMS matching.
- [ ] Scale factors are finite and positive.
- [ ] Scale factors recorded for every slice/method.
- [ ] Output is de-normalized back to original amplitude units.

## Phase dynamics

For every method / slice:

- [ ] `E_grad` reported.
- [ ] `E_shrink` reported.
- [ ] `E_update` reported.
- [ ] final `E_RMA` reported.
- [ ] N0 reproduces M8 inactivity.
- [ ] Normalized variants checked for restored dynamics.
- [ ] No NaN/Inf.

## Real-data diagnostics

- [ ] Observed-sample consistency reported.
- [ ] No GT-dependent NRMSE/PSNR/TBR claimed.
- [ ] M2 is labeled a reference proxy, not GT.
- [ ] Magnitude cosine similarity to M2 reported.
- [ ] Normalized magnitude distance to M2 reported.
- [ ] Per-slice and full-volume proxy results reported.
- [ ] MIP proxy result reported.

## Range-profile preservation

- [ ] Per-slice energy profile computed for all methods.
- [ ] Profile cosine similarity to M2 reported.
- [ ] Dominant range bin reported.
- [ ] Per-slice normalization checked for artificial range equalization.

## Visual evidence

- [ ] Common ROI.
- [ ] Common dB normalization.
- [ ] MIP comparison.
- [ ] Bins 16/17/18 comparison.
- [ ] Range-profile figure.
- [ ] Hammer head assessed.
- [ ] Handle continuity assessed.
- [ ] Clutter/target loss assessed.
- [ ] No superiority claim based only on stronger network activity.

## Regression / implementation

- [ ] `c=1` wrapper reproduces ordinary RMIST.
- [ ] Synthetic reference scale matches M8.
- [ ] N0 reproduces accepted M7 result.
- [ ] No optimizer/backward/train calls.
- [ ] M3-M8 regression tests pass.
- [ ] Working tree clean after commit.

## Outcome

Select exactly one:

```text
[ ] NORMALIZATION_FEASIBLE_WITH_PROXY_BENEFIT
[ ] DYNAMICS_RESTORED_BUT_IMAGE_BENEFIT_UNPROVEN
[ ] NORMALIZATION_DEGRADES_REAL_RECONSTRUCTION
[ ] GLOBAL_AND_PER_SLICE_TRADEOFF_UNRESOLVED
[ ] INCONCLUSIVE
```

## Verdict

```text
[ ] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

After M9, STOP for independent review.
