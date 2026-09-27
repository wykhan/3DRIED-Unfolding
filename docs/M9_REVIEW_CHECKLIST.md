# M9_REVIEW_CHECKLIST.md

# M9 Scale-Normalized RMIST Review

## Governance

- [x] Branch is `milestone/rmist-m9`.
- [x] Branch starts from accepted M8 main.
- [x] No training/fine-tuning occurs.
- [x] M6 best tau/theta remain unchanged.
- [x] M6 mask and M7 preprocessing remain unchanged.
- [x] Normalization variants are fixed before final comparison.

## Synthetic references

- [x] Uses accepted M6 validation generator.
- [x] Uses same deterministic 32-sample control as M8.
- [x] Synthetic alpha0 RMS reference regenerated.
- [x] Synthetic alpha0 p95 reference regenerated.
- [x] Synthetic observed-measurement RMS reference regenerated.
- [x] Regenerated alpha0 RMS matches M8 within tolerance.

## Variants

- [x] N0 raw/unscaled RMIST.
- [x] N1 global alpha0-RMS matching.
- [x] N2 per-slice alpha0-RMS matching.
- [x] N3 per-slice alpha0-p95 matching.
- [x] N4 per-slice observed-measurement RMS matching.
- [x] Scale factors are finite and positive.
- [x] Scale factors recorded for every slice/method.
- [x] Output is de-normalized back to original amplitude units.

## Phase dynamics

For every method / slice:

- [x] `E_grad` reported.
- [x] `E_shrink` reported.
- [x] `E_update` reported.
- [x] final `E_RMA` reported.
- [x] N0 reproduces M8 inactivity.
- [x] Normalized variants checked for restored dynamics.
- [x] No NaN/Inf.

## Real-data diagnostics

- [x] Observed-sample consistency reported.
- [x] No GT-dependent NRMSE/PSNR/TBR claimed.
- [x] M2 is labeled a reference proxy, not GT.
- [x] Magnitude cosine similarity to M2 reported.
- [x] Normalized magnitude distance to M2 reported.
- [x] Per-slice and full-volume proxy results reported.
- [x] MIP proxy result reported.

## Range-profile preservation

- [x] Per-slice energy profile computed for all methods.
- [x] Profile cosine similarity to M2 reported.
- [x] Dominant range bin reported.
- [x] Per-slice normalization checked for artificial range equalization.

## Visual evidence

- [x] Common ROI.
- [x] Common dB normalization.
- [x] MIP comparison.
- [x] Bins 16/17/18 comparison.
- [x] Range-profile figure.
- [x] Hammer head assessed.
- [x] Handle continuity assessed.
- [x] Clutter/target loss assessed.
- [x] No superiority claim based only on stronger network activity.

## Regression / implementation

- [x] `c=1` wrapper reproduces ordinary RMIST.
- [x] Synthetic reference scale matches M8.
- [x] N0 reproduces accepted M7 result.
- [x] No optimizer/backward/train calls.
- [x] M3-M8 regression tests pass (40/40 full-suite tests).
- [x] Working tree clean after commit.

## Outcome

Select exactly one:

```text
[ ] NORMALIZATION_FEASIBLE_WITH_PROXY_BENEFIT
[ ] DYNAMICS_RESTORED_BUT_IMAGE_BENEFIT_UNPROVEN
[x] NORMALIZATION_DEGRADES_REAL_RECONSTRUCTION
[ ] GLOBAL_AND_PER_SLICE_TRADEOFF_UNRESOLVED
[ ] INCONCLUSIVE
```

## Verdict

```text
[x] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

External review: **PASS — independently reviewed**. Scientific outcome remains `NORMALIZATION_DEGRADES_REAL_RECONSTRUCTION` under the tested proxy framework. M2 is not ground truth. No production normalization is accepted.

After M9, the milestone is closed.
