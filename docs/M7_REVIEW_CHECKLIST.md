# M7_REVIEW_CHECKLIST.md

# M7-R Real Hammer Inference Review

Independent review verdict: **TECHNICAL PASS WITH IMPORTANT SCIENTIFIC
LIMITATION**. The Hammer remains recognizable with preserved range/orientation,
but learned improvement over RMA50/Ini-ISTA50 is **NOT DEMONSTRATED**.

## Governance

- [x] Branch is `milestone/rmist-m7`.
- [x] Branch starts from accepted M6 main.
- [x] No training/fine-tuning/domain adaptation occurred.
- [x] Rx1-only policy preserved.
- [x] Raw data unchanged.

## Model provenance

- [x] M6 best epoch is 99.
- [x] Best tau/theta match committed M6 values.
- [x] Checkpoint hash verified if checkpoint exists.
- [x] Fallback-from-committed-parameters, if used, documented (not needed in this run).

## Mask

- [x] M6 mask seed `20260928`.
- [x] Full-grid mask hash matches accepted M6 hash.
- [x] Effective physical-support mask defined correctly.
- [x] Realized physical-aperture sampling rate recorded.
- [x] Same mask used for all compared sparse methods.

## Data/preprocessing

- [x] Rx1 / Python 0.
- [x] Motion correction unchanged.
- [x] Serpentine correction unchanged.
- [x] Range FFT unchanged.
- [x] MATLAB bins 12-24.
- [x] Padding `[200,407] -> [512,512]` unchanged.

## Eq. (32) slice-wise inference

- [x] Each slice uses its own `range_m`.
- [x] Each slice uses corresponding RM / RM_dagger kernels.
- [x] Learned tau/theta shared across slices.
- [x] Different ranges are not incorrectly batched through one scalar range.

## Baselines

- [x] Existing full-data M2 RMA reference included.
- [x] RMA50 uses same masked measurement.
- [x] Ini-ISTA50 uses same masked measurement, 9 steps, paper initialization.
- [x] Trained RMIST50 uses M6 best parameters.
- [x] M4 tuned small-scene parameters are not presented as formal real baseline.

## Real-data metrics

- [x] No GT-dependent NRMSE/PSNR/TBR claimed.
- [x] Observed-sample consistency defined and reported per slice.
- [x] Aggregate consistency reported.
- [x] No entropy/contrast metric reported.

## Visual evidence

- [x] Same normalization/ROI across methods.
- [x] MIP comparison committed.
- [x] Representative slices committed.
- [x] 3-D visualization saved.
- [x] Hammer recognizability assessed.
- [x] Sim-to-real failure/success stated without embellishment.

## Reproducibility

- [x] Config committed.
- [x] Lightweight review metrics committed.
- [x] Large volumes/checkpoints not committed.
- [x] All repository tests pass (31/31).

## Outcome

```text
Technical pipeline:
[x] PASS
[ ] FAIL

Real-data outcome:
[x] SUCCESS
[ ] PARTIAL / DEGRADED
[ ] FAILURE
```

## Next-step gate

M7-R is independently accepted. M8 sim-to-real diagnostics remain locked.
Any diagnostics, domain adaptation, retraining, or calibration work requires a
new milestone and explicit human authorization.
