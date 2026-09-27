# M7_REVIEW_CHECKLIST.md

# M7-R Real Hammer Inference Review

## Governance

- [ ] Branch is `milestone/rmist-m7`.
- [ ] Branch starts from accepted M6 main.
- [ ] No training/fine-tuning/domain adaptation occurred.
- [ ] Rx1-only policy preserved.
- [ ] Raw data unchanged.

## Model provenance

- [ ] M6 best epoch is 99.
- [ ] Best tau/theta match committed M6 values.
- [ ] Checkpoint hash verified if checkpoint exists.
- [ ] Fallback-from-committed-parameters, if used, documented.

## Mask

- [ ] M6 mask seed `20260928`.
- [ ] Full-grid mask hash matches accepted M6 hash.
- [ ] Effective physical-support mask defined correctly.
- [ ] Realized physical-aperture sampling rate recorded.
- [ ] Same mask used for all compared sparse methods.

## Data/preprocessing

- [ ] Rx1 / Python 0.
- [ ] Motion correction unchanged.
- [ ] Serpentine correction unchanged.
- [ ] Range FFT unchanged.
- [ ] MATLAB bins 12-24.
- [ ] Padding `[200,407] -> [512,512]` unchanged.

## Eq. (32) slice-wise inference

- [ ] Each slice uses its own `range_m`.
- [ ] Each slice uses corresponding RM / RM_dagger kernels.
- [ ] Learned tau/theta shared across slices.
- [ ] Different ranges are not incorrectly batched through one scalar range.

## Baselines

- [ ] Existing full-data M2 RMA reference included.
- [ ] RMA50 uses same masked measurement.
- [ ] Ini-ISTA50 uses same masked measurement, 9 steps, paper initialization.
- [ ] Trained RMIST50 uses M6 best parameters.
- [ ] M4 tuned small-scene parameters are not presented as formal real baseline.

## Real-data metrics

- [ ] No GT-dependent NRMSE/PSNR/TBR claimed.
- [ ] Observed-sample consistency defined and reported per slice.
- [ ] Aggregate consistency reported.
- [ ] Any entropy/contrast metric is explicitly defined and labeled non-paper.

## Visual evidence

- [ ] Same normalization/ROI across methods.
- [ ] MIP comparison committed.
- [ ] Representative slices committed.
- [ ] 3-D visualization saved.
- [ ] Hammer recognizability assessed.
- [ ] Sim-to-real failure/success stated without embellishment.

## Reproducibility

- [ ] Config committed.
- [ ] Lightweight review metrics committed.
- [ ] Large volumes/checkpoints not committed.
- [ ] All repository tests pass.

## Outcome

```text
Technical pipeline:
[ ] PASS
[ ] FAIL

Real-data outcome:
[ ] SUCCESS
[ ] PARTIAL / DEGRADED
[ ] FAILURE
```

## Next-step gate

After M7, STOP. Any domain adaptation/retraining/calibration work requires a new milestone and human authorization.
