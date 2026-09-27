# M7_RMIST_HAMMER_SPEC.md

# M7-R Real Hammer Slice-Wise RMIST Inference Specification

## 1. Scope

M7-R applies the accepted trained RMIST-Net to real single-Rx 3DRIED Hammer data according to RMIST Eq. (32).

M7-R is inference/evaluation only.

No retraining, fine-tuning, domain adaptation, self-supervision, calibration learning, or parameter modification is allowed.

If sim-to-real performance is poor, report the failure. Do not silently retrain.

## 2. Paper basis

RMIST Eq. (32) reconstructs the 3-D cube slice-wise:

\[
\hat\alpha_{3D}(:,:,r)
=
RMIST(P(S_r\odot Mask), \Phi_r, \Phi_r^\dagger).
\]

Each range slice has its own range-dependent RM kernels, while the learned
network parameters are shared.

The real-data experiments in the paper include both full-data and 50%
uniform-random sampling cases. This project currently has an accepted M6 model
trained for one fixed SR=0.5 mask. Therefore the strict first M7 inference
must use that same fixed-mask condition.

Do not claim that this M6 model is a full-data RMIST model.

## 3. Real input

Dataset:

```text
/home/superws/dataset/3DRIED/Hammer/Hammer (1)
```

Use the accepted M0-M2 pipeline:

- Rx1 only / Python index 0;
- motion correction exactly as M2;
- serpentine correction exactly as M2;
- range FFT exactly as M2;
- MATLAB range bins 12-24;
- corresponding physical ranges from M2;
- physical aperture `[z=200, x=407]`;
- RM domain `[512,512]`.

Do not change data interpretation.

## 4. Accepted trained parameters

Use M6 best-validation model, epoch 99.

Preferred source:

```text
results/m6_rmist_training/checkpoints/best_validation.pt
```

If the local checkpoint exists, load it read-only and verify it matches
committed M6 best parameters.

If it does not exist, reconstruct the inference model exactly from the
committed 18 scalar values:

```text
results/review/M6/summary_metrics.json -> best_parameters
```

This fallback is allowed because `tau[9]` and `theta[9]` are the only
trainable parameters.

Do not use epoch-100 parameters when the accepted best-validation model is
available/reconstructible.

## 5. Sampling mask mapping to real aperture

Regenerate the exact accepted M6 full-grid mask:

```text
shape: 512 x 512
seed: 20260928
target SR: 0.5
expected SHA-256:
43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2
```

Verify hash before inference.

For each real physical range-focused slice:

1. center-pad `[200,407]` to `[512,512]` using accepted M3 padding;
2. construct the physical-support mask corresponding to the padded aperture;
3. define:

\[
Mask_{eff} = Mask_{M6}\odot Mask_{physical-support};
\]

4. define the M7 input:

\[
S_{masked}=Mask_{eff}\odot P(S_r).
\]

Record the realized sampling rate inside the actual physical support.

Applying the cropped M6 mask to the physical aperture before padding is
mathematically equivalent and allowed if tested.

## 6. Slice-wise inference

For each M2 range bin \(r\):

- compute the exact physical `range_m`;
- instantiate/use accepted 3DRIED `RMISTKernel`;
- use the same accepted learned `tau/theta`;
- pass the corresponding `range_m` into RMIST-Net;
- reconstruct independently.

Do not batch different ranges through a single scalar `range_m` call.

Stack output as:

```text
[range_y, vertical_z, horizontal_x]
```

matching M2 volume convention.

## 7. Required comparison methods

For the same real Hammer data, produce:

### A. Full-data M2 RMA reference

Use existing accepted:

```text
results/rma_hammer/volume.npy
```

This is a reference visualization, not same-measurement sparse baseline.

### B. 50% masked RMA / RM_dagger

Using the exact effective mask and each range-dependent inverse kernel:

\[
\hat\alpha_{RMA50}=RM^\dagger(S_{masked}).
\]

### C. 50% masked Ini-ISTA

Use accepted Eq. (19):

```text
tau=1e-4
theta=1e-3
iterations=9
```

on the same masked real measurement.

### D. 50% masked trained RMIST-Net

Use accepted M6 best parameters.

Do not use the M4 tuned diagnostic parameters as a formal real baseline,
because those values were tuned on a different small synthetic protocol.

## 8. No voxel ground truth

Hammer has no voxel-level GT.

Do NOT compute GT-dependent NRMSE/PSNR/TBR for real Hammer.

Required real-data diagnostics:

### Measurement consistency on observed samples

For estimate \(\hat\alpha\):

\[
E_{obs}
=
\frac{
\|Mask_{eff}\odot(RM(\hat\alpha)-P(S_r))\|_2
}{
\|Mask_{eff}\odot P(S_r)\|_2+\epsilon
}.
\]

Report per-slice and aggregate values for:

- RMA50;
- Ini-ISTA50;
- trained RMIST50.

This metric is diagnostic; lower residual does not by itself prove better image quality.

### Optional no-reference image diagnostics

Entropy and contrast may be reported only if their exact definitions are
documented and they are labeled non-paper diagnostics.

## 9. Qualitative evaluation

Use identical visualization normalization/ROI for fair comparison.

Required outputs:

- x-z MIP for each method;
- representative range slices, including target peak region;
- side-by-side comparison figure;
- 3-D threshold visualization for trained RMIST and M2 reference;
- normalized dB images.

Assess:

- Hammer recognizability;
- head/handle continuity;
- artifact/clutter suppression;
- plausible range;
- orientation consistency with accepted M2 convention;
- catastrophic sim-to-real failure or not.

Do not claim superiority if images do not support it.

## 10. Required artifact structure

Create:

```text
docs/M7_RMIST_HAMMER_REPORT.md
configs/rmist_hammer_inference.yaml
scripts/run_rmist_hammer.py
tests/test_rmist_hammer.py

results/review/M7/
    summary_metrics.json
    checkpoint_provenance.json
    mask_summary.json
    per_slice_consistency.csv
    comparison_mip.png
    representative_slices.png
    parameter_values.json
```

Large complex volumes may be saved under ignored paths:

```text
results/m7_rmist_hammer/
```

Do not commit large `.npy` volumes.

## 11. Preflight gates

Before full real inference verify:

1. selected Rx is still Rx1;
2. range preprocessing matches M2;
3. M6 mask hash matches exactly;
4. effective physical-support sampling rate is recorded;
5. best model parameters match M6 accepted values;
6. one Hammer slice runs with finite outputs;
7. M3-M6 regression tests still pass;
8. no training/optimizer exists in the M7 script.

## 12. PASS / outcome classification

M7-R should not be forced to "PASS" merely because inference runs.

Classify outcome:

```text
TECHNICAL PASS:
pipeline executes correctly, outputs finite/reproducible.

REAL-DATA SUCCESS:
Hammer remains recognizable and no catastrophic sim-to-real failure occurs.

REAL-DATA FAILURE:
pipeline is technically correct but trained synthetic RMIST fails to produce
a meaningful Hammer reconstruction.
```

A technically correct real-data failure is scientifically useful and must not
trigger silent retraining in M7.

## 13. Stop condition

After M7 real inference and report:

STOP for independent review.

Do not begin domain adaptation, new training distributions, calibration, or
sparse-mask redesign without a new milestone.
