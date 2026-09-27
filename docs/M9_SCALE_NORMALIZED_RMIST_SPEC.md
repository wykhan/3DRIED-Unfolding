# M9_SCALE_NORMALIZED_RMIST_SPEC.md

# M9 — Scale-Normalized RMIST Real-Data Feasibility

## 1. Scientific goal

M8 established:

\[
\mathrm{RMS}(|\alpha_{0,\mathrm{Hammer}}|)
\approx 2.31\times10^9
\,
\mathrm{RMS}(|\alpha_{0,\mathrm{synthetic}}|)
\]

on the primary Hammer slice, causing the learned absolute thresholds to become
negligible on real data.

M9 asks a narrower next question:

\[
\boxed{
\text{Can a deterministic, unsupervised scale normalization restore RMIST
activity on Hammer without degrading real-data reconstruction?}
}
\]

M9 is **not** retraining.

M9 must not:

- change learned `tau_t` or `theta_t`;
- fine-tune on Hammer;
- train a scale estimator;
- use Hammer labels;
- modify RM/RM_dagger;
- redesign the mask;
- introduce a new neural network.

M9 evaluates inference-time scale normalization only.

---

## 2. Why normalization is scientifically motivated

RMIST uses absolute learned thresholds `theta_t`.

If input/reconstruction amplitude changes by a large factor, the same
`theta_t` changes its relative meaning.

The RMIST paper states that synthetic training amplitudes are set with RMA
reconstruction results as references and chosen to be of the same order of
magnitude. The strict M6 baseline instead used `Uniform(0.5,1.0)` as a
paper-unspecified implementation choice.

M8 showed that matching Hammer initialization scale to the synthetic scale
restores strong phase activity.

M9 tests whether that mechanism can be converted into a reproducible
inference wrapper.

---

## 3. Normalized inference formulation

For a real range slice \(S_r\), estimate a positive scale statistic:

\[
s_r = g(S_r).
\]

Let the corresponding accepted synthetic reference statistic be:

\[
s_{\rm ref}.
\]

Define:

\[
c_r = \frac{s_{\rm ref}}{s_r+\epsilon}.
\]

Normalize the measurement:

\[
\tilde S_r = c_r S_r.
\]

Run the fixed accepted RMIST model:

\[
\tilde\alpha_r
=
RMIST(\tilde S_r;\tau^\star,\theta^\star).
\]

Return to the original amplitude domain:

\[
\boxed{
\hat\alpha_r = \frac{\tilde\alpha_r}{c_r}.
}
\]

No network parameter changes.

---

## 4. Accepted controls

Use:

- accepted M6 epoch-99 `tau[9]`, `theta[9]`;
- accepted M6 mask, seed `20260928`;
- accepted M7 Rx1 Hammer preprocessing;
- MATLAB range bins 12–24;
- accepted range-dependent RM/RM_dagger kernels;
- accepted 50% sparse condition.

Reference methods:

1. full-data M2 RMA — reference only;
2. RMA50;
3. Ini-ISTA50;
4. raw/unscaled RMIST50;
5. normalized RMIST variants defined below.

Do not use M4 tuned small-scene parameters as a formal real baseline.

---

## 5. Synthetic reference statistics

Do not hard-code only one M8 number when the required reference statistic can
be reproduced from the accepted M6 validation generator.

Use the same deterministic 32-sample synthetic control as M8.

For each normalization statistic, compute the median synthetic reference.

Required reference statistics:

### A. alpha0 RMS

\[
s_{\rm ref}^{RMS}
=
\mathrm{median}_{i}
\left[
RMS\left(
|RM^\dagger(S_i)|
\right)
\right].
\]

Expected near the M8 value:

```text
~0.01376872018
```

### B. alpha0 p95

\[
s_{\rm ref}^{P95}
=
\mathrm{median}_{i}
\left[
P_{95}\left(
|RM^\dagger(S_i)|
\right)
\right].
\]

### C. observed-measurement RMS

\[
s_{\rm ref}^{meas}
=
\mathrm{median}_{i}
\left[
RMS(|S_i| \text{ on observed mask support})
\right].
\]

All values must be regenerated and recorded.

---

## 6. Normalization variants to compare

### N0 — No normalization

Accepted M7 trained RMIST50.

This is the fixed baseline.

### N1 — Global alpha0-RMS matching

For each Hammer slice compute:

\[
s_r=RMS(|RM^\dagger(S_r)|).
\]

Define one global real-data scale:

\[
s_{\rm real,global}
=
\mathrm{median}_{r=12}^{24}s_r.
\]

Use the same scale factor for all slices:

\[
c_{\rm global}
=
\frac{s_{\rm ref}^{RMS}}
{s_{\rm real,global}+\epsilon}.
\]

Purpose:

- restores approximate training scale;
- preserves relative amplitude differences between range slices.

### N2 — Per-slice alpha0-RMS matching

For each slice:

\[
c_r=
\frac{s_{\rm ref}^{RMS}}
{RMS(|RM^\dagger(S_r)|)+\epsilon}.
\]

Purpose:

- strongest direct scale alignment;
- may distort inter-slice amplitude relationships.

### N3 — Per-slice robust alpha0-p95 matching

For each slice:

\[
c_r=
\frac{s_{\rm ref}^{P95}}
{P_{95}(|RM^\dagger(S_r)|)+\epsilon}.
\]

Purpose:

- less sensitive than RMS to a few strong scattering peaks.

### N4 — Per-slice observed-measurement RMS matching

For each slice:

\[
c_r=
\frac{s_{\rm ref}^{meas}}
{RMS(|S_r|_{\rm observed})+\epsilon}.
\]

Purpose:

- scale estimated directly in measurement space;
- avoids making the estimator depend on RMA initialization statistics.

All variants are deterministic and use no GT.

---

## 7. Safety bounds

M9 must record scale factors before inference.

Do not silently clip them.

If a scale factor is non-finite or non-positive, STOP.

If any variant produces values outside a very broad numerical sanity range:

```text
1e-14 <= c <= 1e-4
```

do not automatically reject it, but record and inspect before proceeding.

The expected M8 alignment is near `1e-10`.

---

## 8. Required phase-dynamics verification

For each normalization variant and each range slice, compute:

\[
E^{grad}_t,\;
E^{shrink}_t,\;
E^{update}_t,\;
E^{RMA}_t
\]

using the accepted M8 definitions.

M9 must verify:

1. N0 reproduces accepted M7/M8 inactivity;
2. normalized variants restore nontrivial phase dynamics;
3. phase activity does not become non-finite.

A normalized method is not considered useful merely because it changes the
output strongly.

---

## 9. Real-data quality evaluation without GT

Hammer has no voxel ground truth.

Do not claim true NRMSE/PSNR/TBR.

Use three distinct evidence classes.

### 9.1 Observed-sample consistency

Use accepted M7:

\[
E_{obs}
=
\frac{
\|Mask_{eff}\odot(RM(\hat\alpha)-S)\|_2
}{
\|Mask_{eff}\odot S\|_2+\epsilon
}.
\]

Report per slice and aggregate.

Interpret only as measurement consistency.

### 9.2 Full-data M2 reference-agreement proxies

M2 full-data RMA is **not ground truth**.

It may be used only as a reference proxy.

Within the exact common ROI, compute:

#### Magnitude cosine similarity

\[
C_{\rm mag}
=
\frac{
\langle |\hat\alpha|,|\alpha_{M2}|\rangle
}{
\||\hat\alpha|\|_2
\,
\||\alpha_{M2}|\|_2+\epsilon
}.
\]

Compute:

- per-slice mean;
- full-volume value;
- x-z MIP value.

Higher means closer morphology to the full-data RMA reference, not closer to
unknown truth.

#### Normalized magnitude distance

Normalize each magnitude volume to unit L2 norm:

\[
D_{\rm mag}
=
\left\|
\frac{|\hat\alpha|}{\||\hat\alpha|\|_2}
-
\frac{|\alpha_{M2}|}{\||\alpha_{M2}|\|_2}
\right\|_2.
\]

Lower is closer to M2 morphology.

### 9.3 Range-profile preservation

For each method define per-slice energy:

\[
p_r=\|\hat\alpha(:,:,r)\|_2.
\]

Normalize the 13-element profile and compare with M2:

- cosine similarity;
- peak-bin agreement.

This specifically detects whether per-slice normalization destroys relative
range amplitude structure.

---

## 10. Visual evaluation

Use exactly the same:

- ROI;
- orientation;
- common dB scale;
- range bins;
- display conventions

as M7.

Required comparisons:

- M2 full-data;
- RMA50;
- raw RMIST50;
- N1 global RMS;
- N2 per-slice RMS;
- N3 per-slice p95;
- N4 measurement RMS.

Create:

1. x-z MIP comparison;
2. representative bins 16, 17, 18;
3. range-energy profile;
4. optional 3-D threshold views.

Assess:

- hammer head preservation;
- handle continuity;
- false clutter;
- target loss;
- range peak;
- slice-to-slice amplitude distortion.

No method should be called "better" solely from phase activity.

---

## 11. Primary feasibility criteria

M9 must distinguish two separate outcomes:

### A. DYNAMICS RESTORED

For a normalized variant:

- median final `E_RMA` across Hammer slices is at least `0.1`, OR
- median max `E_shrink` is at least `0.05`.

This shows normalization activates the learned network.

### B. REAL-DATA PROXY BENEFIT

A variant may be called **proxy-beneficial** only if all are true:

1. dynamics restored;
2. M2-reference magnitude cosine similarity improves over raw RMIST50;
3. M2-reference normalized magnitude distance does not worsen;
4. range-profile cosine similarity does not materially degrade;
5. dominant range remains consistent with accepted M2 bin 17;
6. observed-sample consistency remains finite and is not catastrophically worse;
7. visual inspection does not show obvious target destruction.

These are proxy criteria, not GT proof.

---

## 12. Comparison between global and per-slice normalization

This comparison is essential.

Expected trade-off:

### Global normalization

Potential advantage:

- preserves physical inter-slice amplitude relationships.

Potential disadvantage:

- may under-normalize or over-normalize slices with very different amplitude.

### Per-slice normalization

Potential advantage:

- aligns every slice to learned threshold scale.

Potential disadvantage:

- may artificially equalize weak and strong range slices.

M9 must explicitly evaluate this trade-off rather than assuming per-slice is
superior.

---

## 13. No parameter tuning on Hammer

M9 may compare the four predefined normalization variants.

It must NOT:

- tune arbitrary exponents;
- optimize a scale factor against M2;
- search hundreds of scale constants;
- choose scale based on visual preference and then hide alternatives.

The normalization rules must be fixed before viewing final method results.

---

## 14. Required files

Create:

```text
docs/M9_SCALE_NORMALIZED_RMIST_REPORT.md
configs/rmist_scale_normalized.yaml
scripts/run_rmist_scale_normalized.py
src/three_dried/imaging/rmist_normalization.py
tests/test_rmist_normalization.py
results/review/M9/
```

Required lightweight artifacts:

```text
summary_metrics.json
synthetic_reference_scales.json
normalization_scales.csv
phase_activity_by_method.csv
observed_consistency.csv
m2_reference_proxies.csv
range_profile.csv
comparison_mip.png
representative_slices.png
range_profile.png
```

Large complex volumes remain ignored.

---

## 15. Required tests

1. With `c=1`, normalization wrapper exactly reproduces ordinary RMIST.
2. Normalize-then-denormalize around a purely linear RM/RM_dagger control
   preserves expected scaling.
3. Synthetic reference scales reproduce M8 RMS within tolerance.
4. N0 reproduces M7 raw RMIST result within tolerance.
5. No optimizer/backward/train call appears in M9 execution path.
6. All scale values finite and positive.
7. M3-M8 regression suite still passes.

---

## 16. Outcome classification

M9 must end with one of:

```text
A. NORMALIZATION_FEASIBLE_WITH_PROXY_BENEFIT
B. DYNAMICS_RESTORED_BUT_IMAGE_BENEFIT_UNPROVEN
C. NORMALIZATION_DEGRADES_REAL_RECONSTRUCTION
D. GLOBAL_AND_PER_SLICE_TRADEOFF_UNRESOLVED
E. INCONCLUSIVE
```

Do not force category A.

---

## 17. Stop

After M9 results:

STOP for independent review.

Do not retrain or design an adaptive network until M9 has been reviewed.
