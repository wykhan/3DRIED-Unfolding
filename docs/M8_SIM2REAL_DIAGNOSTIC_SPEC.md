# M8_SIM2REAL_DIAGNOSTIC_SPEC.md

# M8 — RMIST Sim-to-Real Diagnostic

## 1. Scientific question

M6 shows measurable synthetic validation gain:

```text
NRMSE 0.6148313 -> 0.5302653
```

but M7 shows:

```text
RMIST50 vs RMA50 complex relative difference ≈ 1.1364e-6
```

on real Hammer.

M8 asks:

\[
\boxed{
\text{Why does learned RMIST gain on synthetic data collapse to an almost identity/RMA mapping on real data?}
}
\]

M8 is diagnosis only.

Do NOT retrain.
Do NOT fine-tune on Hammer.
Do NOT change learned `tau_t` or `theta_t`.
Do NOT add normalization to the production pipeline.
Do NOT change masks or operators.

Any rescaling experiment in M8 is counterfactual diagnostic evidence only.

---

## 2. Main hypotheses

### H1 — amplitude/scale mismatch

The RMIST paper states that synthetic amplitude values are chosen with
reference to RMA reconstruction magnitudes and remain of the same order of
magnitude.

Our strict M6 implementation instead used:

```text
amplitude ~ Uniform(0.5, 1.0)
```

as an explicitly paper-unspecified implementation choice.

If real Hammer initialization magnitudes are on a very different scale, the
learned thresholds may become ineffective:

\[
\theta_t \ll |r_t|
\]

so:

\[
F_{st}(r_t,\theta_t)\approx r_t.
\]

### H2 — residual/update degeneracy on real data

Because:

\[
\alpha_0=RM^\dagger(S)
\]

and the accepted RM/RM_dagger kernels are reciprocal, the initial residual:

\[
v_0=S-RM(\alpha_0)
\]

may already be small. If all learned correction terms remain tiny:

\[
\frac{\|\tau_t RM^\dagger(v_t)\|_2}{\|\alpha_t\|_2}
\ll 1,
\]

the network will remain near its RMA initialization.

### H3 — distribution mismatch beyond scalar amplitude

Even after amplitude alignment, real Hammer may differ from synthetic data in:

- spatial sparsity/compressibility;
- clutter/background;
- phase statistics;
- target support/extended structure;
- measurement distribution.

If scale alignment does not restore nontrivial phase activity, then mismatch
is not explained by amplitude alone.

---

## 3. Data used

### Synthetic side

Use the accepted M6 validation generator/config:

```text
grid: 512 x 512
range: 0.5 m
SR: 0.5
mask seed: 20260928
20 dB AWGN
100 point scatterers
amplitude U(0.5,1.0)
```

Use a deterministic subset of validation samples:

```text
32 samples
```

### Real side

Use accepted M7 Hammer preprocessing and the same accepted M6 mask.

Analyze:

- all 13 slices for aggregate scale statistics;
- bin 17 / 614.310 mm as the primary detailed phase-trace slice;
- optionally bins 16 and 18 as neighboring checks.

---

## 4. Diagnostic A — measurement and initialization scale

For synthetic and real data, record magnitude statistics for:

1. masked measurement \(|S|\);
2. initialization \(|\alpha_0|=|RM^\dagger(S)|\).

Required statistics:

```text
RMS
mean
median
p90
p95
p99
max
```

For sparse arrays, report both all-pixel and active-support statistics where meaningful.

Define scale ratios, e.g.:

\[
\rho_{\rm RMS}
=
\frac{RMS(|\alpha_{0,\rm real}|)}
{median_{\rm synth\ samples}\;RMS(|\alpha_{0,\rm synth}|)}.
\]

Do not infer cause from a single statistic.

---

## 5. Diagnostic B — threshold activity

For every phase \(t=1,\dots,9\), before shrinkage:

\[
r_t=\alpha_t+\tau_tRM^\dagger(v_t).
\]

Record:

\[
q_{50,t}=\frac{\theta_t}{P_{50}(|r_t|)+\epsilon},
\]

\[
q_{90,t}=\frac{\theta_t}{P_{90}(|r_t|)+\epsilon},
\]

\[
q_{99,t}=\frac{\theta_t}{P_{99}(|r_t|)+\epsilon}.
\]

Also report:

```text
fraction |r_t| <= theta_t
fraction |r_t| <= 2*theta_t
fraction |r_t| <= 5*theta_t
```

and relative shrinkage magnitude:

\[
E^{shrink}_t
=
\frac{\|\alpha_{t+1}-r_t\|_2}
{\|r_t\|_2+\epsilon}.
\]

---

## 6. Diagnostic C — residual and gradient-step activity

Per phase record:

\[
E^{res}_t=
\frac{\|v_t\|_2}{\|S\|_2+\epsilon},
\]

\[
E^{grad}_t=
\frac{\|\tau_tRM^\dagger(v_t)\|_2}
{\|\alpha_t\|_2+\epsilon},
\]

\[
E^{update}_t=
\frac{\|\alpha_{t+1}-\alpha_t\|_2}
{\|\alpha_t\|_2+\epsilon}.
\]

Report synthetic mean/std and real Hammer values.

---

## 7. Diagnostic D — cumulative divergence from RMA initialization

For every phase:

\[
E^{RMA}_t=
\frac{\|\alpha_t-\alpha_0\|_2}
{\|\alpha_0\|_2+\epsilon}.
\]

Record how far the model ever moves away from RMA initialization.

---

## 8. Diagnostic E — scale-alignment counterfactual

This is NOT a production fix.

Compute:

\[
c_{align}
=
\frac{
median_{\rm synth}\;RMS(|\alpha_{0,\rm synth}|)
}{
RMS(|\alpha_{0,\rm real}|)
}.
\]

For Hammer bin 17 evaluate the fixed trained model on:

```text
c = 1
c = 0.25*c_align
c = 0.5*c_align
c = 1.0*c_align
c = 2.0*c_align
c = 4.0*c_align
```

For each:

1. run model on `c*S_real`;
2. divide reconstruction by `c` for original-scale comparison;
3. record phase activity;
4. record relative difference from RMA initialization.

Interpretation:

```text
SCALE_MISMATCH_SUPPORTED
```

only if scale alignment materially increases shrinkage/update activity and
moves the model away from RMA compared with `c=1`.

Otherwise classify amplitude alone as insufficient.

---

## 9. Diagnostic F — synthetic active control

Use deterministic M6 validation samples where learned synthetic gain is known.

Report the same:

```text
E_res
E_grad
E_shrink
E_update
E_RMA
```

alongside Hammer.

This prevents diagnosing inactivity without a known active control.

---

## 10. Root-cause classification

M8 must end with one of:

```text
A. SCALE_MISMATCH_SUPPORTED
B. SCALE_MISMATCH_NOT_SUFFICIENT / BROADER_DISTRIBUTION_MISMATCH_SUPPORTED
C. RESIDUAL_DEGENERACY_DOMINANT
D. MULTIPLE_FACTORS_SUPPORTED
E. INCONCLUSIVE
```

The classification must be supported quantitatively.

Do not automatically turn diagnosis into an algorithm modification.

---

## 11. Required outputs

Create:

```text
docs/M8_SIM2REAL_DIAGNOSTIC_REPORT.md
configs/rmist_sim2real_diagnostic.yaml
scripts/diagnose_rmist_sim2real.py
tests/test_rmist_sim2real_diagnostic.py
results/review/M8/
```

Required lightweight artifacts:

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

---

## 12. Required tests

1. Diagnostic instrumentation does not alter RMIST outputs.
2. Traced forward pass matches ordinary M7 RMIST output.
3. Scale-statistic helpers reproduce toy values.
4. Phase decomposition is numerically consistent.
5. No optimizer/backward/training call appears in M8 execution path.
6. M3-M7 regression suite still passes.

---

## 13. M8 acceptance

M8 is successful if it produces a defensible diagnosis, not if a preferred
hypothesis wins.

PASS requires:

- synthetic and real scales quantified;
- all 9 phase dynamics measured;
- threshold activity quantified;
- gradient-step activity quantified;
- cumulative movement from RMA quantified;
- counterfactual scale alignment completed;
- synthetic active-control comparison completed;
- root-cause classification supported by evidence;
- no training or parameter modification.

STOP after M8 for independent review.
