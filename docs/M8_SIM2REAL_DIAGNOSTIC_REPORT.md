# M8 RMIST Sim-to-Real Diagnostic Report

Date: 2026-09-27 (Asia/Shanghai)

Status: **PASS — SCALE_MISMATCH_SUPPORTED — STOP FOR INDEPENDENT REVIEW**

## Result

The accepted RMIST model is almost inactive on unmodified Hammer measurements
because real reconstruction amplitudes are roughly nine orders of magnitude
above the scale used for M6 synthetic training. The fixed learned thresholds
are consequently negligible: on Hammer bin 17, no pixel is below even five
times the phase threshold, maximum relative shrinkage is `2.9245e-11`, and the
final output moves only `1.1412e-6` from RMA initialization.

Counterfactually scaling the same measurement to the synthetic initialization
RMS restores strong threshold and phase activity without changing any learned
value. At `c_align=4.3283002043e-10`, final movement from RMA becomes
`0.7151915`, or about `626,689` times the unscaled value, and reaches `0.867`
of the synthetic control's median final movement. This supports the prescribed
classification:

```text
SCALE_MISMATCH_SUPPORTED
```

This is mechanism evidence, not a production normalization proposal or an
image-quality claim. M8 did not select a deployment scale.

## Fixed controls and instrumentation

The diagnostic used the accepted epoch-99 model and M6 mask:

```text
checkpoint SHA-256: 4b1925f2a6d361a848a81b7e70a2dedc2dc3e520726a29b2ac92a6de999c26ce
mask SHA-256:       43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2
parameters changed: no
production normalization added: no
```

The external trace records `alpha_before`, residual, gradient correction,
rough estimate, and `alpha_after` for all nine phases. Across every synthetic,
Hammer, and counterfactual run, traced output and ordinary
`RMISTNet.forward()` output were bitwise identical; maximum complex relative
difference was `0.0`. The production model was not modified.

Diagnostic denominators add only `torch.finfo(real_dtype).eps` for numerical
safety.

## Synthetic active control

The active control uses the first 32 deterministic samples from the accepted
M6 validation generator: 512×512, range 0.5 m, fixed SR=0.5 mask, 20 dB AWGN,
100 points, amplitudes `U(0.5,1.0)`, and validation seed `20271000`.

Synthetic `alpha0` RMS median is `0.01376872018`. Across all samples and
phases:

| Metric | Maximum | Median |
|---|---:|---:|
| `E_grad` | 1.568247 | 0.534452 |
| `E_shrink` | 0.849530 | 0.473779 |
| `E_update` | 0.849530 | 0.057916 |

Final phase `E_RMA` has median `0.825165`, mean `0.825640`, and maximum
`0.833503`. The synthetic control is therefore demonstrably active.

## Hammer scale and phase activity

Accepted Rx1 preprocessing, the effective M6 mask, and bins 12–24 were reused
unchanged. Bin 17 at `0.6143103953 m` is the detailed trace.

Bin-17 `alpha0` magnitude statistics are:

| Statistic | Value |
|---|---:|
| RMS | 31,810,917.7538 |
| Mean | 21,481,466.3225 |
| Median | 16,864,806.0 |
| p90 | 37,972,054.0 |
| p95 | 48,022,136.6 |
| p99 | 103,989,317.2 |
| Max | 574,303,488.0 |

The Hammer-to-synthetic ratios are not a max-only effect: RMS is
`2.3104e9`, mean `2.0461e9`, median `1.7306e9`, p90 `2.1388e9`, p95
`2.3693e9`, and p99 `4.1313e9`.

Unscaled Hammer bin-17 activity is:

| Metric | Maximum | Median |
|---|---:|---:|
| `E_grad` | 6.872970e-7 | 9.499256e-8 |
| `E_shrink` | 2.924531e-11 | 2.225782e-12 |
| `E_update` | 6.880620e-7 | 9.639789e-8 |
| final `E_RMA` | 1.141222e-6 | 1.141222e-6 |

Across all 13 Hammer slices, maximum `E_shrink` remains only `2.379620e-9`
and median final `E_RMA` is `1.132022e-6`.

## Hypothesis discrimination

At phase 1, before threshold inactivity can propagate through later phases,
Hammer and synthetic controls have comparable residual and correction ratios:

```text
Hammer/synthetic phase-1 E_res ratio:    1.1163
Hammer/synthetic phase-1 E_grad ratio:   0.9650
Hammer/synthetic phase-1 E_shrink ratio: 3.4833e-11
```

Thus the initial residual/correction is not uniquely degenerate on Hammer.
The decisive separation is threshold activity: synthetic phase-1 mean
`E_shrink` is `0.83958`, whereas Hammer phase-1 `E_shrink` is
`2.92453e-11`. Once synthetic shrinkage moves the state, its later residual
and gradient corrections become active; Hammer stays near the reciprocal-kernel
fixed point because shrinkage never initiates that movement.

## Counterfactual scale sweep

For each scale, the input was `c*S_real`, the output was divided by `c`, and
the fixed accepted model was used unchanged.

| Scale | `c` | Max `E_shrink` | Max `E_update` | Final `E_RMA` |
|---|---:|---:|---:|---:|
| original | 1.0 | 2.9245e-11 | 6.8806e-7 | 1.1412e-6 |
| 0.25×align | 1.0821e-10 | 0.922456 | 0.922456 | 0.901553 |
| 0.5×align | 2.1642e-10 | 0.827215 | 0.827215 | 0.809142 |
| 1×align | 4.3283e-10 | 0.736629 | 0.736629 | 0.715191 |
| 2×align | 8.6566e-10 | 0.589212 | 0.589212 | 0.546853 |
| 4×align | 1.7313e-9 | 0.374152 | 0.374152 | 0.334447 |

The monotonic scale response and restoration of synthetic-order activity
support amplitude mismatch as the observed inactivity mechanism. The sweep
does not establish which scale, if any, would improve real reconstruction;
there is no Hammer voxel ground truth and deployment changes were outside M8.
Broader distribution mismatch is not ruled out.

## Reproduction and artifacts

```bash
PYTHONPATH=src python3 scripts/diagnose_rmist_sim2real.py \
  --config configs/rmist_sim2real_diagnostic.yaml
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Lightweight evidence is under `results/review/M8/`. No large traced tensors
were saved. M8 stops here for independent review; no training, parameter
change, production normalization, or M9 work is authorized.
