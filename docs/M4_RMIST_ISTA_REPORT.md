# M4-R Strict RM-Kernel ISTA Report

Date: 2026-09-27 (Asia/Shanghai)
Status: **PASS — STOP FOR INDEPENDENT REVIEW**

## Result

Conventional RM-kernel ISTA now reproduces RMIST Eqs. (16)--(19) using the
strict M3-R `RM` and `RM_dagger` operators. The paper-initialized nine-step
mode completed on complex random-phase point targets with NRMSE `0.03741`
noise-free and `0.04553` at 20 dB. A separately identified 3DRIED diagnostic
grid search selected `tau=1.0`, `theta=0.01`, producing NRMSE `0.01016`
noise-free and `0.01027` at 20 dB.

## Equation-to-code mapping

`src/three_dried/solvers/rmist_ista.py` implements one iteration literally:

```text
v_t       = S - RM(alpha_t)                  Eq. (19), residual
r_t       = alpha_t + tau*RM_dagger(v_t)     Eq. (17)/(19), rough estimate
alpha_t+1 = F_st(r_t, theta)                 Eq. (18)/(19), reconstruction
```

The complex shrinkage is
`r/(abs(r)+epsilon) * ReLU(abs(r)-theta)`, so real and imaginary parts share
one magnitude scale and nonzero phase is preserved. The solver initializes
with `RM_dagger(S)`, consistent with RMIST's inverse-kernel initialization.
It contains no Hilbert-adjoint replacement, FISTA momentum, learned
parameters, CNN/proximal network, or ground-truth-dependent update.

## Synthetic protocol

The small M4-R diagnostic is deliberately separate from the later 512-square
paper-style benchmark:

- grid: 128×128;
- range: 0.5 m;
- targets: 12 unit-amplitude point scatterers at uniformly selected pixels;
- phase: independent uniform `[0,2*pi)`;
- echo: strict `RM(alpha_gt)`;
- initial test: noise-free, full `Mask=1`;
- noise test: circular complex AWGN, requested 20 dB and achieved 19.973 dB;
- sampling extension: separate Bernoulli 50% mask, realized rate 0.50549;
- seed: 20260927.

Ground truth is used only to compute synthetic evaluation metrics and to
select the optional diagnostic parameters; it never enters an ISTA update.

## Parameter modes and results

Ini-ISTA uses exactly `tau=1e-4`, `theta=1e-3`, and 9 iterations. These are
the initialization values reported by RMIST for its network. The paper does
not report them as optimal conventional ISTA parameters and does not disclose
its final Tuned-ISTA values.

| Case | Mode | NRMSE | normalized echo residual | runtime (s) |
|---|---|---:|---:|---:|
| noise-free, Mask=1 | Ini-ISTA | 0.037414 | 0.087364 | 0.0291 |
| noise-free, Mask=1 | tuned diagnostic | 0.010156 | 0.010156 | 0.0290 |
| 20 dB, Mask=1 | Ini-ISTA | 0.045532 | 0.132793 | 0.0289 |
| 20 dB, Mask=1 | tuned diagnostic | 0.010271 | 0.100160 | 0.0266 |
| 20 dB, 50% mask | Ini-ISTA | 0.695072 | 0.343173 | 0.0247 |
| 20 dB, 50% mask | tuned diagnostic | 0.671940 | 0.375632 | 0.0237 |

The full script, including the tuning sweep and plots, took 1.88 s on CPU.
The sparse-sampling extension is reported transparently; its weaker result
does not replace the required full-data validation.

## Tuned diagnostic history

The optional tuning sweep tested the Cartesian product:

```text
tau   = [0.25, 0.5, 0.75, 1.0]
theta = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]
```

Each candidate used nine iterations and was ranked by final NRMSE on the
fixed 20 dB full-data synthetic case. The selected `tau=1.0`, `theta=0.01`
is a **PAPER UNSPECIFIED IMPLEMENTATION CHOICE / 3DRIED diagnostic**, not a
paper-reported result. Every candidate and metric is saved in
`results/m4_rmist_ista/tuning.json`.

The Ini-ISTA residual does not decrease monotonically because the paper's
initial parameter values are not reported as tuned conventional-ISTA values;
this behavior is preserved rather than hidden by changing the algorithm.

## Tests and artifacts

Unit tests verify the exact one-step Eq. (19) update, nine-iteration defaults,
magnitude shrinkage, phase preservation, complex random-phase scenes,
deterministic AWGN, and binary masks. All M3-R kernel and M0-M2 regression
tests also pass.

Artifacts under `results/m4_rmist_ista/` include the exact command,
environment, copied config, metrics, complete tuning history, complex
measurements/reconstructions, masks, normalized-dB images, and residual
curves.

## Reproduction

```bash
PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  -m unittest discover -s tests -v

/home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  scripts/run_rmist_ista.py --config configs/rmist_ista.yaml
```

## Deviations from RMIST

- **SYSTEM ADAPTATION:** 3DRIED frequency, aperture spacing, propagation
  sign, and strict M3-R support handling.
- **PAPER UNSPECIFIED IMPLEMENTATION CHOICE:** PyTorch implementation,
  epsilon `1e-12`, deterministic 128×128 smoke scene, scatterer count, seed,
  and diagnostic tuning grid.
- **EXPERIMENTAL EXTENSION:** separately reported 50% Bernoulli mask case.

No blocking discrepancy was found in Eqs. (16)--(19). M5-R and training have
not started and remain locked pending independent review.
