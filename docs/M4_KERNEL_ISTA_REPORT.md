# M4 Kernel-ISTA Report

Date: 2026-09-26 (Asia/Shanghai)
Status: **PASS (Codex self-review; external review pending)**

## Objective and iteration

M4 solves the conventional, non-learned problem

\[
\min_x \frac12\|Ax-y\|_2^2+\lambda\|x\|_1
\]

with the validated M3 `HolographyOperator`. Each iteration is

\[
r_t=y-Ax_t,\qquad
x_{t+1}=\mathcal S_{\lambda\tau}(x_t+\tau A^H r_t),
\]

where

\[
\mathcal S_\theta(z)=\frac{z}{|z|+\epsilon}\max(|z|-\theta,0).
\]

The implementation thresholds complex magnitude and retains phase. There are
no learned parameters, and ground truth is never passed to the solver.

## Operator, initialization, and step size

The same M3 class supplies both `A` and `A^H`. For
`A=C U^H diag(conj(H)) U`, the centered crop `C` has norm one and the FFT
scale factors cancel, giving the rigorous upper bound

\[
L=\|A\|_2^2\leq\max|H|^2.
\]

The run uses `max|H|²=10,907,043` and
`tau=0.95/max|H|²=8.7099684e-8`. A seeded 60-step power iteration estimates
`10,820,560`, or 99.207% of the bound, confirming it is tight. This selection
uses operator structure only, not ground truth.

Initialization is one gradient step from zero,
`x0=tau*A^H*y`, which is the correctly scaled adjoint initialization for the
objective. Regularization is set by the measurement-only rule
`lambda=0.002*max|A^H*y|=5444.368`; therefore the applied threshold is
`4.7420273e-4`. This fraction was fixed in the config before the first run
and was not tuned by inspecting NMSE.

## Synthetic scene and noise model

The first M4 validation is intentionally noise-free. Seed 20260926 creates a
64×64 single-range scene with eight unique point scatterers, a six-pixel
margin, amplitudes uniform on `[0.5,1]`, and independent phases uniform on
`[0,2*pi)`. The M3 forward operator generates a 40×48 complex measurement at
range 0.614310 m. No Hammer data or RMA labels are used.

## Quantitative results

| Metric | Value |
|---|---:|
| Iterations | 250 |
| Initial scaled-adjoint NMSE | 0.64836472 |
| Final Kernel-ISTA NMSE | 5.8819060e-6 |
| NMSE improvement factor | 110,230.38× |
| Initial objective | 1,016,695.06 |
| Final objective | 31,133.37 |
| Initial data fidelity | 830,735.63 |
| Final data fidelity | 38.3237 |
| Runtime (power iteration + primary ISTA) | 0.484 s |
| Perturbed-initialization final NMSE | 5.8817363e-6 |

The objective is non-increasing within the expected float32 rounding
tolerance. The final array is finite. A second complete run produced the same
reconstruction SHA-256,
`beb92dfd881d5b0850ae98b2979c79639c95019b28d9a842cafa2f8312da263f`.
A 0.1% norm perturbation of the initialization changes final NMSE by less
than 0.003%, providing the required stability sanity check.

The comparison figure shows all eight target locations in the final result,
while the scaled adjoint contains broad sidelobes. `convergence.png` records
the full objective and data-fidelity histories; the machine-readable history
is in `metrics.json`.

## Tests and reproduction

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/run_ista.py --config configs/kernel_ista_baseline.yaml
```

All ten repository tests pass, including M3 adjoint tests, phase-preserving
soft thresholding, deterministic complex scene generation, and an ISTA
objective-decrease test. Outputs are under `results/m4_kernel_ista/`.

## Known limitations

- This acceptance experiment is one small, single-range, noise-free sparse
  scene. A noisy 20 dB test is optional and was not needed for the required
  noise-free gate.
- The result validates conventional Kernel-ISTA, not learned unfolding or
  Hammer inference; those milestones remain locked.
- The analytic norm is a safe upper bound rather than an exact closed-form
  norm after aperture cropping. Power iteration shows the bound is tight.
- Float32 objective values plateau with milliscale rounding fluctuations;
  no divergence or meaningful objective increase is present.

## Self-review

Every blocking M4 item in `REVIEW_CHECKLIST.md` passes. Source, config,
machine-readable metrics, and lightweight figures are prepared for external
review. M5 has not been started and remains locked.
