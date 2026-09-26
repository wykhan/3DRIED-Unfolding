# M3 Forward / Adjoint Operator Report

Date: 2026-09-26 (Asia/Shanghai)  
Status: **PASS (Codex self-review; external review pending)**

## Final operator

For one fixed range plane, let `P` center-pad the physical `[Nz,Nx]`
aperture into the `[Nzi,Nxi]` image grid, and let `C=P^H` be the matching
center crop. With `k=2*pi*fc/c`, define

\[
k_y=\sqrt{(2k)^2-k_x^2-k_z^2},\qquad
H_r=\operatorname{fftshift}\left(k_y e^{-j r k_y}\right).
\]

Samples for which the radicand is negative are non-propagating and set to
zero. The implemented image-to-aperture forward model and its adjoint are

\[
A_r x=C\,\mathcal F^{-1}\{\overline{H_r}\odot\mathcal F\{x\}\},
\]

\[
A_r^H y=\mathcal F^{-1}\{H_r\odot\mathcal F\{P y\}\}.
\]

This is a deterministic linear operator with no learned parameters. The
adjoint is exactly the validated M2 slice RMA. M2's `ky` amplitude factor is
included in both members of the Hilbert pair; it had been omitted from the
earlier abstract forward equation and is now made explicit in
`RESEARCH_SPEC.md`.

## Coordinates, grids, and transforms

- x: horizontal aperture direction; z: vertical; y/r: range.
- Hammer validation uses x spacing 1 mm, z spacing 2 mm, fc=78.8 GHz.
- Range uses M2's documented one-based MATLAB-bin formula without change.
- `kx` and `kz` are `linspace(-pi/d, pi/d, N)`, matching M2.
- The physical aperture is center-padded from 200×407 to 512×512; the
  forward applies the exact transpose support operation, center cropping.
- `fftshift` is applied to `H`, not to the data or reconstructed image.
- FFT calls use `norm="backward"`: forward FFT unnormalized, inverse FFT
  divided by the total sample count, matching MATLAB/NumPy M2.
- Wavenumber/phase construction uses float64 before the final kernel is cast
  to the complex input dtype. Complex64 and complex128 I/O are preserved.
- The established x handedness and all M2 offsets remain unchanged.

## Complex adjoint identity

The test computes

\[
e_{adj}=\frac{|\langle Ax,y\rangle-\langle x,A^Hy\rangle|}
{\max(|\langle Ax,y\rangle|,10^{-12})}
\]

with `torch.vdot`, which conjugates its first argument. Results:

| Device | Dtype | Seed | Relative error |
|---|---:|---:|---:|
| CPU | complex64 | 7 | 3.860e-7 |
| CPU | complex64 | 23 | 1.409e-6 |
| CPU | complex64 | 101 | 1.371e-6 |
| CPU | complex128 | 7 | 1.664e-15 |
| CPU | complex128 | 23 | 1.024e-15 |
| CPU | complex128 | 101 | 1.596e-15 |
| CUDA | complex64 | 7 | 3.271e-7 |

All are below the required `1e-5`. Shape, dtype, finite-value, CPU, and GPU
tests also pass.

The sensitivity control deliberately uses `H` instead of `conj(H)` in the
forward operation while retaining the correct adjoint. Its error is `1.102`,
versus `7.25e-16` for the correct complex128 pair. The test therefore detects
the relevant phase/conjugation mistake; broken code is not retained in the
operator.

## M2 consistency

The comparison reloads Rx1, repeats the approved M2 preprocessing, uses the
same 13 range bins and physical grids, and compares the saved M2 complex
volume with independently computed `operator.adjoint()` slices.

- unaligned complex relative difference: `2.238e-7`;
- magnitude relative difference: `1.840e-7`;
- magnitude Pearson correlation: `0.9999999999999872`;
- least-squares scalar mapping M3 to M2:
  `1.000000119 - 1.375e-9j`;
- relative difference after scalar alignment: `1.960e-7`.

The scalar is reported only as a diagnostic; it is not applied to the saved
comparison or hidden in the implementation. Coordinates, target orientation,
and range are identical within floating-point precision. The lightweight
comparison figure is `results/m3_operator/m2_consistency.png`.

## Reproduction

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 scripts/validate_operator.py --config configs/m3_operator.yaml
```

Machine-readable results, command, config, and environment are under
`results/m3_operator/`.

## Self-review and unresolved issues

Every blocking M3 item in `REVIEW_CHECKLIST.md` passes. The generic operator
contains no Hammer-specific correction: Hammer parameters are supplied only
through configuration and `HolographyGeometry`.

The absolute x-axis handedness noted at M2 remains an external-coordinate
uncertainty. M3 preserves the approved convention exactly, so it does not
affect adjointness or M2 equivalence. External review is still required; M5
remains locked.

