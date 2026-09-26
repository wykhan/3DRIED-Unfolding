# DECISIONS.md

This file records scientific and engineering decisions that should remain stable unless explicitly revised.

---

## D001 — Create a clean 3DRIED-specific baseline

**Status:** FIXED

Do not continue building the baseline inside the TomoSAR-Toe-LISTA implementation.

Reason:

- TomoSAR geometry and 3DRIED near-field planar imaging are different;
- Toe-LISTA/Gamma-Net introduces unnecessary structural complexity;
- the baseline should make physics/operator debugging easy.

---

## D002 — Use ISTA unfolding as the first learning-imaging baseline

**Status:** FIXED

The first learning model is minimal physics-based ISTA unfolding.

Planned per-layer update:

\[
r_t = y-Ax_t
\]

\[
z_t=x_t+\tau_tA^Hr_t
\]

\[
x_{t+1}=\mathcal S_{\theta_t}(z_t).
\]

Only per-layer step sizes and thresholds are learned in the initial version.

---

## D003 — Use one Rx only

**Status:** FIXED

The baseline uses one receive channel only.

Do not:

- average Rx channels;
- coherently combine Rx channels;
- form multi-Rx features;
- calibrate channels jointly.

The exact Rx index used in Hammer reconstruction must be documented.

---

## D004 — Conventional RMA must be reproduced before learning

**Status:** FIXED

No unfolding model may be trained until the real Hammer data can be reconstructed by a conventional RMA/holographic baseline.

Reason:

A failed learning result is not interpretable unless the data preprocessing and physical imaging chain are already known to work.

---

## D005 — Physics operator remains fixed

**Status:** FIXED

The planned forward/adjoint operator is deterministic and geometry-based.

The baseline does not learn the forward operator.

---

## D006 — Synthetic oracle labels, not Hammer RMA labels

**Status:** FIXED

The real Hammer RMA image is not used as supervised ground truth.

Synthetic scattering images are generated and propagated through the physics operator to create paired training data.

Real RMA may later be used only for:

- amplitude-scale estimation;
- qualitative comparison;
- debugging;
- visualization reference.

---

## D007 — No sparse-aperture experiment in V1

**Status:** FIXED

The first baseline uses the full available single-Rx aperture.

Artificial sparse-height, sparse-frequency, or sparse-position experiments are deferred.

---

## D008 — M0-M2 are the only currently unlocked milestones

**Status:** FIXED

Codex must stop after M2 and wait for human review.

M3-M7 remain locked until PLAN.md is explicitly updated.

---

## D009 — Initial unfolding depth

**Status:** PROVISIONAL

Planned initial network depth:

```text
T = 5
```

May later be increased to 8 or 10 only if synthetic experiments show five layers are insufficient.

---

## D010 — Initial training distribution

**Status:** PROVISIONAL

Initial synthetic training uses point-like scatterers with random:

- positions;
- amplitudes;
- phases;
- additive complex noise.

Surface-like structured training targets are deferred until there is evidence of synthetic-to-real mismatch.

---

## D011 — Hammer receiver and aperture convention

**Status:** FIXED FOR M2

Use MATLAB Rx1 (Python index 0), because the bundled official/reference
`Demo_RMA2D_Imaging.m` selects `Raw_echo(1:4:end,:)`. Use a 407-sample
horizontal axis at 1 mm spacing and a 200-sample vertical axis at 2 mm
spacing, apply the reference 43-frame motion correction, and reverse every
second scan row to undo serpentine acquisition. Do not load or combine the
other receivers for imaging.

---

## D012 — Reproduce reference RMA FFT convention

**Status:** FIXED FOR M2

Use the bundled MATLAB RMA's default FFT scaling (forward transform
unnormalized, inverse transform scaled by `1/N`) instead of the preferred
orthonormal convention. This is an explicit reference-reproduction exception
to AGENTS.md section 5.3. The output remains complex through reconstruction.

---

## D013 — M2 range-bin and coordinate convention

**Status:** FIXED FOR M2

Reconstruct MATLAB range bins 12--24 with the bundled reference range
formula, including its direct use of the one-based bin number. Use x as the
horizontal aperture coordinate, y as range, and z as vertical, with the
reference display offsets x=-50 mm and z=+120 mm. Retain the full 512×512
complex grid and apply a 500 mm x-z ROI only to visualizations.

---

## D014 — Preserve reference x handedness pending review

**Status:** ACCEPTED CONVENTION; EXTERNAL UNCERTAINTY RETAINED

The focused Hammer is diagonal and recognizable, but its horizontal slope is
opposite the supplied 2-D JPEG under the bundled RMA coordinate labels. Do
not silently flip x: the MATLAB scan ordering and offsets are reproduced as
written, and the absolute aperture direction is not independently calibrated.
M0--M2 human review accepted continuing with this convention. M3 preserves
it exactly; the absolute external x sign remains documented rather than
silently changed.

---

## D015 — Define M3 as the exact Hilbert pair of M2 RMA

**Status:** FIXED

For a range slice, define the M3 adjoint to be exactly the validated M2
operation `ifft2(fft2(center_pad(y)) * H)`, with
`H=fftshift(ky*exp(-j*range*ky))`. The forward is its Hilbert adjoint:
`center_crop(ifft2(fft2(x)*conj(H)))`. This preserves M2's `ky`
amplitude/obliquity factor, centered aperture support, phase sign, and
MATLAB-compatible FFT scaling. The previous abstract specification omitted
the `ky` factor and did not make pad/crop explicit; `RESEARCH_SPEC.md` is
updated accordingly without changing the physical model.

Wavenumber grids are computed in float64, as in M2, before casting the final
kernel to the complex input dtype. This avoids a measured range-phase error
from float32 grid construction while preserving complex64 operator I/O.

---

## D016 — M4 step size, initialization, and regularization

**Status:** FIXED FOR M4 BASELINE

For `A = C U^H diag(conj(H)) U`, centered cropping has norm one and the
unitary-equivalent FFT factors cancel, so `||A||² <= max|H|²`. Use the safe
step `tau = 0.95 / max|H|²`; retain a seeded 60-step power iteration as an
independent tightness check. In the accepted run, the power estimate is
99.207% of the analytic upper bound.

Initialize with one scaled adjoint/gradient step from zero,
`x0 = tau A^H y`. Set `lambda = 0.002 max|A^H y|`, making the soft threshold
`lambda*tau`. These rules depend only on the operator and measurement, never
on synthetic ground truth. The fraction was declared in the config before
the first M4 run and was not tuned against NMSE.
