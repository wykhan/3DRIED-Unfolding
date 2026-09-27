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

**Status:** HUMAN REVIEW REQUIRED

The focused Hammer is diagonal and recognizable, but its horizontal slope is
opposite the supplied 2-D JPEG under the bundled RMA coordinate labels. Do
not silently flip x: the MATLAB scan ordering and offsets are reproduced as
written, and the absolute aperture direction is not independently calibrated.
Human review is required before M3 fixes an operator coordinate convention.



## D017 — RMIST paper becomes algorithmic source of truth

**Status:** FIXED

From M3-R onward, the baseline is a method-level reproduction of RMIST-Net.
Do not substitute generic ISTA unfolding.

## D018 — RMIST RM / RM_dagger semantics

**Status:** FIXED

Preserve RMIST reciprocal amplitude kernels:

- forward `RM`: `1/ky`;
- inverse `RM_dagger`: `ky`.

`RM_dagger` is the inverse RM kernel used by RMIST and must not be replaced by a Hilbert adjoint.

## D019 — 3DRIED sign adaptation

**Status:** FIXED — SYSTEM ADAPTATION

M2 validates inverse phase `ky * exp(-j*r*ky)`.
Therefore the mapped RMIST pair is:

- `Phi_dagger = ky * exp(-j*r*ky)`
- `Phi = (1/ky) * exp(+j*r*ky)`

## D020 — RMIST training hyperparameters

**Status:** FIXED

Strict baseline:

- 9 phases;
- Adam;
- lr = 1e-4;
- tau_t init = 1e-4;
- theta_t init = 1e-3;
- 100 epochs;
- 1000 synthetic training pairs;
- default 20 dB complex AWGN.

Batch size is paper-unspecified and must be labeled as an implementation choice.

## D021 — Previous generic M3/M4 branch

**Status:** HISTORICAL / NOT ACCEPTED

`milestone/m3-m4` remains as experimental history but is not the accepted learning-physics baseline and must not be merged into `main`.

## D022 — RMIST singular support and numerical precision

**Status:** FIXED — PAPER UNSPECIFIED IMPLEMENTATION CHOICE

Set both strict kernels to zero for non-propagating samples and for
`ky <= 1e-12`. Form `ky` and kernel phase in float64 before casting to the
requested complex dtype. This makes `1/ky` handling explicit and preserves
M2 agreement without changing valid-support kernel semantics.

## D023 — RMIST FFT normalization and physical padding

**Status:** FIXED — SYSTEM ADAPTATION

Use PyTorch's default `backward` FFT convention to reproduce M2/MATLAB, and
center-pad the physical `[z=200,x=407]` single-Rx echo to `[512,512]` with
`z=(156,156)` and `x=(52,53)`. This is the approved 3DRIED grid mapping.

## D024 — M4-R initialization and diagnostic tuning

**Status:** FIXED — PAPER UNSPECIFIED IMPLEMENTATION CHOICE

Initialize conventional strict RM-kernel ISTA with `RM_dagger(S)`. Preserve
the paper-reported network initialization values as an explicit Ini-ISTA
mode: `tau=1e-4`, `theta=1e-3`, and 9 iterations. The paper does not report
final Tuned-ISTA values. For a separate 3DRIED diagnostic, grid-search
`tau=[0.25,0.5,0.75,1.0]` and
`theta=[0.0001,0.0005,0.001,0.005,0.01,0.02]` on the fixed 20 dB full-data
synthetic case; the selected values are `tau=1.0`, `theta=0.01`.

## D025 — M4-R small synthetic protocol

**Status:** FIXED — PAPER UNSPECIFIED IMPLEMENTATION CHOICE

Use a deterministic 128×128, 12-point, random-phase scene at 0.5 m for the
M4-R smoke/diagnostic validation. Keep it explicitly separate from the later
512×512, 100-target, 50-trial paper-style benchmark. The additional 50%
Bernoulli sampling case is classified as an **EXPERIMENTAL EXTENSION**.
