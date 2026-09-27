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

**Status:** HISTORICAL / SUPERSEDED BY D017-D020

Historical plan: the first learning model was generic physics-based ISTA
unfolding. It is not the active specification.

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

The accepted strict replacement is RMIST-Net Eq. (20)-(26), governed by
`docs/RMIST_REPRO_SPEC.md`.

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

Historical wording referred to a generic forward/adjoint operator. The
accepted fixed physics is now the strict `RM` / `RM_dagger` reciprocal pair.

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

**Status:** HISTORICAL / SUPERSEDED BY CURRENT PLAN

Historical gate: Codex had to stop after M2 and wait for human review.

That review occurred. Current milestone locks are defined in `PLAN.md`.

---

## D009 — Initial unfolding depth

**Status:** SUPERSEDED BY D020

Historical planned network depth:

```text
T = 5
```

The accepted strict RMIST depth is 9 phases; this historical value is not active.

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

**Status:** HISTORICAL / ACCEPTED NON-BLOCKING UNCERTAINTY

The focused Hammer is diagonal and recognizable, but its horizontal slope is
opposite the supplied 2-D JPEG under the bundled RMA coordinate labels. Do
not silently flip x: the MATLAB scan ordering and offsets are reproduced as
written, and the absolute aperture direction is not independently calibrated.
Human review retained this convention without a silent flip. The strict
3DRIED RMIST sign mapping was subsequently validated and independently
accepted; absolute display handedness remains a documented non-blocking
uncertainty.



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

## D026 — M3-R/M4-R independent review accepted

**Status:** FIXED

- M3-R is independently reviewed and accepted PASS.
- M4-R is independently reviewed and accepted with non-blocking observations.
- Ini-ISTA's non-monotonic residual under the paper initialization values is
  retained as documented behavior.
- The weaker 50% sampling diagnostic is not a blocker.
- Strict RMIST Eq. (13)-(19) is the accepted physics and optimization baseline.
- M5-R may be planned only after the accepted branch is merged and a human
  explicitly unlocks it.

## D027 — M5-R complex representation and strict architecture

**Status:** FIXED — PAPER UNSPECIFIED IMPLEMENTATION CHOICE

Implement the paper's separate real/imaginary paths with mathematically
equivalent PyTorch complex tensors internally. Provide exact conversion
helpers for the conceptual `[..., 2, H, W]` representation. Each of the nine
phases still uses the paper's complex RM/RM_dagger operations and one shared
magnitude shrinkage scale for real and imaginary components. The only
trainable tensors are independent `tau[9]` and `theta[9]`; no positivity
reparameterization or learned operator is introduced.

## D028 — M5-R independent review accepted

**Status:** FIXED

- Strict 9-phase RMIST-Net Eq. (20)-(26) accepted.
- Exactly 18 trainable scalars: `tau[9]`, `theta[9]`.
- M4-to-M5 regression passed.
- Gradient-flow validation passed.
- PyTorch complex internal representation remains a paper-unspecified
  implementation choice.
- M6-R remains locked until separate human authorization.

## D029 — M6-R strict synthetic training choices

**Status:** FIXED — PAPER UNSPECIFIED IMPLEMENTATION CHOICE

For the first strict M6 model, use one 512×512, 0.5 m, SR=0.5 condition with
100 discrete point scatterers per oracle, amplitudes uniform on `[0.5,1.0]`,
one fixed deterministic mask, 200 disjoint validation pairs, batch size 8,
deterministic seeds and epoch shuffling, and best-checkpoint selection by
minimum validation Eq. (27) loss. Use the displayed Eq. (27) mean per-sample
complex L2 norm. Cache accepted fixed-range kernel tensors as an
**IMPLEMENTATION PERFORMANCE OPTIMIZATION — algorithm unchanged**. Do not
commit the procedural dataset or local checkpoints.

## D030 — M6-R independent review accepted

**Status:** FIXED

- Strict RMIST Eq. (27)-(31) synthetic training accepted.
- 1000 training pairs, 20 dB AWGN, Adam `lr=1e-4`, 9 phases, and 100 epochs
  verified.
- Held-out validation NRMSE improved from `0.6148313` to `0.5302653`.
- The improvement is accepted as a baseline result, not as reproduction of
  the paper's exact table values.
- M6 paper-unspecified choices remain documented.
- The Python 3.11 / PyTorch 2.2.2 / CUDA 12.1 M6 environment differs from
  earlier milestones but is accepted because all regression tests pass.
- Future inference must use the accepted best scalar parameters with
  range-dependent RM kernels and must accurately disclose the fixed SR=0.5
  mask training condition.
- M7-R remains locked pending separate human authorization.

## D031 — M7-R real Hammer inference authorized

**Status:** FIXED

- M6-R is independently reviewed and accepted with no blocking scientific issues.
- M7-R is unlocked for strict Eq. (32) real Hammer inference/evaluation only.
- Use the accepted M6 best-validation parameters and exact fixed SR=0.5 mask.
- Training, fine-tuning, domain adaptation, calibration learning, optimizers,
  and learned-parameter changes remain prohibited.

## D032 — M7-R physical mask mapping and outcome interpretation

**Status:** FIXED — 3DRIED SYSTEM ADAPTATION

- Intersect the exact accepted 512×512 M6 mask with the centered 200×407
  physical aperture; the effective real-data SR is `40682/81400 =
  0.4997788698`.
- Compare all methods with one common peak and the accepted 500 mm ROI.
- Classify the real-data outcome as SUCCESS because the Hammer remains
  recognizable with no catastrophic sim-to-real failure.
- Do not claim a learned improvement: RMIST50 is nearly identical to RMA50
  and Ini-ISTA50 on the unmodified real-data amplitude scale.

## D033 — M7-R independent review accepted

**Status:** FIXED

- Eq. (32) real Hammer inference pipeline is technically accepted.
- The Hammer remains recognizable under the accepted 50% sparse condition,
  with preserved range and orientation.
- RMIST50 differs from RMA50 by only about `1.1364e-6` in complex relative norm.
- Observed-sample consistency is essentially identical across RMA50,
  Ini-ISTA50, and RMIST50.
- Therefore real-data learned improvement is not demonstrated.
- The next scientific question is diagnostic: determine why synthetic learned
  gain collapses to an almost identity/RMA mapping on real data.
- M8 sim-to-real diagnostics remain locked pending separate human authorization.
