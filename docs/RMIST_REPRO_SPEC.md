# RMIST_REPRO_SPEC.md

# Strict RMIST-Net Reproduction Specification for 3DRIED

## 1. Authority

From M3-R onward, the algorithmic source of truth is:

1. Wang et al., **RMIST-Net: Joint Range Migration and Sparse Reconstruction Network for 3-D mmW Imaging**, especially Sec. III and Eq. (13)-(31).
2. This file.
3. Human-approved 3DRIED M0-M2 geometry/preprocessing.
4. Repository code.

If a mathematically cleaner alternative conflicts with RMIST, reproduce RMIST. Do **not** silently replace RMIST's inverse RM kernel with a Hilbert adjoint.

## 2. What must remain faithful to RMIST

Keep:

- forward RM kernel with `1/ky`;
- inverse RM kernel `RM_dagger` with `ky`;
- reciprocal forward/inverse phase pair;
- padded matrix-form echo;
- RM-kernel ISTA Eq. (16)-(19);
- phase-wise learnable `tau_t`, `theta_t`;
- complex magnitude shrinkage;
- synthetic oracle training;
- 1000 training pairs;
- default 20 dB complex AWGN;
- averaged L2/RMSE-style reconstruction loss;
- Adam, learning rate `1e-4`;
- `tau_t` init `1e-4`;
- `theta_t` init `1e-3`;
- 9 phases;
- 100 epochs;
- slice-wise 3-D reconstruction after range focusing.

Allowed 3DRIED adaptations:

- system/radar parameters;
- aperture dimensions/spacings;
- single-Rx selection;
- range coordinates;
- padding dimensions;
- global propagation sign convention required to match validated M2;
- PyTorch instead of TensorFlow;
- batch size, because the paper does not specify it.

Every non-paper choice must be labeled either `3DRIED system adaptation` or `paper-unspecified implementation choice`.

## 3. Canonical RMIST kernels

RMIST defines:

\[
\Phi_r = k_y^{-1}e^{-j k_y r},
\qquad
\Phi_r^\dagger = k_y e^{+j k_y r}.
\]

\[
RM(\alpha)=\mathcal F^{-1}_{2D}
[\mathcal F_{2D}(\alpha)\odot\Phi_r]
\]

\[
RM^\dagger(S)=\mathcal F^{-1}_{2D}
[\mathcal F_{2D}(S)\odot\Phi_r^\dagger].
\]

Terminology:

- `RM` = forward RM operator.
- `RM_dagger` = inverse RM operator used by RMIST.
- Do not call `RM_dagger` a Hilbert adjoint in the strict baseline.

## 4. 3DRIED sign mapping

The validated 3DRIED M2 inverse/reconstruction kernel uses:

\[
k_y e^{-j k_y r}.
\]

Therefore the 3DRIED-mapped RMIST pair is fixed as:

\[
\Phi_{r,3D}^\dagger = k_y e^{-j k_y r}
\]

\[
\Phi_{r,3D} = k_y^{-1} e^{+j k_y r}.
\]

This is a coordinate/sign adaptation only. The reciprocal amplitude structure `1/ky` versus `ky` must not change.

On valid propagating support:

\[
\Phi_{r,3D}\Phi_{r,3D}^\dagger \approx 1.
\]

## 5. Wavenumber/grid

Use the validated M2 convention:

\[
k=\frac{2\pi f_c}{c},
\qquad
k_y=\sqrt{4k^2-k_x^2-k_z^2}.
\]

Keep M2's x/z sampling, range-bin convention, FFT-grid size, and padding convention. Explicitly mask non-propagating samples. Avoid division by zero/near-zero `ky`; invalid forward-kernel entries must be set to zero and documented.

## 6. Padding

RMIST works with padded matrix-form echo:

\[
S=P(S_r).
\]

Current 3DRIED baseline:

```text
physical single-Rx slice: [200, 407]
padded RM domain:         [512, 512]
```

Implement explicit padding/cropping utilities. Do not introduce a vectorized sensing matrix.

# 7. M3-R — RM / RM_dagger

Suggested file:

```text
src/three_dried/imaging/rmist_kernel.py
```

Required conceptual interface:

```python
rm(image, range_m, mask=None)
rm_dagger(padded_echo, range_m)
pad_measurement(measurement)
crop_measurement(padded_echo)
```

Expose/test `phi`, `phi_dagger`, and valid spectral support.

## Required M3-R tests

### A. Formula test
Verify implemented kernels equal:

\[
\Phi_{r,3D}=k_y^{-1}e^{+j k_y r}
\]

\[
\Phi_{r,3D}^\dagger=k_y e^{-j k_y r}.
\]

### B. Reciprocal-kernel test
On valid support:

\[
\Phi_{r,3D}\Phi_{r,3D}^\dagger\approx1.
\]

Report max and RMS error.

### C. M2 consistency
Apply `RM_dagger` to the same Hammer range-focused slices as M2. Target:

```text
complex relative difference < 1e-5
```

### D. Synthetic round trip
Use a synthetic scene supported on valid propagating spectrum:

\[
\alpha\to RM(\alpha)\to RM^\dagger(RM(\alpha)).
\]

Evaluate round-trip error separately from physical-aperture crop effects.

### E. Padding test
Verify the physical 3DRIED slice is padded exactly into the RM matrix domain.

### Prohibited PASS criterion
Do not require a Hilbert-adjoint identity. M3-R is about RMIST's forward/inverse kernels, not replacing `RM_dagger` with a mathematical adjoint.

# 8. M4-R — RM-kernel ISTA

Implement RMIST Eq. (16)-(19):

\[
\hat\alpha=\arg\min_\alpha
\frac12\|S-RM(\alpha)\|_F^2+\lambda\|\alpha\|_1
\]

\[
v_t=S-RM(\alpha_t)
\]

\[
r_t=\alpha_t+\tau RM^\dagger(v_t)
\]

\[
\alpha_{t+1}=F_{st}(r_t,\theta).
\]

Do not substitute FISTA, ADMM, LISTA, Gamma-Net, Toe-LISTA, learned proximal blocks, or a Hilbert-adjoint operator.

## Complex threshold

\[
F_{st}(r,\theta)
=
\frac{r}{|r|+\epsilon}\operatorname{ReLU}(|r|-\theta).
\]

Real/imaginary parts must share the same magnitude shrinkage factor.

## M4-R parameter modes

### Ini-ISTA reproduction mode

Use:

```text
tau = 1e-4
theta = 1e-3
iterations = 9
```

`tau`/`theta` are RMIST-reported initialization values. Do not claim the paper reports these as optimal conventional ISTA values.

### Tuned diagnostic mode

If needed for 3DRIED-scale validation, allow a separately documented tuned conventional ISTA. Exact tuned values are **3DRIED implementation choices** because the paper does not disclose its final Tuned-ISTA values.

Do not reuse the discarded Hilbert-adjoint Lipschitz logic unless independently justified for the strict RM/RM_dagger algorithm.

## Sampling mask

Code must support a binary `Mask`. First validate full data:

```text
Mask = 1
```

Then add a separate 50% random-sampling experiment. Do not mix sparse sampling into initial M3-R debugging.

# 9. M5-R — RMIST-Net (EXTERNALLY REVIEWED PASS)

When unlocked, implement:

\[
v_t=S-RM(\alpha_t)
\]

\[
r_t=\alpha_t+\tau_t RM^\dagger(v_t)
\]

\[
\alpha_{t+1}=F_{st}(r_t,\theta_t).
\]

Initialization:

\[
\alpha_0=RM^\dagger(P(S_r\odot Mask)).
\]

Trainable parameters only:

\[
\{\tau_t,\theta_t\}_{t=1}^{9}.
\]

No CNN proximal, attention, learned measurement matrix, Gamma-Net support selection, or Toe-LISTA parameterization.

# 10. M6-R — Training (EXTERNALLY REVIEWED PASS)

Generate complex oracle images with random target positions and random phase \(U(0,2\pi)\). Generate echoes by:

\[
S_{\rm cplx}=Mask\odot RM(\alpha_{gt})+n.
\]

Default:

```text
training pairs: 1000
AWGN: 20 dB
optimizer: Adam
learning rate: 1e-4
phases: 9
tau init: 1e-4
theta init: 1e-3
epochs: 100
```

Batch size is paper-unspecified and must be labeled accordingly.

Use the RMIST averaged L2/RMSE-style reconstruction loss. NMSE/NRMSE/PSNR/TBR are evaluation metrics, not the strict training loss.

# 11. Paper-style synthetic benchmark

Where practical, later reproduce:

```text
image: 512 x 512
targets: 100 random points
target intensity: 1
target distance: 0.5 m
sampling rates: 0.1, 0.3, 0.5, 0.7, 0.9
Monte Carlo trials: 50
```

Keep this separate from smaller smoke tests.

# 12. M7-R — Hammer inference (TECHNICAL PASS / REAL-DATA SUCCESS — REVIEW PENDING)

```text
Hammer raw
→ Rx1
→ validated M2 preprocessing/range FFT
→ range-focused slices
→ mask/pad
→ RMIST independently per range slice
→ stack slices
→ 3-D Hammer
```

Each slice uses its range-dependent RM/RM_dagger kernels. Hammer RMA must never become supervised GT.

# 13. Deviation ledger

Every deviation from RMIST must be recorded in `DECISIONS.md` as one of:

```text
SYSTEM ADAPTATION
PAPER UNSPECIFIED IMPLEMENTATION CHOICE
EXPERIMENTAL EXTENSION
```

No silent deviations.

# 14. Stop conditions

Stop for human review if:

1. M2 consistency would require changing the `1/ky` / `ky` RMIST structure;
2. signs cannot be reconciled while preserving the reciprocal RMIST pair;
3. a missing paper parameter materially changes the method;
4. Eq. (19) cannot be reproduced without replacing `RM_dagger`;
5. Codex proposes another unfolding architecture;
6. RMA-derived real labels would be needed for training;
7. M3-R/M4-R cannot pass after reasonable debugging.

Routine code/CUDA/path/plotting failures are not stop conditions.
