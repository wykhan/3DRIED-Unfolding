# RESEARCH_SPEC.md

## 1. Project title

**Strict RMIST-Net Reproduction Baseline for 3DRIED 3-D Reconstruction**

---

## 2. Immediate objective

Construct a minimal, trustworthy, reproducible learning-imaging baseline for the 3DRIED dataset, beginning with the Hammer target.

The project is intentionally baseline-oriented.

The completed foundation establishes:

1. real-data loading;
2. single-Rx preprocessing;
3. conventional RMA reconstruction;
4. strict RMIST `RM` / `RM_dagger` kernels;
5. strict RM-kernel ISTA Eq. (16)-(19).

RMIST-Net Eq. (20)-(26) is independently reviewed PASS. M6-R strict synthetic
training Eq. (27)-(31) is self-validated PASS and awaiting independent review;
M7-R remains locked.

Novelty is not an objective of this baseline stage.

---

## 3. Real dataset

Target:

```text
3DRIED / Hammer
```

Server path:

```text
/home/superws/dataset/3DRIED/Hammer/Hammer (1)
```

### Fixed Rx decision

Only one Rx channel is used.

This mirrors the intended simplified reproduction strategy.

The other Rx channels are ignored for this baseline.

The exact selected Rx index must be logged.

---

## 4. Baseline system architecture

Final target architecture:

```text
raw echo
  ↓
single-Rx extraction
  ↓
range preprocessing / range FFT
  ↓
range-focused complex echo cube
  ↓
strict RM / RM_dagger kernels per range slice
  ↓
strict RM-kernel ISTA Eq. (16)-(19)
  ↓
RMIST-Net Eq. (20)-(26)
  ↓
real Hammer slice-wise inference Eq. (32)
  ↓
3-D complex scattering volume
  ↓
magnitude / dB / MIP / 3-D visualization
```

When separately authorized, M6-R trains the M5-R architecture with the
synthetic protocol in Eq. (27)-(31); its accepted weights then feed the
Eq. (32) inference path. Training is not an operation on the real Hammer
echo and Hammer RMA is not a label.

---

## 5. Conventional RMA baseline

The first scientific baseline is conventional near-field RMA/holographic imaging.

Coordinates:

- x: horizontal synthetic-aperture direction
- z: vertical synthetic-aperture direction
- y: range direction

For each range slice `r`, the reconstruction follows an FFT-domain phase-compensation form.

A generic slice model is:

\[
k = \frac{2\pi f_c}{c}
\]

\[
k_y = \sqrt{4k^2-k_x^2-k_z^2}
\]

and reconstruction uses a 2-D Fourier transform with a phase-compensation term dependent on the range coordinate \(y_r\).

The implementation must follow the actual 3DRIED acquisition parameters and data layout discovered during M1.

Do not hard-code unverified dimensions or distances.

---

## 6. RMIST reproduction basis

All learning-related work from M3-R onward follows `docs/RMIST_REPRO_SPEC.md`.

The strict baseline is RMIST, not the previous generic ISTA-unfolding design:

- RM forward kernel with `1/ky`;
- RM_dagger inverse kernel with `ky`;
- Eq. (19) RM-kernel ISTA;
- Eq. (20)-(26) RMIST-Net with only phase-wise `tau_t`, `theta_t` trainable;
- 9 phases;
- synthetic RM-generated training echoes;
- 1000 training pairs;
- default 20 dB AWGN;
- Adam, lr=1e-4;
- tau init=1e-4;
- theta init=1e-3;
- 100 epochs;
- RMIST averaged L2/RMSE-style reconstruction loss;
- batch size remains paper-unspecified.

For 3DRIED, the propagation sign is mapped so `RM_dagger` reproduces the
validated M2 RMA. This system-coordinate adaptation has passed independent
review and does not change the reciprocal RMIST amplitude structure.

The strict network has 9 phases. Its only trainable parameters are
`{tau_t, theta_t}` for each phase. `tau_t` initializes to `1e-4` and
`theta_t` to `1e-3`. No learned measurement matrix, CNN proximal module,
Gamma-Net, or Toe-LISTA parameterization is allowed.

---

## 7. Authorized synthetic training data (M6-R)

Training does not use the Hammer RMA image as ground truth.

Synthetic complex scattering images will be generated from random scatterers.

For scatterer \(i\):

\[
\gamma_i=a_i e^{j\phi_i}
\]

with:

\[
\phi_i\sim U(0,2\pi).
\]

Amplitude scale may be calibrated from real RMA reconstruction statistics, but the real RMA image is not a supervised label.

Training input is generated through the strict RM kernel:

\[
S_{\rm cplx}=Mask\odot RM(\alpha_{\rm gt})+n.
\]

Strict RMIST training settings:

```text
train samples: 1000
default SNR: 20 dB
phases: 9
optimizer: Adam
learning rate: 1e-4
epochs: 100
tau_t initialization: 1e-4
theta_t initialization: 1e-3
batch size: paper-unspecified
```

The first version uses point-like scatterers only.

Structured surface-like targets are deferred unless a clear synthetic-to-real failure is observed.

---

## 8. Authorized loss (M6-R)

Use RMIST's averaged L2/RMSE-style reconstruction loss defined by Eq. (27).
NMSE/NRMSE, PSNR, and TBR are evaluation metrics, not the strict training
loss.

Do not add perceptual, GAN, SSIM, or self-supervised losses in the baseline version.

---

## 9. Planned evaluation

### Synthetic

- NMSE
- PSNR
- TBR
- runtime
- peak GPU memory

### Real Hammer

No voxel-level ground truth is assumed.

Use:

- qualitative comparison with conventional RMA;
- entropy;
- contrast;
- measurement consistency:

\[
E_{\rm echo}
=
\frac{
\|RM(\hat\alpha)-S\|_2
}{
\|S\|_2
}.
\]

---

## 10. Milestone state and success criteria

The baseline is considered successful when:

1. Hammer is reconstructed correctly by conventional RMA;
2. strict `RM` / `RM_dagger` reproduces the reciprocal RMIST kernels and M2;
3. strict Eq. (19) RM-kernel ISTA recovers synthetic sparse targets;
4. when authorized, 9-phase RMIST-Net reproduces Eq. (20)-(26);
5. when authorized, training reproduces Eq. (27)-(31);
6. when authorized, Eq. (32) produces a recognizable Hammer reconstruction
   without using Hammer RMA as supervised ground truth.

M0-M2 are human-reviewed. M3-R, M4-R, and M5-R are independently reviewed and
accepted. M6-R synthetic training is self-validated PASS and awaiting
independent review. M7-R remains locked.

---

## 11. Explicit non-goals

This baseline does not attempt to solve:

- sparse-height sampling;
- multi-Rx fusion;
- system calibration;
- motion-error compensation;
- domain adaptation;
- self-supervision;
- learned forward operators;
- Gamma-Net reproduction;
- Toe-LISTA reproduction;
- publication-level novelty.
