# RESEARCH_SPEC.md

## 1. Project title

**Physics-Based ISTA Unfolding Baseline for 3DRIED 3-D Reconstruction**

---

## 2. Immediate objective

Construct a minimal, trustworthy, reproducible learning-imaging baseline for the 3DRIED dataset, beginning with the Hammer target.

The project is intentionally baseline-oriented.

The first priority is correctness of:

1. real-data loading;
2. single-Rx preprocessing;
3. conventional RMA reconstruction;
4. later, the forward/adjoint physics operator;
5. later, Kernel-ISTA;
6. later, learned ISTA unfolding.

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
2-D near-field holographic operator per range slice
  ↓
Kernel-ISTA
  ↓
ISTA-Unfolding
  ↓
3-D complex scattering volume
  ↓
magnitude / dB / MIP / 3-D visualization
```

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

## RMIST reproduction basis

All learning-related work from M3-R onward follows `docs/RMIST_REPRO_SPEC.md`.

The strict baseline is RMIST, not the previous generic ISTA-unfolding design:

- RM forward kernel with `1/ky`;
- RM_dagger inverse kernel with `ky`;
- Eq. (19) RM-kernel ISTA;
- Eq. (20) phase-wise `tau_t`, `theta_t`;
- 9 phases;
- synthetic RM-generated training echoes;
- 1000 training pairs;
- default 20 dB AWGN;
- Adam, lr=1e-4;
- tau init=1e-4;
- theta init=1e-3;
- 100 epochs;
- RMIST reconstruction loss.

For 3DRIED, the propagation sign is mapped so `RM_dagger` reproduces the validated M2 RMA. This is a system-coordinate adaptation, not an algorithm change.


## 9. Planned ISTA-Unfolding baseline

The learning model will use the same physics operator.

For layer \(t\):

\[
r_t = y-Ax_t
\]

\[
z_t=x_t+\tau_t A^H r_t
\]

\[
x_{t+1}=\mathcal S_{\theta_t}(z_t).
\]

Only the following are trainable in the minimal baseline:

\[
\{\tau_t,\theta_t\}_{t=1}^{T}.
\]

Initial depth target:

```text
T = 5
```

Increase only if evidence shows five layers are insufficient.

No learned measurement matrix is used.

No CNN proximal module is used.

---

## 10. Planned synthetic training data

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

Training input is generated through the physics operator:

\[
y=A(x_{\rm gt})+n.
\]

Initial data target:

```text
train samples: 1000
validation samples: 200
default SNR: 20 dB
```

The first version uses point-like scatterers only.

Structured surface-like targets are deferred unless a clear synthetic-to-real failure is observed.

---

## 11. Planned loss

Initial loss:

\[
\mathcal L_{\rm NMSE}
=
\frac{
\|\hat x-x_{\rm gt}\|_2^2
}{
\|x_{\rm gt}\|_2^2+\epsilon
}.
\]

Do not add perceptual, GAN, SSIM, or self-supervised losses in the baseline version.

---

## 12. Planned evaluation

### Synthetic

- NMSE
- PSNR
- SSIM
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
\|A\hat x-y\|_2
}{
\|y\|_2
}.
\]

---

## 13. Success criteria for the complete baseline

The baseline is considered successful when:

1. Hammer is reconstructed correctly by conventional RMA;
2. forward/adjoint operator passes \(e_{\rm adj}<10^{-5}\);
3. Kernel-ISTA recovers synthetic sparse targets;
4. 5-10 layer ISTA-Unfolding improves over simple adjoint reconstruction and approaches a multi-iteration Kernel-ISTA baseline;
5. the trained model produces a recognizable Hammer reconstruction without catastrophic synthetic-to-real failure.

---

## 14. Explicit non-goals

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
