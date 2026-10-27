# Exact phase-only BP acceleration: 3DRIED single-slice benchmark

Date: 2026-10-27, Asia/Shanghai. Branch: `baseline/ista-net-validation`.
The user explicitly replaces the mandatory unoptimized-BP requirement with
an optimized BP baseline. RMIST, its weights and the SAR learning operators
are unchanged.

## Formula and scope

At fixed depth y, on the shared regular planar lattice:

```
R(p,a) = sqrt((x_p-x_a)^2 + y^2 + (z_p-z_a)^2)
BP(p) = sum_{a in observed aperture} echo(a) * exp(-j*4*pi*fc*R(p,a)/c)
```

The phase depends only on spatial coordinate differences. Consequently this
same discrete sum is a linear convolution. The new `fft_bp.py` evaluates it
without far-field approximation, 1/R amplitude weighting, sensitivity
normalization, learned filtering or postprocessing. Complex128 is retained
on both CPU and GPU; speed does not rely on reducing phase precision.

The old `baseline/methods.py:backproject` also uses FFT convolution but has
a Green 1/R weight and full-aperture sensitivity normalization. Those are
separate model/gain choices. That historical function remains unchanged and
is not the new phase-only BP implementation.

This is fixed-depth, single-frequency spherical-wave BP. It does not perform
geometric range interpolation through raw FMCW profiles. The accepted legacy
one-bin range-label discrepancy is not changed, and full FMCW/Hammer BP has
not been promoted to PASS by this experiment.

## What is optimized

| Change | Effect | Numerical consequence |
|---|---|---|
| FFT linear convolution | Replaces pixel/aperture nested Python sums with FFT/multiply/inverse FFT | Same phase-only sum up to roundoff |
| Full linear padding and explicit crop | Avoids circular wrap at image boundaries | Not a periodic propagation approximation |
| Kernel-spectrum cache per depth/geometry/carrier | Reuses distances, complex phases and kernel FFT | No change for a fixed configuration |
| CUDA FFT backend | Executes the same FFT/multiply/inverse FFT on GPU | Complex128 CPU/CUDA agreement tested |

For a 512x512 image, the kernel is 1023x1023. Full convolution needs at least
1534 samples per dimension; this implementation uses FFT-friendly 1536x1536.
The observed 200x407 aperture is placed at its accepted image-grid indices.
Masking sets unobserved coefficients to zero, which exactly omits those terms
from the BP sum. This is not interpolation or recovery of missing echoes and
does not claim a general off-grid/nonuniform-aperture FFT imaging method.

## Acquisition and simulation

- Physical aperture: z=200, x=407; dz=2 mm, dx=1 mm; Rx1 convention.
- Output: full z=512, x=512, preserving the accepted coordinates and offsets.
- Carrier: 78.8 GHz; c=299792458 m/s.
- Depth: 0.6143103952815643 m, the accepted reference bin-17 coordinate.
- Independent analytic echo: sum rho_q*exp(+j*4*pi*fc*R(a,q)/c).
- One target and five separated complex targets, all at this same depth.
- Noiseless; no RM-generated echo, raw ADC synthesis or empirical scaling.
- Height-row sampling: 100/70/50/20%, 200/140/100/40 rows,
  81400/56980/40700/16280 aperture positions, seed 20261024.
- Nested masks shared between the two scenes and every backend.
- Full coordinates, target amplitudes/phases and settings are saved.

The five points are a fixed localization control, not a resolution-limit
study. Simulations use grid-aligned targets and matched spherical propagation.

## Measured single-slice time

Environment: ComplexValue, Python 3.8.17, SciPy 1.10.1, PyTorch 2.0.1+cu118;
Intel Xeon Gold 5317 @ 3 GHz (one SciPy FFT worker), NVIDIA RTX 4090.
GPU0 was idle at measurement start. The run used the existing `3dried` tmux
session. Two warmups, ten repetitions per condition; median wall time, CUDA
synchronized. No data loading, echo synthesis, range preprocessing or plotting
is included.

| Implementation | Time per 512x512 slice across the eight controls |
|---|---:|
| CPU, kernel and kernel FFT recomputed within each timed call | 203.84--206.59 ms |
| CPU, cached kernel spectrum | 111.29--113.42 ms |
| RTX 4090, cached kernel, including host echo upload and image download | 2.036--2.125 ms |
| RTX 4090, cached kernel, input/output already on GPU | 1.385--1.388 ms |

These are warmed timings, not application startup latency. One-time kernel
preparation was measured separately: CPU 127.46 ms, GPU 79.67 ms (GPU includes
CPU kernel generation/upload but excludes CUDA context initialization).
The kernel must be prepared again when depth, geometry or carrier changes.

At 100% on the single-point control, the medians are 204.54 ms (CPU uncached),
113.41 ms (CPU cached), 2.125 ms (GPU with transfers) and 1.388 ms (GPU resident).
The earlier ~4.1-hour literal full-slice value was a Python-loop extrapolation,
not a completed full-slice benchmark; no measured speedup ratio is asserted.

The FFT domain stays the same at every sampling rate. Therefore lower rates
do not reduce this implementation's runtime in proportion to observation
count. These are single-slice latency measurements, not measured sustained
multi-slice throughput or full raw-FMCW reconstruction time.

## Numerical and image evidence

- All 68 project regression tests pass, including five new FFT-BP tests.
- Small-grid tests compare every output pixel with literal direct sums,
  including shifted rectangular grids, both mask families and boundary points.
- Full-size direct-sum spot checks include targets and opposite image edges
  for every scene/rate. Maximum relative error is 1.476e-12.
- Maximum CPU/CUDA complex image relative error is 8.70e-16.
- All eight single/five-point controls locate the assigned strongest peaks
  exactly on the target grid (zero grid localization error).
- One shared absolute dB reference, -40 to 0 dB, across all eight images.
  Reduced observation amplitude is retained; no per-rate gain compensation.
- Single-point height-axis PSLR worsens from about -17.61 dB at 100% to
  -11.05 dB at 20%. This is a fixed-scene result, not a universal guarantee.
- Unified entry smoke: BP, 20%, all 13 RM-generated slices, CUDA backend;
  finishes as DIAGNOSTIC, retaining the cross-model/gain warning.

All phase-coherent images, masks and large local artifacts remain in ignored
`results/bp_fft_slice_20261027/`. Review JSON/CSV/configs/test logs are under
`results/review/BP_FFT/`; no raw echoes or model checkpoints are committed.

## Reproduce

```bash
conda activate ComplexValue
PYTHONPATH=src python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_bp_slice.py --output results/bp_fft_slice_reproduce

# Unified operator interface; these RM-generated inputs are cross-model
# diagnostics, not the independent spherical-wave controls above.
python scripts/run_imaging_baseline.py --algorithm BP --dataset point_sim \
  --rates 1.0 0.7 0.5 0.2 --bp-backend fft_cuda --output results/bp_fft_unified
```

`--bp-backend fft_cpu` is the active default. `--bp-backend literal` retains
the bounded direct-loop reference. Full raw FMCW range-migration BP needs its
own physical phase/range validation and timing before a Hammer claim is made.
