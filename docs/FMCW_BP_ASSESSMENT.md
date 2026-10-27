# Full FMCW range-migration BP: optimization and slice assessment

Date: 2026-10-27 (Asia/Shanghai). The user explicitly selects full FMCW
range-migration BP for the 3DRIED baseline. Single-frequency FFT BP remains
a numerical control and its 2.1 ms timing is not used for the full method.

## Computational contract

Input is one Rx, a 200x407 physical planar aperture, 256 complex ADC samples
per aperture position. Output is one full 512x512 cross-range plane at
y=0.6143103952815643 m, preserving the accepted x/z coordinates. Height-row
masks at 100/70/50/20% retain 81400/56980/40700/16280 positions, seed 20261024.
Full sampling has 21,338,521,600 pixel/aperture contributions.

Every contribution computes the actual spherical slant range, queries the
complex range profile at that range, compensates its phase, and accumulates.
No fixed distance-bin echo, far-field approximation, aperture decimation,
adaptive weighting, or single-frequency phase-only input substitutes for it.

## Declared FMCW simulation and phase

For the positive-beat, stationary-during-chirp synthetic convention:

```
tau = 2*R/c + instrument_delay
s(a,t) = sum rho * exp(j*(2*pi*(f0+S*t)*tau - pi*S*tau**2))
t_n = n/fs, n=0,...,255
```

Parameters: fs=5 MHz, slope S=70.295 THz/s, delay=0.62516 ns, first-ADC-sample
frequency f0=77 GHz. Here f0 defines the synthetic waveform; this is not a
claim that metadata start frequency proves the actual Hammer ADC start time.
The exactly derived centered reference frequency is
f_ref=f0+S*(255)/(2*fs)=78.7925225 GHz. The accepted legacy RMA processing
frequency 78.8 GHz remains unchanged. These are explicitly distinguished,
not silently treated as interchangeable phase conventions.

The 256-sample IF is zero-padded to 2048 for range FFT. A phase factor centers
the profile at t_center=255/(2*fs) before complex linear interpolation. At a
queried delay tau, the compensation phase is
2*pi*f_ref*tau-pi*S*tau**2, including residual video phase. The native range
coordinate is c/2*(m*fs/(S*Nfft)-delay), not the legacy one-based RMA label.
Only the new FMCW path uses this native FFT coordinate; legacy operators
and labels are not modified.

Range grid spacing is 5.206 mm after zero padding; physical range resolution
is still 41.648 mm. Zero padding refines interpolation and adds no bandwidth.
Single and five separated complex targets are generated from this full IF
at the output depth. This is a model-matched physical simulation, not real
Hammer validation or a claim of independently calibrated hardware physics.

## Implemented optimization

| Optimization | Implementation | Effect on the imaging sum |
|---|---|---|
| GPU parallelism | One block per output pixel, 128 threads sharing aperture terms | Full aperture retained |
| Kernel fusion | Slant range, range interpolation, carrier/RVP compensation and accumulation in one CUDA kernel | No large intermediate tensors |
| Data layout and reduction | Range-major contiguous aperture profiles, block-local FP64 reduction, no global output atomics | Same complex sum up to roundoff |
| Reuse | Cache range-compressed profiles and coordinate vectors | Preprocessing reused across output slices |
| Exact lattice cache | Precompute range index/fraction and FP64 sine/cosine for relative spatial offsets | Full migrated range sampling retained; regular lattice verified |
| Optional mixed precision | FP32 geometry and trigonometry; FP64 interpolation and accumulation, complex128 storage | Small measured numerical approximation; checked against FP64 |

No --use_fast_math compiler flag is used. A full pixel-by-aperture distance
table alone would need 79.49 GiB in float32, exceeding a 24 GiB RTX 4090.
Full distance/phase matrices are not cached. For the verified regular lattice,
an offset table instead needs 20,909,088 bytes (19.94 MiB), because many
pixel/aperture pairs share the same offset. It stores the exact range-bin
index, interpolation fraction, cosine and sine. Each contribution still
reads the appropriate range samples from its own aperture profile. This is
not a single-frequency FFT convolution. Off-lattice positions are rejected;
the general fused kernel remains available. Rebuild the cache when depth,
carrier/chirp/delay parameters or lattice geometry changes. At full aperture,
the 2048-point complex128 range-profile buffer is about 2.48 GiB.

The standalone CUDA kernel is compiled with CUDA 12.4 for the detected GPU
architecture and receives PyTorch device pointers/current stream through a
small ctypes wrapper. This avoids a PyTorch/CUDA C++ extension ABI mismatch
with the installed PyTorch 2.0.1+cu118. Compilation and initial context setup
are excluded from warmed reconstruction timing and recorded separately in
the build command/log.

## Measured single-slice latency

Hardware: RTX 4090 GPU0, initially idle; CPU Xeon Gold 5317. Environment:
ComplexValue, Python 3.8.17, PyTorch 2.0.1+cu118, CUDA compiler 12.4.
Existing tmux session: 3dried. Two warmups/five synchronized wall-time repeats,
reporting medians. All output planes are 512x512; no display-window cropping
is used to reduce reconstruction cost.

Single-point control, BP core only with range profiles already on GPU:

| Height sampling | Positions | FP64 | Mixed precision |
|---|---:|---:|---:|
| 100% | 81400 | 2.405 s | 1.419 s |
| 70% | 56980 | 1.683 s | 0.993 s |
| 50% | 40700 | 1.202 s | 0.710 s |
| 20% | 16280 | 0.481 s | 0.284 s |

For the full-aperture single-point control, including raw ADC upload,
2048-point range FFT, profile centering, BP and output image download:
FP64 2.475 s; mixed precision 1.490 s. This excludes file loading, waveform
synthesis, compilation/context initialization and coordinate construction.
No CPU full-size BP latency or multi-GPU speedup is asserted.

The exact lattice-cache follow-up is faster while retaining FP64:

| Height sampling | Cached FP64 BP core, single/five-point range |
|---|---:|
| 100% | 0.374--0.405 s |
| 70% | 0.293--0.294 s |
| 50% | 0.211--0.212 s |
| 20% | 0.0818--0.0823 s |

Full-aperture cached FP64 including raw ADC upload, padded range FFT,
centering, BP and image download: **0.448 s** on the single-point control.
Cache construction/upload is excluded and separately measured at 44--64 ms
across the eight controls. Compilation/context initialization is also excluded.
The cache pipeline measurement is one warmed single-slice condition, not a
measurement of sustained multi-slice throughput. Use this exact FP64 cache
for the nominal regular 3DRIED lattice; mixed precision is not needed for
the preferred measured implementation.

These are actual full-size simulation measurements of a first fused GPU
implementation, not a claim of optimal achievable latency. Sparse direct BP
does less work in proportion to the retained aperture positions, unlike
the earlier fixed-size FFT control. Five-point input does not multiply the
number of BP contributions because the echoes have already superposed.

## Validation and scope

Tests cover centered FFT carrier/RVP phase, migrated distance-profile peaks,
complex linearity, CUDA/NumPy full complex sum agreement, sparse profile
selection, mixed precision and explicit isolation of the old narrowband path.
Full-size measurements compare CUDA with FP64 NumPy direct summation at
target locations and opposite image edges. A single-point coherent peak is
also checked against its known complex scattering coefficient (no fitted
phase or gain). Shared display uses one absolute dB reference across rates.

Final evidence: **77 tests PASS**, including nine new FMCW tests. All eight
single/five-point scenes locate every assigned target on its ground-truth
grid in FP64 and mixed precision. FP64 CUDA/NumPy spot relative error is at
most 5.27e-14; mixed/full FP64 image error is at most 6.87e-5. Cached/full
FP64 whole-image error is at most 1.44e-13. The single-point coherent peak
error is about 0.43%, consistent with the declared finite interpolation;
no empirical gain or phase fit is used. PyTorch's peak allocated memory in
the general-kernel pipeline measurement was 10.26 GiB (not total GPU usage).

The legacy RM-slice unified entry now defaults to the full-FMCW requirement
and reports SKIPPED for incompatible RM-generated slice inputs or unverified
Hammer phase conventions. The old control requires
`--bp-model narrowband_diagnostic` explicitly. The functional full FMCW
simulation/performance entry is `scripts/benchmark_fmcw_bp.py`.

Hammer remains NOT_RUN for this new method: actual mixer sign, ADC timing
and phase reference, instrument-delay interpretation and absolute range
mapping need verification against the capture/reference before scientific
acceptance. Simulation equality and 3DRIED-size timings do not close that
gate. Existing accepted RMA/RMIST operators and weights are unchanged.

## Reproduce

```bash
conda activate ComplexValue
PYTHONPATH=src python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_fmcw_bp.py --output results/fmcw_bp_reproduce
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_fmcw_bp_cache.py --reference results/fmcw_bp_reproduce \
  --output results/fmcw_bp_cache_reproduce
```

Configuration: `configs/fmcw_bp_benchmark.yaml`. Full artifacts and logs:
`results/fmcw_bp_assessment_20261027/`. Lightweight review records:
`results/review/FMCW_BP/`. Raw ADC simulation arrays, complex images and
compiled libraries remain ignored by Git.
The original general-kernel timing run's exact source snapshots are archived
with its provenance; the cache follow-up saves its own source hashes.

## References

- [Zaugg, Generalized Image Formation for Pulsed and LFM-CW SAR, §4.2](https://www.scp.byu.edu/long/theses/phddiss_zaugg.pdf): BP interpolates each aperture range profile at the pixel's slant range and compensates the expected FMCW phase.
- [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): data layout, transfer costs, device reuse and precision/performance tradeoffs.
