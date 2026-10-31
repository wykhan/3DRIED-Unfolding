# FMCW BP: fixes requested by independent review

Reviewed baseline: `60f447263128d576b24589b77ee5dd2f62f62bf0`, draft PR #2.
Date: 2026-10-27, Asia/Shanghai. Status: **IMPLEMENTED AND TESTED; AWAITING
INDEPENDENT RE-REVIEW**. The PR remains draft; no merge or Ready action.

Scope remains the declared positive-beat, stop-and-go FMCW simulation and
performance assessment. Hammer mixer sign, ADC sample times, delay and
absolute distance mapping have not been accepted. No new Hammer BP inference
or scientific acceptance is claimed.

## Requested corrections and evidence

| Item | Correction | Executed evidence |
|---|---|---|
| 1. FFT length misuse | Store constructor n_fft and require profile range length to match before any launch, for every mode; CPU reference checks the same contract | `test_all_modes_reject_different_fft_lengths`: 1024 and 2049 rows rejected for a 2048 operator, double/mixed/cached_double; CPU mismatch test also PASS |
| 2. Current CUDA device | Allocate output, obtain operator-device current stream, and launch ctypes kernel inside `torch.cuda.device(self.device)`; resolve unindexed CUDA devices at construction | `test_launch_uses_operator_device_and_restores_other_current_device`: owner/current 0/1 and 1/0, three modes, non-default owner streams, matching results and restored caller device; both RTX 4090s available, no hardware skip |
| 3. Range endpoints | Closed native FFT interval [0,n_fft-1], exact last sample uses final interpolation interval at weight one; outside contributes zero | CPU and all three CUDA modes test first, exact last, below-first and above-last. No extrapolation, circular wrap or tolerance-based clamping |
| 4. Live environment | Shared collector records fresh process/time/capture ID, interpreter/libraries, hardware, driver, UUID/utilization/memory and compiler. Cache records reference environment hash separately | New full and cache environments have different capture IDs/timestamps. Unit test checks fresh IDs. The old environment is never used as the current snapshot |
| 5. Independent physics reference | Direct complex raw-ADC matched summation, with no FFT, centered-profile helper or interpolation. Separate interpolation and implementation errors; no fitted phase/gain | 60 depth/query/oversampling cases; CPU and ordinary CUDA PASS; 30 cached lattice-query cases PASS. 30 off-lattice query sets explicitly SKIPPED for the lattice cache, while generic CUDA still runs them |

Full suite: **85 PASS, 0 FAIL, 0 ERROR, 0 SKIPPED**. The two-GPU test has an
explicit unittest SKIPPED reason on hardware with fewer than two CUDA devices;
it ran successfully on this server. The 30 off-lattice cache exclusions are
experiment geometry exclusions, not hardware skips or missing generic BP.

## Independent ADC reference and error separation

`simulation/fmcw_adc_reference.py` directly evaluates the conjugate expected
ADC waveform for each candidate point and sums raw samples over fast time
and aperture. Its fixed 1/N_ADC normalization matches the existing contract.
It does not call range FFT, profile interpolation, f_ref phase compensation
helpers or the CPU/CUDA reconstruction. A known single complex scatterer is
also checked at its exact off-grid position against aperture_count*rho.

The saved matrix uses 25 aperture positions on the nominal 3DRIED lattice,
off-grid scattering coordinates, depths 0.43/0.61431/0.83 m, separate single
and simultaneous multi-depth target scenes, regular query grids and genuinely
off-grid query positions, and FFT oversampling 1/2/4/8/16. Parameters and
limits were fixed in `configs/fmcw_bp_review.yaml` before the experiment.

For each case, records distinguish:

- **Interpolation error:** CPU complex linear interpolation minus direct ADC
  matched summation, before any CUDA implementation is involved.
- **Implementation error:** each CUDA output minus that same CPU interpolated
  formula; FP64 and mixed precision have separately declared limits.
- **Total error:** each CUDA output minus direct ADC matching.
- Absolute amplitude error and complex phase difference. Phase is reported
  only where ADC magnitude is at least the predetermined 5% of the known
  signal bound; counts are saved. Complex-error metrics retain every query.

There is no empirical image rescaling, global phase alignment or coefficient
fit. The known signal bound is aperture_count*sum(abs(rho)); it is specified
from the simulation inputs, not fitted to reconstruction output.

Maximum implementation relative L2 errors over the matrix:

| Backend | Relative to CPU interpolated formula |
|---|---:|
| CUDA FP64 | 7.516e-13 |
| Cached CUDA FP64 | 3.098e-13 |
| CUDA mixed | 2.661e-4 |

Interpolation is a separate, observable approximation:

| FFT oversampling | Max relative L2 error vs ADC | Max amplitude absolute error / known signal bound | Max significant-reference phase error (rad) |
|---|---:|---:|---:|
| 1 | 0.347305 | 0.279295 | 0.762669 |
| 2 | 0.123601 | 0.087235 | 0.132722 |
| 4 | 0.036096 | 0.018480 | 0.032459 |
| 8 | 0.006105 | 0.004954 | 0.008119 |
| 16 | 0.002144 | 0.001130 | 0.001372 |

The default remains 8x; no parameter search or post-result change to this
default was made. Low oversampling is not presented as accurate merely
because CUDA agrees with the interpolated CPU implementation.

For the centered finite ADC transform, linear interpolation has the absolute
curvature bound pi^2*(1-1/N_ADC^2)/(24*oversampling^2), after division by the
known signal bound. It follows from the second frequency derivative of the
finite sum and interval error <= max|g''|*delta_f^2/8. Every case satisfies
this bound. This is an interpolation bound under the declared simulation,
not a real-data imaging guarantee or a relative-error bound at signal nulls.

## Repeated performance and historical regression

Both CUDA kernels' bounds handling and the Python launch path changed, so the
general benchmark and exact-cache benchmark were rerun at full 200x407
aperture /512x512 output, one/five-point scenes, all four height rates.
RTX 4090 GPU0, ComplexValue, two warmups/five synchronized repetitions,
medians below. No other experiment ran concurrently on GPU0.

| Rate | General FP64 core | Cached FP64 core |
|---|---:|---:|
| 100% | 2.463--2.476 s | 0.376--0.394 s |
| 70% | 1.730--1.733 s | 0.297--0.301 s |
| 50% | 1.238--1.238 s | 0.216--0.218 s |
| 20% | 0.496--0.496 s | 0.0831--0.0836 s |

Full-aperture single-point pipeline (ADC upload, padded range FFT, centering,
BP, image download): FP64 general 2.543 s; mixed 1.558 s; FP64 cache 0.454 s.
Cache construction is separate, 54.7--85.4 ms across conditions. Compilation,
context initialization, disk I/O and waveform synthesis remain excluded.

All eight cached outputs agree with the new general FP64 outputs, and with
the preserved pre-review full-size images, within **1.436e-13 relative L2**.
Both new and historical reference source hashes/image hashes are saved by
the cache benchmark. No historical result is overwritten or relabeled.

The historical `results/review/FMCW_BP/cache_environment.json` was copied from
its reference run by the old script. It is retained as historical evidence
of that provenance defect and must not be treated as a freshly captured
cache-run hardware state. New cache environment collection is live.

The new full/cache runs have independently generated capture IDs; the raw
identifiers remain in local records and are excluded from the public packet.
Full/cache capture UTC timestamps: 2026-10-27T14:35:47.302282+00:00 and
2026-10-27T14:39:13.607080+00:00. Compiler and PyTorch CUDA build versions are
recorded separately; torch.version.cuda describes the PyTorch build.

## Records and reproduction

Full local records: `results/fmcw_bp_review_20261027/`.
Public evidence: `results/review/FMCW_BP_REVIEW/public_*` contains source
SHA256s, test counts/log, aggregate synthetic errors/timings and freshly
collected environment summaries without host identifiers, GPU UUIDs or
absolute paths. Full commands, environments, per-case errors, timing samples
and historical correspondence remain in the server records above; the full
packet was rejected for public upload by automatic approval review because
of identifying environment metadata. The complete local records are intact.
Complex arrays, new figures, compiled libraries and large outputs remain
local. An early small-matrix run is preserved locally with
its executed-source snapshots; the final matrix is `adc_validation_v2`.

```bash
conda activate ComplexValue
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
python scripts/validate_fmcw_bp_review.py --output results/review_reproduce/adc
python scripts/benchmark_fmcw_bp.py --output results/review_reproduce/full
python scripts/benchmark_fmcw_bp_cache.py --reference results/review_reproduce/full \
  --historical-reference results/fmcw_bp_assessment_20261027/run \
  --output results/review_reproduce/cache
```

The historical-reference argument requires the retained local original
images; omit only that optional argument on a fresh clone without them.
Fresh output directories are required. The recorded review wrapper also
saves machine-readable unittest counts and full test/source provenance.

Protected source/config comparison with 60f4472 and accepted raw/weight SHA256
checks pass: RMIST kernels, network and accepted weights, historical geometry
and loader, shared physics configs and raw Hammer data remain unchanged.
Scientific acceptance awaits another independent review.
