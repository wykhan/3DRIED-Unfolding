# Baseline review record

**Full FMCW BP follow-up:** Now authorized by the user. 77 regression tests
PASS; eight full-size FMCW single/five-point controls at 100/70/50/20% have
zero grid localization error. FP64 CUDA/NumPy spot error <=5.27e-14;
cached/full FP64 whole-image error <=1.44e-13. Exact regular-lattice caching
retains migrated range sampling, giving 0.374--0.405 s for the full slice
core, with a warmed 0.448 s ADC-upload/range-FFT/BP/download measurement.
Review the declared positive-beat IF, FFT centering, interpolation and RVP
phase in docs/FMCW_BP_ASSESSMENT.md. This is physical simulation, not Hammer
hardware-phase/calibration acceptance. The legacy narrowband entry requires
explicit diagnostic selection and cannot substitute for full FMCW BP.

**Latest BP follow-up:** The user authorizes FFT acceleration, caching and
GPU execution. The earlier literal-only requirement below is superseded.
Five new regression tests (68 total PASS), full-size direct-sum spot checks
and CPU/CUDA comparisons support numerical equivalence for fixed-depth
phase-only BP. Eight one/five-point controls at 100/70/50/20% complete.
Review [BP_ACCELERATION_REPORT.md](docs/BP_ACCELERATION_REPORT.md) for timing
boundaries and remaining raw-FMCW/Hammer limitations. Legacy Green weighting
and sensitivity normalization are not used by the new accelerated operator.

Status: **SELF-REVIEW COMPLETE FOR DECLARED PARTIAL MATRIX; INDEPENDENT REVIEW PENDING.**

## Inherited evidence

Preflight ran inside `tmux 3dried / ComplexValue`. Historical baseline source
is a94b4f62d1674e1b00733c6822cc91cdf503d224. All 49 legacy tests passed with
CUDA and the real MAT present. Full Hammer RMA differs from saved M2 by
3.3168107200130116e-8; raw SHA-256 remains
167c5977d75deddac782710c287d5bd49cffefe04a67e0c88fa45ab8e30a7475.
Full-size adjoint checks passed at every rate and both mask families.

The two Python 3.8 compatibility repairs (typing.Union and AST source-segment
inspection) do not change any imaging operation. The M9 files are retained
for negative-result provenance, without treating M9 as independently accepted.

## New tests

The initial 61-test suite passed in ComplexValue, including complex128
adjoint/autograd equality, physical-unit operator scaling, every-phase
network gradients, a 30-step single-scene optimization improvement, split
hash disjointness, classical ISTA objective descent, and literal BP coherent
summation. Later tests and experiment evidence are recorded after execution.

## Required review concerns

1. Original ISTA-Net topology is preserved; initializer and initialization
   choices are explicit system adaptations. This is not the paper's image-CS
   training setup or a claim to reproduce its image benchmark tables.
2. RM simulation matches the solver model. Independent test seeds establish
   held-out scenes within that distribution, not physical or real generality.
3. The global spectral Lipschitz bound is conservative. A finite 200-step
   ISTA comparison is not a convergence certificate; objective/iteration
   limitations must accompany any performance statement.
4. The literal BP reference retains direct loops. The now-authorized FFT
   backend must agree with its complex sum, preserve linear-convolution
   boundaries and disclose its fixed-depth narrowband scope. Partial images
   and legacy Green-weighted results cannot substitute for these checks.
5. M6 RMIST weights were trained with a different fixed Bernoulli mask. New
   exact-count height-line and random-2D masks are unvalidated even at 50%.
6. Never use Hammer images as labels, rescale on Hammer to optimize metrics,
   or interpret descriptive entropy as a universally better reconstruction.
7. Common coordinates, display window and a single dB reference per dataset
   must be fixed across all rates and methods. No phase alignment, gain fit
   or per-image enhancement is used before Complex NMSE.
8. Timings use declared hardware, synchronization, warmups and repetitions,
   exclude preprocessing and file I/O, and require an idle GPU for comparisons.

No main branch merge is authorized. Review the final implementation and
evidence before accepting scientific claims.

## Final evidence and open acceptance gates

- Final suite: 63 tests PASS in ComplexValue, including CUDA and actual MAT.
- Four separate height-rate checkpoints completed 20 epochs each, with
  positive finite phase steps/thresholds, validation selection and disjoint test.
- 100 matrix rows: 54 technically completed, 46 honest SKIPPED conditions.
- Six full-observation cross-pattern comparisons are bit-identical; 81,400
  positions retained. No gain/phase fitting before Complex NMSE.
- Raw data hash and M2 parity match preflight; no Hammer GT is assumed.
- Hammer learned quality is NOT DEMONSTRATED: at 50%, ISTA-Net residual is
  1.57259 versus RMA 0.05518. No tuning follows this observation.
- Literal full-FOV BP acceptance is open: roughly 54 CPU-hours by local
  extrapolation, plus unresolved raw phase/range convention. It is SKIPPED,
  with no fast BP or partial image substituted.
- New-mask RMIST generalization remains UNVALIDATED and SKIPPED. M6/M7
  acceptance is retained; M9 normalizers are not adopted.
- 2-D random ISTA-Net at 70/50/30% is untrained and SKIPPED. 100% reuse is
  justified by identical observations, not a generalization claim.
- ISTA200 is underconverged; the 20000-step validation diagnostic improves
  substantially and still does not certify convergence.
- The fixed two-point controls do not measure physical system resolution.

Full five-method scientific ranking, full FMCW physical acceptance, general
Hammer superiority and independent review remain open. The report does not
mark those claims PASS.
