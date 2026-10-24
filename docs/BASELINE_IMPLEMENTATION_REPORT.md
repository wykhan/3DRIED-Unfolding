# Baseline implementation report — 2026-10-24

Status: **IMPLEMENTED / POINT_SIM_COMPLETED / REAL_BLOCKED_MISSING_DATA**.
This is a first baseline implementation and numerical validation, not an
independent scientific acceptance of all algorithms or real-data performance.

## Delivered

- Five explicitly named methods: BP, RMA, accepted Ini-ISTA, classical
  complex-valued ISTA, and fixed accepted M6 RMIST.
- One shared 3DRIED acquisition config and an independent parameter document.
- Both point simulation and a read-only real Hammer entry point.
- Nested exact-count random-2D and height-line masks at 100/70/50/30%.
- Complex volumes, GT policy, observed-domain diagnostics, timings, MIP,
  representative range slices, 3-D point clouds, masks and energy profiles.
- One CLI; the original accepted scripts and kernel/model files are unchanged.

## Evidence

`python -m pytest -q`: **46 passed, 3 skipped** on CPU. Two skips require CUDA;
one requires the original Hammer MAT. A pre-existing PyTorch warning about
converting a grad-enabled scalar appears in an unchanged regression test.

Meaningful new checks cover shared configuration vs the accepted Hammer
configuration, mask counts/nesting, physical-only support, a masked adjoint
inner-product identity, RMA/M2 parity, NumPy/PyTorch fixed RMIST parity,
CV-ISTA objective decrease and complex phase equivariance, geometric BP
linear-convolution/direct-sum equivalence, an independent spherical-point
focal-response check, simulator GT/crop correctness, missing-data reporting,
reference cosine precision/range, and all 80 small-fixture matrix entries. The HDF5 fixture also verifies
that unused Rx channels are not loaded and source bytes are unchanged.
The HDF5 fixture is generated test input, not an actual 3DRIED measurement.

The full 200×407 aperture / 512×512 image / 13-range-bin point experiment
completed all **40 rows**. Its configuration fingerprint is
`043903c8833b99ab7e21bf31222f9435b4d19697872ebc44c3d70709bc878855`.
Selected metrics, configuration, provenance, masks and figures are preserved
under `results/review/BASELINE/`. Complete complex volumes are reproducible
with the runbook; bulky generated volumes are not committed to Git.

## What the evidence does not establish

1. **Actual Hammer remains unexecuted in this Work container.** The original
   capture `/home/superws/dataset/3DRIED/Hammer/Hammer (1)/output_rawdata.mat`
   is absent. The runner returns exit code 2 and overall INCOMPLETE when it
   requests both data sources. This is not masked by successful synthetic
   results or skipped integration tests.
2. **The point simulator is model-matched RM, not full physical FMCW.** It
   uses the real system geometry and spectral/range conventions. It does not
   model ADC dechirping or arbitrary cross-range range migration.
3. **BP is slice-wise narrowband spherical-wave BP.** It is a genuine
   geometric spatial sum accelerated by linear convolution, with fixed
   full-aperture sensitivity normalization. It does not share exactly the
   periodic/truncated RM forward used for synthetic data. Its complex GT
   errors and RM residuals include that model mismatch.
4. **CV-ISTA is a documented provisional name/definition.** No CV-ISTA paper
   was supplied. It currently means classical masked complex-L1 ISTA. The
   global safe spectral step is conservative: max(|Phi|²) is about 10277 times
   the median propagating value at this geometry. A 100-step run is a fixed
   budget, not a convergence claim, and currently reconstructs weakly. Do not
   conclude from this budget that CV-ISTA intrinsically underperforms.
5. **Historical ISTA/RMIST residuals remain unchanged.** They compare the
   predicted full padded echo with zero-filled measurements, rather than
   using a newly substituted masked gradient. CV-ISTA uses the true masked
   adjoint separately. Their objectives are therefore different.
6. **RMIST is not retrained across SR/pattern.** All new masks differ from the
   historical M6 mask. SR!=0.5 and height-line conditions add further OOD.
   The accepted training range was 0.5 m; this matrix also probes other ranges.
7. **No real GT is assumed.** RMA100 is a reference proxy only. The accepted
   one-based range convention and its one-bin offset from canonical FFT
   coordinates remain documented and unchanged.
8. **No independent review has been performed on this new extension.** M9
   remains separate from this implementation. Main is unchanged.

## Server continuation

A connected remote project should open the same repository and branch, then
run the full command in `docs/BASELINE_RUNBOOK.md`. Verify the actual raw MAT,
restore the established Python environment, and compare RMA100 to M2 before
interpreting the new real-method outputs. The unified runner currently uses
CPU NumPy/SciPy; access to a GPU is useful for existing PyTorch tasks but is
not a prerequisite for this inference matrix.

During final QA, the reference cosine computed from complex64 reductions could
slightly exceed one. Metrics now accumulate in float64 and bound the cosine
to [0,1]; a regression test covers this. Reconstruction arrays are unchanged.
