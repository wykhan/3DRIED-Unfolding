# REVIEW_CHECKLIST.md

# 3DRIED-Unfolding Review Gates

## 1. Purpose

This file defines the independent review gates for milestone completion.

The review process has three levels:

1. **Codex self-review** — implementation, tests, experiment execution, documentation.
2. **Independent ChatGPT review** — inspect GitHub code, reports, tests, and experiment evidence.
3. **Human scientific decision** — approve merge/unlock of the next research milestone.

A milestone is not considered externally accepted merely because Codex marks it PASS.

## 2. Repository-level checks for every milestone

Before requesting external review, verify:

- [ ] Work is on the intended milestone branch.
- [ ] `git status` is clean after the milestone commit.
- [ ] The milestone commit has been pushed to the remote repository.
- [ ] Raw 3DRIED data are not committed.
- [ ] Large generated binaries such as `volume.npy`, checkpoints, caches, and temporary data are not accidentally committed unless explicitly intended.
- [ ] Source code, configs, tests, Markdown reports, logs/metrics needed for review, and lightweight figures are available in Git.
- [ ] `STATUS.md` reflects the actual current state.
- [ ] `EXPERIMENTS.md` contains the executed commands and outputs.
- [ ] `DECISIONS.md` records every nontrivial scientific assumption introduced during the milestone.
- [ ] No locked milestone has been started.
- [ ] No previously fixed scientific decision was silently changed.
- [ ] No raw dataset file was modified.
- [ ] The selected single-Rx policy remains unchanged.

# 3. M3 Review Gate — Forward / Adjoint Operator

## 3.1 Scope

M3 establishes the validated physics operator used by all later reconstruction algorithms.

Expected core interface:

```python
operator.forward(x, ...)
operator.adjoint(y, ...)
```

Suggested implementation location:

```text
src/imaging/holography.py
```

## 3.2 Physics/model consistency

- [ ] The implementation matches the forward model frozen in `RESEARCH_SPEC.md`, or any necessary deviation is explicitly documented and approved.
- [ ] Coordinate definitions for x, y, z match the validated M2 RMA implementation.
- [ ] Range coordinate convention is explicit.
- [ ] The phase sign in forward propagation is explicit and justified.
- [ ] The phase sign in the adjoint is the complex-conjugate counterpart of the forward model.
- [ ] `kx`, `kz`, and `ky` grids are generated from physical sampling intervals and radar parameters rather than arbitrary indices.
- [ ] Invalid/evanescent wavenumber regions are handled explicitly.
- [ ] Zero-padding/cropping conventions are explicit.
- [ ] FFT shift conventions are explicit and consistent.
- [ ] FFT normalization is explicit and consistent.
- [ ] Aperture support/mask treatment is consistent between forward and adjoint.
- [ ] No Hammer-specific hard-coded correction is hidden inside the generic operator.
- [ ] No learned parameter is introduced into the physics operator.

## 3.3 Adjoint test

Use the complex inner product with correct conjugation.

For random complex `x` and `y`, test:

\[
e_{\rm adj}
=
\frac{
|\langle Ax,y\rangle-\langle x,A^Hy\rangle|
}{
\max(|\langle Ax,y\rangle|,10^{-12})
}.
\]

Required:

\[
e_{\rm adj}<10^{-5}.
\]

Review items:

- [ ] The test uses a correct complex inner product, e.g. `torch.vdot` or an equivalent conjugating formulation.
- [ ] The test is not accidentally comparing magnitudes only.
- [ ] The test is run for more than one random seed.
- [ ] At least one high-precision (`complex128`) reference test is reported.
- [ ] The reported error is reproducible.
- [ ] The test fails when an intentional phase/sign error is introduced, or there is otherwise convincing evidence that the test is sensitive to operator errors.

## 3.4 RMA equivalence / consistency

The adjoint should be consistent with the validated M2 conventional RMA/holographic reconstruction.

Review items:

- [ ] The comparison uses the same preprocessed Hammer data as M2.
- [ ] Coordinate grids match.
- [ ] Range selection matches.
- [ ] Any global scale or global complex phase difference is identified rather than silently hidden.
- [ ] A quantitative comparison is reported where meaningful.
- [ ] Magnitude images / MIPs are visually consistent.
- [ ] Any residual difference is explained.

Suggested evidence:

```text
docs/M3_OPERATOR_REPORT.md
results/m3_operator/
```

## 3.5 Software tests

- [ ] Forward output shape is correct.
- [ ] Adjoint output shape is correct.
- [ ] Complex dtype is preserved.
- [ ] CPU execution works for a small test.
- [ ] GPU execution works if CUDA is available.
- [ ] No NaN/Inf for valid inputs.
- [ ] Tests cover at least one nontrivial range slice.
- [ ] Tests are runnable from a documented command.

## 3.6 M3 acceptance

M3 external verdict:

```text
[ ] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

Blocking issues:

```text
...
```

Non-blocking issues:

```text
...
```

Reviewer evidence:

```text
...
```

M4 may proceed only if M3 self-tests PASS. Merge/unlock decisions remain subject to external review.

# 4. M4 Review Gate — Kernel-ISTA

## 4.1 Scope

M4 implements a traditional, non-learned Kernel-ISTA baseline using the validated M3 operator.

Target optimization:

\[
\hat x
=
\arg\min_x
\frac12\|y-Ax\|_2^2+\lambda\|x\|_1.
\]

Iteration:

\[
x_{t+1}
=
\mathcal S_{\lambda\tau}
\left[
x_t+\tau A^H(y-Ax_t)
\right].
\]

Complex soft-threshold:

\[
\mathcal S_{\theta}(z)
=
\frac{z}{|z|+\epsilon}
\max(|z|-\theta,0).
\]

## 4.2 Algorithm correctness

- [ ] The residual is `y - A(x)`.
- [ ] The gradient correction uses `A^H(residual)`.
- [ ] The update sign is correct.
- [ ] Soft-thresholding is applied to complex magnitude while preserving phase.
- [ ] The threshold is nonnegative.
- [ ] The initial estimate is documented, preferably `A^H y`.
- [ ] The step size is scientifically justified.
- [ ] If an operator norm/Lipschitz estimate is used, the estimation method is documented.
- [ ] No trainable parameters are introduced in M4.
- [ ] Ground truth is not used inside the reconstruction algorithm.

## 4.3 Synthetic-data correctness

- [ ] Synthetic ground truth is generated independently from the reconstruction.
- [ ] The same validated M3 forward operator generates the synthetic echo.
- [ ] Random seed is fixed and recorded.
- [ ] Complex scatterer phase is represented correctly.
- [ ] Noise-free recovery is tested before noisy recovery.
- [ ] If noise is added, SNR definition and implementation are documented.
- [ ] Train/validation concepts are not required in M4 because M4 is not a learning milestone.

## 4.4 Convergence evidence

Required evidence should include:

- [ ] initial adjoint reconstruction;
- [ ] final Kernel-ISTA reconstruction;
- [ ] ground-truth synthetic scene;
- [ ] objective or data-fidelity curve;
- [ ] NMSE curve or final NMSE;
- [ ] runtime;
- [ ] iteration count.

The final solution must improve materially over the plain adjoint initialization on the synthetic test.

Do not claim convergence solely because the image looks cleaner.

## 4.5 Numerical/stability checks

- [ ] No NaN/Inf.
- [ ] Results are reproducible with fixed seed.
- [ ] A small change in initialization does not cause unexplained catastrophic divergence.
- [ ] Step size is not selected by using test ground truth in an undocumented way.
- [ ] Parameter tuning, if performed, is logged in `EXPERIMENTS.md`.

## 4.6 M4 report

Required:

```text
docs/M4_KERNEL_ISTA_REPORT.md
```

Suggested result directory:

```text
results/m4_kernel_ista/
```

Report must contain:

1. exact objective;
2. exact iteration equation;
3. initialization;
4. step-size selection;
5. threshold / lambda selection;
6. synthetic scene generation;
7. quantitative results;
8. convergence curves;
9. runtime;
10. known limitations.

## 4.7 M4 acceptance

M4 external verdict:

```text
[ ] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

Blocking issues:

```text
...
```

Non-blocking issues:

```text
...
```

Reviewer evidence:

```text
...
```

Do not begin M5 until the M3-M4 review is completed and the human explicitly unlocks M5.

# 5. Independent review protocol

When M3-M4 are complete:

1. Push the milestone branch to GitHub.
2. Do not merge it into `main` yet.
3. Provide the branch name and commit hashes.
4. Ask ChatGPT to independently inspect:
   - control files;
   - M3/M4 reports;
   - operator implementation;
   - tests;
   - Kernel-ISTA implementation;
   - experiment evidence.
5. Resolve blocking review comments with additional commits on the same branch.
6. Re-review if needed.
7. Merge only after human approval.

# 6. Reviewer output template

```text
Review scope:
Branch:
M3 commit:
M4 commit:

M3 verdict:
Blocking issues:
Non-blocking issues:
Evidence checked:

M4 verdict:
Blocking issues:
Non-blocking issues:
Evidence checked:

Repository hygiene:
Scientific assumptions changed?:
Recommended fixes:
Recommended next action:
```
