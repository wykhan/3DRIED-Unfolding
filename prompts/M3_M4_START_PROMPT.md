# M3_M4_START_PROMPT.md

# Long-Running Codex Task: M3 + M4

Execute the next authorized milestones of the `3DRIED-Unfolding` repository autonomously.

The human has already reviewed M0, M1, and M2 and confirmed that they are correct.

Your scope for this run is strictly:

- M3 — Forward / Adjoint Operator Validation
- M4 — Kernel-ISTA

Do not begin M5 or any later milestone.

## 1. Read the repository control files first

Before editing code, read completely:

- `AGENTS.md`
- `RESEARCH_SPEC.md`
- `PLAN.md`
- `STATUS.md`
- `DECISIONS.md`
- `EXPERIMENTS.md`
- `REVIEW_CHECKLIST.md`

Also read:

- `docs/HAMMER_DATA_AUDIT.md`
- `docs/M2_RMA_HAMMER_REPORT.md`

and inspect the actual M2 RMA/preprocessing implementation before defining the M3 operator.

Treat these files and the validated M2 implementation as the source of truth.

Do not silently replace the coordinate convention, preprocessing convention, radar parameters, or single-Rx decision established in M0-M2.

## 2. Git workflow

Before implementation:

1. Confirm the working tree is clean.
2. Confirm `main` contains the human-approved M0-M2 state.
3. Create or switch to:

```text
milestone/m3-m4
```

from the approved `main`.

All M3-M4 work must remain on this branch.

After M3 passes:

1. update project records;
2. commit M3;
3. push the branch to the remote repository.

Then continue to M4 only if all M3 self-review gates pass.

After M4 passes:

1. update project records;
2. commit M4;
3. push the branch again.

Do not merge the branch into `main`.

If GitHub CLI is already authenticated, you may create or update a pull request from `milestone/m3-m4` into `main`. Failure to create a PR is not a reason to lose or redo local work; the pushed branch and commit hashes are sufficient.

Do not commit raw 3DRIED data or unintended large generated binaries.

# 3. M3 — Forward / Adjoint Operator Validation

## 3.1 Goal

Implement a reusable physics-based near-field holographic forward/adjoint operator consistent with the validated M2 3DRIED Hammer reconstruction.

Suggested file:

```text
src/imaging/holography.py
```

Suggested interface:

```python
operator.forward(x, ...)
operator.adjoint(y, ...)
```

The implementation must not be Hammer-specific.

## 3.2 Physics model

Use the physical model currently frozen in `RESEARCH_SPEC.md`, but first reconcile it carefully with the validated M2 RMA implementation.

Do not assume that the sign, FFT shift, range indexing, or zero-padding conventions in the abstract specification exactly match the final M2 code.

The validated M2 implementation determines the actual coordinate/phase convention.

If an implementation-level adjustment is required to make the generic operator mathematically consistent with M2:

1. document the issue;
2. record the decision in `DECISIONS.md`;
3. update `RESEARCH_SPEC.md` only if necessary;
4. do not change the physical meaning of the model.

Do not introduce learned parameters.

## 3.3 Required M3 tests

### A. Shape/dtype tests

Verify:

- forward input/output shapes;
- adjoint input/output shapes;
- complex dtype preservation;
- CPU smoke execution;
- GPU smoke execution if CUDA is available.

### B. Complex adjoint identity

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

Use the mathematically correct complex inner product with conjugation.

Run the test for multiple random seeds.

At least one reference run must use high precision (`complex128`).

Required:

\[
e_{\rm adj}<10^{-5}.
\]

If the test fails, diagnose and repair the operator before continuing.

Do not weaken the threshold to make the test pass.

### C. Sensitivity sanity check

Verify that the adjoint test is meaningful.

For example, temporarily introducing an intentional sign/conjugation error should substantially worsen the adjoint error, or provide another convincing test showing the validation detects operator mistakes.

Do not leave intentionally broken code in the repository.

### D. M2 RMA consistency

Using the same processed Hammer data and geometry as M2, compare:

```text
M2 RMA result
```

with:

```text
M3 operator.adjoint(...)
```

Check:

- physical coordinates;
- target orientation;
- target range;
- magnitude structure;
- normalization;
- possible global scale / global complex phase differences.

Do not hide discrepancies by arbitrary image manipulation.

If a global scalar/phase alignment is needed only for comparison, compute and report it explicitly.

## 3.4 M3 outputs

Create:

```text
docs/M3_OPERATOR_REPORT.md
```

and a suitable result directory such as:

```text
results/m3_operator/
```

The report must contain:

1. final forward equation;
2. final adjoint equation;
3. coordinate conventions;
4. FFT normalization/shift conventions;
5. mask/padding conventions;
6. adjoint test implementation;
7. adjoint errors for all reported seeds/dtypes;
8. M2 RMA consistency evidence;
9. unresolved uncertainties;
10. exact commands required to reproduce the tests.

Update:

- `STATUS.md`
- `EXPERIMENTS.md`
- `DECISIONS.md` if needed.

Run the M3 section of `REVIEW_CHECKLIST.md` as a self-review.

Only mark M3 PASS if every blocking item passes.

Then:

```text
git commit
git push
```

Record the M3 commit hash in `STATUS.md`.

If M3 cannot pass after reasonable debugging, STOP and request human review.

Do not continue to M4 with a questionable operator.

# 4. M4 — Kernel-ISTA

Proceed only if M3 self-review passes.

## 4.1 Goal

Implement a conventional, non-learned Kernel-ISTA baseline using the validated M3 operator.

Suggested file:

```text
src/solvers/kernel_ista.py
```

Target problem:

\[
\hat x
=
\arg\min_x
\frac12\|y-Ax\|_2^2+\lambda\|x\|_1.
\]

Target iteration:

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

Do not introduce trainable parameters.

## 4.2 Step size

Choose the ISTA step size from the operator's Lipschitz constant/operator norm when practical.

Preferred approach:

- analytically determine the operator norm if the validated operator structure makes this straightforward; otherwise
- estimate \(\|A\|_2^2\) using a reproducible power iteration.

Then choose a stable step size consistent with ISTA theory.

Do not choose the step size by repeatedly looking at ground truth until the plot appears good without recording the tuning process.

Record the exact method and value.

## 4.3 Synthetic validation before real data

Do not use Hammer as the first M4 validation.

Create a minimal synthetic complex sparse scene:

- one range slice first;
- random point target positions;
- random complex phases;
- reproducible random seed.

Use the validated M3 forward operator to generate the measurement:

\[
y=A(x_{\rm gt}).
\]

First run a noise-free test.

Only after the noise-free path works, optionally add a documented 20 dB complex AWGN test.

Ground truth must never enter the reconstruction update.

## 4.4 Required comparisons

For the same synthetic measurement, save and compare:

1. ground truth;
2. adjoint initialization \(A^Hy\);
3. final Kernel-ISTA reconstruction.

Required numerical evidence:

- initial NMSE;
- final NMSE;
- objective or data-fidelity curve;
- runtime;
- iteration count.

Kernel-ISTA must materially improve reconstruction over the plain adjoint initialization.

If it does not, debug:

- operator use;
- step size;
- threshold/lambda;
- normalization;
- soft-threshold implementation.

Do not proceed by simply declaring the synthetic scene "too difficult".

## 4.5 M4 outputs

Suggested files:

```text
src/solvers/kernel_ista.py
src/simulation/synthetic.py
scripts/run_ista.py
configs/kernel_ista_baseline.yaml
```

Create:

```text
docs/M4_KERNEL_ISTA_REPORT.md
results/m4_kernel_ista/
```

The report must include:

1. optimization objective;
2. exact implemented iteration;
3. initialization;
4. operator used;
5. step-size determination;
6. lambda/threshold choice;
7. synthetic scene generation;
8. noise model if used;
9. quantitative results;
10. convergence curves;
11. runtime;
12. known limitations;
13. reproduction commands.

Update:

- `STATUS.md`
- `EXPERIMENTS.md`
- `DECISIONS.md` if needed.

Run the M4 section of `REVIEW_CHECKLIST.md` as a self-review.

Only mark M4 PASS if all blocking items pass.

Then:

```text
git commit
git push
```

Record the M4 commit hash in `STATUS.md`.

# 5. Stop after M4

After M4 completes:

1. ensure all M3-M4 work is pushed to the remote branch;
2. ensure the working tree is clean;
3. do not merge to `main`;
4. do not start M5;
5. provide a concise milestone summary.

The final summary must include:

```text
Branch:
Remote push status:

M3:
PASS / FAIL
commit:
key tests:
adjoint error:
M2 consistency:
report path:

M4:
PASS / FAIL
commit:
initial NMSE:
final NMSE:
iterations:
runtime:
report path:

Changed scientific assumptions:
Unresolved issues:
Recommended next action:
```

Then STOP for independent ChatGPT review and human approval.

# 6. Autonomous execution policy

Routine implementation failures are not reasons to stop.

For:

- Python errors;
- import errors;
- tensor shapes;
- dtype/device issues;
- CUDA OOM;
- plotting errors;
- test harness bugs;
- file path mistakes;
- ordinary numerical implementation mistakes;

diagnose, repair, rerun, and continue.

Stop only for scientific ambiguity, failure of the validated physics model, explicit repository stop conditions, or inability to satisfy M3/M4 acceptance criteria after reasonable attempts.

Do not start M5.
