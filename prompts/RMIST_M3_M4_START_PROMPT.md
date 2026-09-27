# RMIST_M3_M4_START_PROMPT.md

# Long-running Codex task: strict RMIST M3-R + M4-R

Execute a method-level reproduction of RMIST-Net for the 3DRIED repository.

This run is strictly limited to:

- M3-R — RM / RM_dagger kernel implementation and validation
- M4-R — conventional RM-kernel ISTA reproducing RMIST Eq. (16)-(19)

Do not begin M5-R or RMIST-Net training.

## 1. Source-of-truth order

Before editing code, read completely:

1. `docs/RMIST_REPRO_SPEC.md`
2. `AGENTS.md`
3. `RESEARCH_SPEC.md`
4. `PLAN.md`
5. `STATUS.md`
6. `DECISIONS.md`
7. `EXPERIMENTS.md`
8. `REVIEW_CHECKLIST.md`
9. `docs/M2_RMA_HAMMER_REPORT.md`
10. the actual M2 RMA implementation

The RMIST paper is authoritative for the algorithm. `docs/RMIST_REPRO_SPEC.md` defines the 3DRIED mapping.

Do not replace RMIST with a mathematically cleaner forward/Hilbert-adjoint pair.

## 2. Git safety

Confirm:

```text
current branch = milestone/rmist-m3-m4
```

Confirm it descends from human-approved `main`.

Do not merge or use the old `milestone/m3-m4` physics definition as authority. Generic utilities may be inspected, but algorithm semantics must come from RMIST.

## 3. M3-R — strict RMIST kernels

Suggested file:

```text
src/three_dried/imaging/rmist_kernel.py
```

RMIST canonical pair:

\[
\Phi_r=k_y^{-1}e^{-j k_y r}
\]

\[
\Phi_r^\dagger=k_y e^{+j k_y r}.
\]

3DRIED validated sign mapping:

\[
\Phi_{3D}^\dagger=k_y e^{-j k_y r}
\]

\[
\Phi_{3D}=k_y^{-1}e^{+j k_y r}.
\]

The reciprocal `1/ky` / `ky` structure is mandatory.

Do not call `RM_dagger` a Hilbert adjoint.

### Required tests

1. exact kernel-formula test;
2. reciprocal-kernel product on valid support;
3. explicit non-propagating and zero/near-zero `ky` handling;
4. physical `[200,407]` to `[512,512]` padding test;
5. synthetic full-domain `RM → RM_dagger` round trip;
6. Hammer `RM_dagger` equivalence to human-approved M2 RMA;
7. complex64 and complex128 tests;
8. CUDA smoke test if available.

Target M2 complex relative difference:

```text
< 1e-5
```

Do not use Hilbert-adjoint identity as a PASS criterion.

If strict `1/ky` / `ky` reproduction cannot be reconciled with M2, STOP. Do not silently redesign the operator.

### M3-R deliverables

```text
docs/M3_RMIST_KERNEL_REPORT.md
results/m3_rmist_kernel/
```

Report:

- Eq. (13)-(15) mapping to code;
- 3DRIED sign mapping;
- padding;
- ky support handling;
- reciprocal error;
- round-trip error;
- M2 consistency;
- exact reproduction commands;
- deviations from paper.

Update `STATUS.md`, `EXPERIMENTS.md`, and `DECISIONS.md`, then commit and push M3-R.

## 4. M4-R — strict RM-kernel ISTA

Proceed only after M3-R self-review passes.

Implement RMIST Eq. (16)-(19):

\[
v_t=S-RM(\alpha_t)
\]

\[
r_t=\alpha_t+\tau RM^\dagger(v_t)
\]

\[
\alpha_{t+1}=F_{st}(r_t,\theta).
\]

Suggested files:

```text
src/three_dried/solvers/rmist_ista.py
src/three_dried/simulation/rmist_synthetic.py
scripts/run_rmist_ista.py
configs/rmist_ista.yaml
```

Do not merely rename the discarded generic operator.

### Complex shrinkage

\[
F_{st}(r,\theta)
=
\frac{r}{|r|+\epsilon}
ReLU(|r|-\theta).
\]

Unit-test magnitude shrinkage and phase preservation.

### Ini-ISTA mode

Provide:

```text
tau = 1e-4
theta = 1e-3
iterations = 9
```

State precisely that `tau` and `theta` are RMIST's reported initialization values; the paper does not provide exact final Tuned-ISTA values.

### Tuned diagnostic mode

If needed, a separate 3DRIED tuned conventional ISTA is allowed. Record every value and tuning method. Never label tuned values as RMIST-reported.

Do not reuse the previous Hilbert-adjoint Lipschitz argument unless independently valid for the strict RM/RM_dagger algorithm.

### Synthetic validation

First:

- complex point scatterers;
- random phase;
- `Mask=1`;
- strict `RM` echo generation;
- noise-free smoke test.

Then separately:

- 20 dB complex AWGN.

Keep small smoke tests separate from the later 512×512 paper-style benchmark.

### M4-R deliverables

```text
docs/M4_RMIST_ISTA_REPORT.md
results/m4_rmist_ista/
```

Report Eq. (16)-(19) mapping, parameters, Ini-ISTA result, optional tuned result, noise-free/20 dB results, curves, runtime, commands, and deviations.

Commit and push M4-R.

## 5. Stop after M4-R

After M4-R:

1. push all commits to `milestone/rmist-m3-m4`;
2. ensure working tree is clean;
3. do not merge;
4. do not begin M5-R;
5. STOP for independent ChatGPT review.

Final summary:

```text
Branch:

M3-R:
PASS/FAIL:
commit:
kernel formula test:
reciprocal test:
M2 consistency:
synthetic round trip:

M4-R:
PASS/FAIL:
commit:
Ini-ISTA parameters/result:
tuned diagnostic parameters/result:
20 dB result:
runtime:

RMIST deviations:
unresolved issues:
```

Routine coding/CUDA/path/plotting failures should be fixed autonomously. Scientific ambiguity is a stop condition.
