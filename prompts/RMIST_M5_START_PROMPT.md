# RMIST_M5_START_PROMPT.md

# Long-running Codex task: M5-R strict RMIST-Net architecture

The human explicitly authorizes M5-R architecture implementation.

This task is limited to RMIST-Net Eq. (20)-(26).

Do NOT start M6-R training.
Do NOT generate the full 1000-pair training dataset.
Do NOT run Adam.
Do NOT run 100 epochs.
Do NOT perform real Hammer inference.
Do NOT unlock M6-R or M7-R.

The accepted `main` branch already contains externally reviewed M3-R and M4-R.

## 1. Verify M5 branch

The human has already created `milestone/rmist-m5` from the accepted `main`
and committed the M5 specification/prompt files before launching this task.

Run:

```bash
git status
git branch --show-current
git fetch origin
git log --oneline --decorate -10
git merge-base --is-ancestor origin/main HEAD
```

Confirm:

```text
current branch = milestone/rmist-m5
```

and confirm the branch descends from the accepted `main` containing M3-R/M4-R.

If the current branch is not `milestone/rmist-m5`, if ancestry is unexpected,
or if unrelated unexplained changes are present, STOP and report.

Do not create, delete, rename, reset, or force-push branches.

## 2. Read source-of-truth files

Read completely:

- `AGENTS.md`
- `RESEARCH_SPEC.md`
- `PLAN.md`
- `STATUS.md`
- `DECISIONS.md`
- `docs/RMIST_REPRO_SPEC.md`
- `docs/M5_RMIST_NET_SPEC.md`
- `docs/M5_REVIEW_CHECKLIST.md`
- `docs/M3_RMIST_KERNEL_REPORT.md`
- `docs/M4_RMIST_ISTA_REPORT.md`

Inspect:

- `src/three_dried/imaging/rmist_kernel.py`
- `src/three_dried/solvers/rmist_ista.py`
- relevant tests.

The RMIST paper remains algorithmic authority.

## 3. Unlock only M5-R

Before implementation, update active governance.

### AGENTS.md

Set:

```text
M5-R: UNLOCKED
M6-R: LOCKED
M7-R: LOCKED
```

State M5-R is architecture validation only.

### PLAN.md

Set:

```text
M3-R PASS
M4-R PASS
M5-R UNLOCKED
M6-R LOCKED
M7-R LOCKED
```

Add M5-R acceptance criteria from `docs/M5_RMIST_NET_SPEC.md`.

### STATUS.md

Set current task:

```text
Implement and validate strict 9-phase RMIST-Net Eq. (20)-(26).
Stop after M5-R for independent review.
```

### REVIEW_CHECKLIST.md

Keep accepted M3-R/M4-R history and reference:

```text
docs/M5_REVIEW_CHECKLIST.md
```

Commit governance unlock before scientific code:

```bash
git add AGENTS.md PLAN.md STATUS.md REVIEW_CHECKLIST.md docs/M5_RMIST_NET_SPEC.md docs/M5_REVIEW_CHECKLIST.md prompts/RMIST_M5_START_PROMPT.md
git commit -m "chore: unlock strict RMIST M5-R architecture"
git push
```

## 4. Implement RMIST-Net Eq. (20)-(26)

Suggested file:

```text
src/three_dried/models/rmist_net.py
```

Implement exactly:

\[
v_t=S-RM(\alpha_t)
\]

\[
r_t=\alpha_t+\tau_t RM^\dagger(v_t)
\]

\[
\alpha_{t+1}=F_{st}(r_t,\theta_t).
\]

Initialization:

\[
\alpha_0=RM^\dagger(S).
\]

Strict baseline:

```text
phases = 9
tau_t init = 1e-4
theta_t init = 1e-3
```

Only phase-wise `tau_t` and `theta_t` may be trainable.

Recommended:

```python
self.tau = nn.Parameter(torch.full((9,), 1e-4))
self.theta = nn.Parameter(torch.full((9,), 1e-3))
```

Do not add softplus, clipping, or another reparameterization.

Do not add CNNs, biases, FISTA momentum, Gamma-Net, Toe-LISTA, attention, learned masks, learned measurement matrices, or trainable RM kernels.

## 5. Complex representation

The paper describes separate real/imaginary paths.

PyTorch complex tensors are allowed internally only as a mathematically equivalent implementation choice.

Implement/test:

```python
complex_to_two_channel(...)
two_channel_to_complex(...)
```

and document representation choice.

## 6. Required tests

### A. Parameter structure

Verify:

```text
tau shape = [9]
theta shape = [9]
total trainable scalar count = 18
```

No hidden trainable parameters.

### B. Initialization

Verify:

```text
alpha0 == RM_dagger(S)
```

within numerical tolerance.

### C. Literal one-phase test

Create a deterministic one-phase reference calculation and verify the network phase equals Eq. (20).

### D. Blocking M4 -> M5 regression

With:

```text
all tau_t = 1e-4
all theta_t = 1e-3
phases = 9
```

compare `RMISTNet` against:

```python
rmist_ista(
    ...,
    tau=1e-4,
    theta=1e-3,
    iterations=9,
)
```

Required:

```text
complex64 relative difference < 1e-5
complex128 relative difference < 1e-12
```

### E. Gradient flow

Use deterministic synthetic input and target.

Compute a simple differentiable reconstruction loss and call:

```python
loss.backward()
```

Verify:

- `tau.grad` exists;
- `theta.grad` exists;
- all gradient entries finite;
- every phase receives meaningful gradient in the deterministic test.

Do not create an optimizer or take optimizer steps.

### F. Representation round trip

Verify:

```text
complex → two-channel real/imag → complex
```

preserves values.

### G. Shape/device

Test:

- CPU complex64;
- CPU complex128;
- CUDA complex64 if available;
- single sample;
- batch input if existing M3-R semantics support batching without physics changes.

If batching would require changing M3-R physics semantics, STOP rather than redesigning M3-R.

### H. Phase trace

Provide an optional diagnostic trace of exactly 9 entries:

```text
phase
tau_t
theta_t
residual_norm
estimate_norm
```

Trace must not affect output.

## 7. No training in M5-R

Do not create:

```text
optimizer
epoch loop
1000-pair dataset
checkpoint-selection logic
training/validation loop
```

A backward-only gradient test is required and is not training.

## 8. Lightweight review artifacts

Create:

```text
docs/M5_RMIST_NET_REPORT.md
results/review/M5/
```

Commit lightweight evidence:

```text
results/review/M5/summary_metrics.json
results/review/M5/architecture.txt
results/review/M5/parameter_values.json
results/review/M5/phase_trace.json
```

If `.gitignore` blocks `results/review/`, narrowly allow this directory only.

Do not commit large `.npy`, `.pt`, `.pth`, or checkpoint files.

## 9. Report requirements

The M5 report must state:

1. Eq. (20)-(26) to code mapping;
2. 9-phase architecture;
3. exact trainable parameters;
4. initialization values;
5. complex representation choice;
6. initialization regression result;
7. M4→M5 regression result in complex64 and complex128;
8. gradient-flow result;
9. batch/device results;
10. deviations/implementation choices;
11. explicit confirmation that M6 training was not started.

## 10. Validate

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

M5-R PASS requires every blocking item in `docs/M5_REVIEW_CHECKLIST.md`.

Update:

- `STATUS.md`
- `EXPERIMENTS.md`
- `DECISIONS.md` only when needed.

Commit:

```bash
git add ...
git commit -m "feat: implement strict 9-phase RMIST-Net architecture"
git push origin milestone/rmist-m5
```

Do not merge to main.

## 11. Stop

Return:

```text
Branch:

M5-R:
PASS / FAIL
commit:

phases:
trainable tensors:
trainable scalar count:

tau initialization:
theta initialization:

alpha0 regression:
M4→M5 complex64 relative difference:
M4→M5 complex128 relative difference:

gradient tau finite/nonzero:
gradient theta finite/nonzero:

CPU tests:
CUDA tests:
batch test:

report:
review artifacts:

RMIST deviations / implementation choices:
unresolved issues:

M6-R status: LOCKED
```

Then STOP for independent ChatGPT review.

Do not start training.
