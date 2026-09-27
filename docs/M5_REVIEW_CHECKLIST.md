# M5_REVIEW_CHECKLIST.md

# M5-R Strict RMIST-Net Architecture Review

## Repository gate

- [x] Branch is `milestone/rmist-m5`.
- [x] Branch starts from accepted `main`.
- [x] M3-R/M4-R accepted code remains unchanged except explicitly documented regression-safe refactors.
- [x] M6-R and M7-R remain locked.
- [x] No optimizer/training loop has started.
- [x] No real Hammer inference has started.

## Architecture fidelity

- [x] `RMISTNet` implements RMIST Eq. (20).
- [x] Exactly 9 phases execute.
- [x] Initialization is exactly `RM_dagger(S)`.
- [x] Stage 1 is `S - RM(alpha_t)`.
- [x] Stage 2 is `alpha_t + tau_t * RM_dagger(v_t)`.
- [x] Stage 3 is RMIST complex magnitude shrinkage.
- [x] `tau_t` and `theta_t` are phase-specific.
- [x] No parameter sharing.
- [x] No FISTA momentum.
- [x] No CNN proximal.
- [x] No Gamma-Net/Toe-LISTA logic.
- [x] No learned forward/RM kernel.

## Parameter gate

- [x] `tau.shape == [9]`.
- [x] `theta.shape == [9]`.
- [x] Initial `tau_t == 1e-4`.
- [x] Initial `theta_t == 1e-3`.
- [x] Exactly 18 trainable scalars.
- [x] No hidden trainable parameters.
- [x] No undocumented positivity reparameterization.

## Complex representation

- [x] Complex phase is preserved.
- [x] PyTorch-complex internal representation, if used, is documented as an implementation choice.
- [x] real/imag two-channel conversion helpers are tested.
- [x] conversion round trip preserves values.

## Blocking M4 -> M5 regression

With all phases set to:

```text
tau_t = 1e-4
theta_t = 1e-3
```

- [x] M5 output matches M4 `rmist_ista(..., iterations=9)`.
- [x] complex64 relative difference < `1e-5`.
- [x] complex128 relative difference < `1e-12`.

This is a blocking gate.

## Initialization regression

- [x] `alpha0` matches accepted M3-R `RM_dagger(S)`.
- [x] No alternative initialization is used.

## Gradient-flow gate

After deterministic loss and `backward()`:

- [x] `tau.grad` exists.
- [x] `theta.grad` exists.
- [x] all gradients are finite.
- [x] gradient flow reaches every phase in the deterministic test.
- [x] no optimizer step is taken.

## Shape/device gate

- [x] single-sample input works.
- [x] batch input works if possible without changing physics semantics.
- [x] CPU complex64 passes.
- [x] CPU complex128 passes.
- [x] CUDA complex64 passes when available.

## Trace/reproducibility

- [x] exactly 9 phase trace entries.
- [x] fixed seed used for stochastic tests.
- [x] lightweight metrics are committed under `results/review/M5/`.
- [x] no large checkpoints/tensors are committed.

## Verdict

```text
[x] PASS
[ ] PASS WITH NON-BLOCKING ISSUES
[ ] NEEDS FIX
[ ] FAIL
```

## M6-R unlock gate

M6-R remains locked until:

- [ ] M5-R independent review is complete.
- [ ] no blocking discrepancy with RMIST Eq. (20)-(26).
- [ ] accepted M5-R is merged to `main`.
- [ ] human explicitly unlocks M6-R.
