# M5_REVIEW_CHECKLIST.md

# M5-R Strict RMIST-Net Architecture Review

## Repository gate

- [ ] Branch is `milestone/rmist-m5`.
- [ ] Branch starts from accepted `main`.
- [ ] M3-R/M4-R accepted code remains unchanged except explicitly documented regression-safe refactors.
- [ ] M6-R and M7-R remain locked.
- [ ] No optimizer/training loop has started.
- [ ] No real Hammer inference has started.

## Architecture fidelity

- [ ] `RMISTNet` implements RMIST Eq. (20).
- [ ] Exactly 9 phases execute.
- [ ] Initialization is exactly `RM_dagger(S)`.
- [ ] Stage 1 is `S - RM(alpha_t)`.
- [ ] Stage 2 is `alpha_t + tau_t * RM_dagger(v_t)`.
- [ ] Stage 3 is RMIST complex magnitude shrinkage.
- [ ] `tau_t` and `theta_t` are phase-specific.
- [ ] No parameter sharing.
- [ ] No FISTA momentum.
- [ ] No CNN proximal.
- [ ] No Gamma-Net/Toe-LISTA logic.
- [ ] No learned forward/RM kernel.

## Parameter gate

- [ ] `tau.shape == [9]`.
- [ ] `theta.shape == [9]`.
- [ ] Initial `tau_t == 1e-4`.
- [ ] Initial `theta_t == 1e-3`.
- [ ] Exactly 18 trainable scalars.
- [ ] No hidden trainable parameters.
- [ ] No undocumented positivity reparameterization.

## Complex representation

- [ ] Complex phase is preserved.
- [ ] PyTorch-complex internal representation, if used, is documented as an implementation choice.
- [ ] real/imag two-channel conversion helpers are tested.
- [ ] conversion round trip preserves values.

## Blocking M4 -> M5 regression

With all phases set to:

```text
tau_t = 1e-4
theta_t = 1e-3
```

- [ ] M5 output matches M4 `rmist_ista(..., iterations=9)`.
- [ ] complex64 relative difference < `1e-5`.
- [ ] complex128 relative difference < `1e-12`.

This is a blocking gate.

## Initialization regression

- [ ] `alpha0` matches accepted M3-R `RM_dagger(S)`.
- [ ] No alternative initialization is used.

## Gradient-flow gate

After deterministic loss and `backward()`:

- [ ] `tau.grad` exists.
- [ ] `theta.grad` exists.
- [ ] all gradients are finite.
- [ ] gradient flow reaches every phase in the deterministic test.
- [ ] no optimizer step is taken.

## Shape/device gate

- [ ] single-sample input works.
- [ ] batch input works if possible without changing physics semantics.
- [ ] CPU complex64 passes.
- [ ] CPU complex128 passes.
- [ ] CUDA complex64 passes when available.

## Trace/reproducibility

- [ ] exactly 9 phase trace entries.
- [ ] fixed seed used for stochastic tests.
- [ ] lightweight metrics are committed under `results/review/M5/`.
- [ ] no large checkpoints/tensors are committed.

## Verdict

```text
[ ] PASS
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
