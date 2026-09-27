# M5_RMIST_NET_SPEC.md

# M5-R Strict RMIST-Net Architecture Specification

## Scope

M5-R implements RMIST-Net Eq. (20)-(26) only. It is architecture validation, not training.

M5-R must not create the 1000-pair training set, run Adam, run 100 epochs, perform real Hammer inference, or unlock M6-R/M7-R.

The accepted M3-R `RM` / `RM_dagger` implementation and M4-R Eq. (19) solver are fixed.

## Paper mapping

For each phase \(t\):

\[
v_t=S-RM(\alpha_t)
\]

\[
r_t=\alpha_t+\tau_t RM^\dagger(v_t)
\]

\[
\alpha_{t+1}=F_{st}(r_t,\theta_t)
\]

Use exactly 9 phases.

Each phase has its own learnable real scalars:

\[
\tau_t,\theta_t.
\]

Initialize:

```text
tau_t = 1e-4
theta_t = 1e-3
```

No parameter sharing across phases.

## Initialization

With padded measurement

\[
S=P(S_r\odot Mask),
\]

initialize:

\[
\alpha_0=RM^\dagger(S).
\]

For M5 validation the model may accept already-padded complex `S`.

## Stage 3 complex shrinkage

\[
F_{st}(r,\theta)
=
\frac{r}{|r|+\epsilon}\operatorname{ReLU}(|r|-\theta).
\]

Real and imaginary components share the same magnitude shrinkage factor.

## Trainable parameters

Strict M5-R has only:

```text
tau[0:9]
theta[0:9]
```

Exactly 18 trainable scalar parameters.

No trainable convolution, bias, RM kernel, normalization, attention, support selection, CNN proximal block, mask, or measurement matrix.

Recommended:

```python
self.tau = nn.Parameter(torch.full((9,), 1e-4))
self.theta = nn.Parameter(torch.full((9,), 1e-3))
```

Do not introduce softplus, exponentiation, clipping, or other positivity reparameterization in the strict baseline.

## Complex representation

The paper describes separate real/imaginary paths. PyTorch complex tensors may be used internally only as a mathematically equivalent, paper-unspecified implementation choice.

Provide and test:

```python
complex_to_two_channel(...)
two_channel_to_complex(...)
```

conceptually mapping to `[B, 2, H, W]`.

## Suggested class

```text
src/three_dried/models/rmist_net.py
```

Suggested API:

```python
class RMISTNet(nn.Module):
    def __init__(
        self,
        operator: RMISTKernel,
        phases: int = 9,
        tau_init: float = 1e-4,
        theta_init: float = 1e-3,
    ):
        ...

    def forward(self, padded_measurement, range_m):
        ...
```

## Blocking regression: M4 -> M5

Set all nine phases to:

```text
tau_t = 1e-4
theta_t = 1e-3
```

Compare `RMISTNet` against accepted:

```python
rmist_ista(
    S,
    operator,
    range_m,
    tau=1e-4,
    theta=1e-3,
    iterations=9,
)
```

Require:

```text
complex64 relative difference < 1e-5
complex128 relative difference < 1e-12
```

This is the primary proof that Eq. (19) was unfolded into Eq. (20) without changing the algorithm.

## Other mandatory tests

1. `alpha0 == RM_dagger(S)`.
2. Exactly 9 phases execute.
3. `tau.shape == [9]`.
4. `theta.shape == [9]`.
5. Exactly 18 trainable scalar parameters.
6. Phase parameters are independently addressable.
7. One-phase update matches a literal Eq. (20) calculation.
8. Complex/two-channel conversion round trip preserves values.
9. `loss.backward()` gives finite gradients for all `tau_t` and `theta_t` in a deterministic test.
10. No optimizer step occurs.
11. CPU complex64 passes.
12. CPU complex128 reference passes.
13. CUDA complex64 passes when CUDA is available.
14. Batch input is tested if supported without changing M3-R physics semantics.

## Optional diagnostic trace

Record exactly 9 entries:

```text
phase
tau_t
theta_t
residual_norm
estimate_norm
```

Trace must not affect output.

## Required outputs

```text
docs/M5_RMIST_NET_REPORT.md
results/review/M5/
```

Keep only lightweight review artifacts under Git:

```text
summary_metrics.json
architecture.txt
parameter_values.json
phase_trace.json
```

Do not commit large checkpoints/tensors merely for M5 review.

## PASS criteria

M5-R passes only if:

- 9 phases;
- only phase-wise `tau_t`, `theta_t` trainable;
- exactly 18 trainable scalars;
- initialization equals `RM_dagger(S)`;
- Eq. (20) is implemented literally;
- Eq. (25)-(26) shrinkage is preserved;
- M5 output matches M4 9-step Ini-ISTA under identical parameters;
- gradients reach all phases in a deterministic test;
- no M6 training or Hammer inference is started;
- all repository regression tests pass.

After M5-R, STOP for independent review.
