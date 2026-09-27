# M5-R Strict RMIST-Net Architecture Report

Date: 2026-09-27 (Asia/Shanghai)
Status: **PASS — STOP FOR INDEPENDENT REVIEW**

## Result

The strict RMIST-Net architecture implements Eqs. (20)--(26) as exactly nine
unfolded phases. Its only trainable tensors are independent real-valued
`tau[9]` and `theta[9]`, for exactly 18 trainable scalars. Initialization is
`RM_dagger(S)`. The blocking M4-to-M5 relative differences are `0.0` for
complex64 and `1.21362588613636e-13` for complex128.

## Equation-to-code mapping

`src/three_dried/models/rmist_net.py` maps the paper equations directly:

```text
initialize():             alpha_0 = RM_dagger(S)                         Eq. (21)
phase_update(), stage 1:  v_t = S - RM(alpha_t)                          Eq. (20)/(23)
phase_update(), stage 2:  r_t = alpha_t + tau_t RM_dagger(v_t)           Eq. (20)/(24)
phase_update(), stage 3:  alpha_t+1 = r_t/(|r_t|+eps) ReLU(|r_t|-theta) Eq. (25)-(26)
```

`RMISTNet.forward()` executes this phase update exactly nine times. It rejects
any other phase count in the strict baseline. There is no momentum, CNN
proximal, attention, support selection, learned mask, learned measurement
matrix, or trainable RM kernel.

## Parameters and representation

The parameter tensors are:

```text
tau:   shape [9], initialized to 1e-4
theta: shape [9], initialized to 1e-3
total: 18 independently addressable trainable real scalars
```

PyTorch complex tensors are used internally as a **PAPER UNSPECIFIED
IMPLEMENTATION CHOICE**. This is mathematically equivalent to the paper's
separate real/imaginary paths because RM and RM_dagger act in complex form and
both components share the Eq. (25)--(26) magnitude scale. Tested
`complex_to_two_channel()` and `two_channel_to_complex()` helpers provide the
conceptual `[...,2,H,W]` mapping and preserve values exactly.

## Blocking validation

| Gate | Result |
|---|---:|
| phases | 9 |
| trainable tensor names | `tau`, `theta` |
| trainable scalar count | 18 |
| alpha0 relative difference, complex64/128 | `0.0` / `0.0` |
| M4-to-M5 relative difference, complex64 | `0.0` (`<1e-5`) |
| M4-to-M5 relative difference, complex128 | `1.21362588613636e-13` (`<1e-12`) |
| deterministic phase trace entries | 9 |
| all `tau` gradients finite/nonzero | PASS |
| all `theta` gradients finite/nonzero | PASS |
| single and batch input | PASS |
| CPU complex64 / complex128 | PASS / PASS |
| CUDA complex64 | PASS, NVIDIA GeForce RTX 4090 |
| complete repository tests | 22/22 PASS |

The gradient check uses a deterministic complex64 input with the accepted
strict `RMISTKernel`, a simple squared reconstruction loss, and one backward
call. It creates no optimizer and takes no optimizer step. Ground truth does
not enter a reconstruction update.

## Reproduction and artifacts

```bash
PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  scripts/run_rmist_net_validation.py --output results/review/M5

PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  -m unittest discover -s tests -v
```

The exact validation command and environment are embedded in
`results/review/M5/summary_metrics.json`. Lightweight evidence consists of
`summary_metrics.json`, `architecture.txt`, `parameter_values.json`, and
`phase_trace.json`. No checkpoint or large tensor is present.

## Deviations and stop state

- **SYSTEM ADAPTATION:** the accepted 3DRIED RM/RM_dagger geometry, sign,
  support, and FFT conventions remain unchanged.
- **PAPER UNSPECIFIED IMPLEMENTATION CHOICE:** PyTorch complex internal
  representation; deterministic 32-square architecture-validation input and
  seed; epsilon `1e-12`, matching accepted M4 shrinkage.

No scientific discrepancy is unresolved. M6 training was not started: no
optimizer, epoch loop, 1000-pair dataset, or checkpoint-selection logic was
created. Real Hammer inference was not performed. M6-R and M7-R remain locked.
