# M3-R Strict RMIST Kernel Report

Date: 2026-09-27 (Asia/Shanghai)
Status: **PASS**

## Result

The strict RMIST `RM` / `RM_dagger` pair is implemented without replacing
`RM_dagger` by a Hilbert adjoint. On the 3DRIED grid, the reciprocal product
has maximum absolute error `4.4755e-16`, and `RM_dagger` reproduces the
human-approved M2 Hammer volume with complex relative difference
`2.2378e-7`, below the required `1e-5` threshold.

## Paper equations and code mapping

RMIST Eq. (13) defines the reciprocal phase kernels; Eqs. (14) and (15)
apply each through a forward 2-D FFT, Hadamard multiplication, and inverse
2-D FFT. `RMISTKernel.phi`, `phi_dagger`, `rm`, and `rm_dagger` implement
those equations directly in `src/three_dried/imaging/rmist_kernel.py`.

The paper convention is mapped to the validated 3DRIED propagation sign:

```text
RM forward phi       = (1/ky) * exp(+j*ky*r)
RM_dagger inverse phi = ky * exp(-j*ky*r)
```

This sign change is a **SYSTEM ADAPTATION**. The mandatory reciprocal
amplitude structure is unchanged. `RM_dagger` is described only as RMIST's
inverse RM kernel.

## Grid, support, and padding

The implementation uses M2's `k=2*pi*fc/c` and
`ky=sqrt(4*k^2-kx^2-kz^2)` grid with inclusive `linspace` endpoints. The
kernel is FFT-shifted before spectral multiplication, exactly as in M2.
Non-propagating samples and samples with `ky <= 1e-12` are invalid; both
kernels are explicitly zero there, so the forward kernel never divides by
zero or near-zero `ky`. On the 512-square 3DRIED grid, 258,772 bins are
valid and 3,372 are invalid; the smallest valid `ky` is 27.2048 rad/m.

The physical `[z=200,x=407]` echo is centered in `[512,512]`, with padding
`z=(156,156)` and `x=(52,53)`. Cropping exactly reverses this operation.

FFTs use PyTorch's default `backward` normalization (unnormalized forward,
`1/N` inverse). This is a documented reference-reproduction exception that
matches M2/MATLAB. It does not alter the reciprocal-kernel algorithm.

## Validation

| Test | Result |
|---|---:|
| Exact formula, complex128 | PASS (bit-exact against constructed formula) |
| Reciprocal max / RMS error | `4.4755e-16` / `1.1609e-16` |
| Invalid and near-zero `ky` handling | PASS |
| `[200,407]` to `[512,512]` padding/crop | PASS |
| Full-domain round trip, complex64 | `4.1301e-7` relative error |
| Full-domain round trip, complex128 | `7.2185e-16` relative error |
| Hammer M2 complex relative difference | `2.2378e-7` |
| Worst individual Hammer slice | `2.2494e-7` |
| CUDA complex64 smoke test | PASS (RTX 4090) |

The synthetic scene was created in the spectral domain and restricted to
valid propagating support before inverse FFT. Thus its round-trip metric is
separate from physical-aperture cropping. No Hilbert-adjoint identity was
used as evidence.

## Reproduction

```bash
PYTHONPATH=src /home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  -m unittest discover -s tests -p 'test_rmist_kernel.py' -v

/home/superws/anaconda3/envs/ComplexValue/bin/python3 \
  scripts/run_rmist_kernel_validation.py --config configs/rmist_kernel.yaml
```

Artifacts are under `results/m3_rmist_kernel/`: `metrics.json`, the exact
command, copied config, environment record, and reproduced complex Hammer
volume.

## Deviations from RMIST

- **SYSTEM ADAPTATION:** 3DRIED propagation signs and acquisition geometry.
- **PAPER UNSPECIFIED IMPLEMENTATION CHOICE:** `ky <= 1e-12` is excluded to
  make singular support handling explicit.
- **PAPER UNSPECIFIED IMPLEMENTATION CHOICE:** PyTorch with M2-compatible
  default FFT normalization; kernel geometry is formed in float64 before
  casting to the requested complex dtype.

There are no unresolved scientific discrepancies. M3-R passes self-review,
so M4-R may proceed under the authorized start prompt.
