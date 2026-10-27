# Current five-method baseline

The active extension uses full FMCW range-migration BP, RMA, true-adjoint ISTA, original ISTA-Net and frozen RMIST-Net. Full-size FMCW simulation and CUDA BP are evaluated by `scripts/benchmark_fmcw_bp.py`; see [FMCW BP assessment](docs/FMCW_BP_ASSESSMENT.md). The RM-slice unified entry explicitly skips full FMCW BP until suitable raw inputs/phase validation are available. The previous single-frequency BP is accessible only through `--bp-model narrowband_diagnostic`, with `--bp-backend fft_cpu`, `fft_cuda` or `literal`; its [old slice timings](docs/BP_ACCELERATION_REPORT.md) are not wideband results. Read [BASELINE_REPORT.md](BASELINE_REPORT.md) and [BASELINE_REVIEW.md](BASELINE_REVIEW.md) for learned models, negative real-data findings and SKIPPED conditions. The earlier Green-weighted FFT-BP/CV-ISTA runner below remains historical.

# 3DRIED imaging baseline suite

Independent branch `baseline/3dried-imaging-suite`: BP, RMA, ISTA, CV-ISTA and
fixed-weight RMIST; point simulation and single-Rx 3DRIED/Hammer; 100%, 70%,
50%, 30% sampling with both random 2-D points and random height lines.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/run_baseline_suite.py --datasets point_sim --output-dir results/baseline_points
python scripts/run_baseline_suite.py --output-dir results/baseline_full
```

Shared explicit acquisition parameters:
[configs/baseline_3dried.yaml](configs/baseline_3dried.yaml), explained in
[BASELINE_PARAMETERS](docs/BASELINE_PARAMETERS.md).

Read [BASELINE_SPEC](BASELINE_SPEC.md) for exact algorithm definitions and
[BASELINE_RUNBOOK](docs/BASELINE_RUNBOOK.md) for execution and output details.
The first point simulation uses the accepted RM range-slice model, and BP is
slice-wise narrowband spherical backprojection. Full raw FMCW BP/simulation
is not claimed. CV-ISTA is explicitly defined as classical complex L1 ISTA
with a true masked adjoint; it is not an unspecified learned network.

The raw capture is external and opened read-only. No voxel ground truth is
assumed for Hammer. Missing raw data produces an explicit blocked status,
not a successful real experiment. Existing M0–M8 implementations and reports
are retained; M9 is not merged as part of this branch.

The original conventional entry points remain available:

```bash
python scripts/smoke_test.py
python scripts/inspect_hammer.py --output results/hammer_audit.json
python scripts/run_rma.py --config configs/hammer.yaml
```
