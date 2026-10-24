# Running the independent baseline branch

## Install and test

```bash
git fetch origin
git switch --track origin/baseline/3dried-imaging-suite
python -m pip install -e '.[test]'
python -m pytest -q
```

If the branch already exists locally, use `git switch baseline/3dried-imaging-suite` instead. Use a clean worktree when another experiment is active. No raw capture is copied into Git.

Missing external Hammer data and CUDA make their integration tests explicitly skipped. This is not real/GPU acceptance. The actual experiment runner records missing input separately and exits 2.

## Small numerical smoke check

```bash
python scripts/run_baseline_suite.py --smoke --datasets point_sim --output-dir results/baseline_smoke
```

This uses a reduced aperture and 16×16 grid, labels its profile accordingly, and must not be presented as the full 3DRIED acquisition experiment. Acquisition spacings, carrier, fast-time parameters and bin convention remain unchanged.

## Full 3DRIED geometry: synthetic and real

```bash
python scripts/run_baseline_suite.py --output-dir results/baseline_full
```

Default: five methods, two patterns, four SRs, point simulation plus the external Hammer capture. Raw path comes from the shared YAML. Override only its directory if necessary:

```bash
python scripts/run_baseline_suite.py --dataset-dir '/your/3DRIED/Hammer/Hammer (1)' --output-dir results/baseline_full_local
```

Run synthetic only without access to the original capture:

```bash
python scripts/run_baseline_suite.py --datasets point_sim --output-dir results/baseline_points
```

Every output directory must be empty or new. Existing results are not silently overwritten. `--no-figures` omits plotting for batch runs. The current unified runner uses CPU NumPy/SciPy; it does not require a GPU. Existing PyTorch RMIST training/inference scripts remain available independently. A remote GPU connection alone does not accelerate this runner.

## Outputs

- `config_used.yaml`, `environment.json`, `command.txt`, `rmist_provenance.json`, `implementation_hashes.json`.
- `run_summary.json`, `REPORT.md`, `summary_metrics.csv`, `sampling_manifest.json`.
- `coordinates.npz`: SI coordinates shared by all methods.
- `point_sim/ground_truth.npz`: point and bandlimited GT.
- `<dataset>/<pattern>/srNNN_<method>.npz`: complex64 volume, shape `[13,512,512]`.
- Per-method diagnostics: observed masked RM residuals, fixed iteration settings/CV objective curves.
- Per-pattern MIP/slice with a common reference scale and per-panel normalization; 3-D point clouds; mask images; range energy profiles without per-slice gain adjustment.

CPU reconstruction times exclude source loading, figure generation and output serialization. There is no GPU speedup claim. Full-data RMA is marked reference proxy on real data, never ground truth. Geometry/normalization differs between geometric BP and the RM family, so raw magnitude or RM residual alone cannot establish a winner.

## Remaining acceptance work

1. Run the full command on the actual read-only Hammer capture.
2. Compare RMA100 with the accepted M2 output before interpreting real-data model differences.
3. Independently review the geometric BP approximation and the intended CV-ISTA definition.
4. If full raw FMCW BP/simulation is required, implement it separately and reconcile the documented one-bin range convention first.
5. Record actual outcomes; do not turn absent-data skips or HDF5 fixtures into real-data PASS.
