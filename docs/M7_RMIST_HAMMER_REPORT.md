# M7-R Real Hammer Slice-Wise RMIST Inference Report

Date: 2026-09-27 (Asia/Shanghai)

Status: **TECHNICAL PASS / REAL-DATA SUCCESS — STOP FOR INDEPENDENT REVIEW**

## Result

Strict RMIST Eq. (32) inference completed independently for all 13 accepted
Hammer range slices. The trained 50% reconstruction is finite and preserves a
clearly recognizable claw hammer: the diagonal handle is continuous, the head
is distinct, and the dominant response remains at the accepted bin 17 range of
614.310 mm. There is no catastrophic sim-to-real failure.

This is a deliberately narrow success classification: trained RMIST is nearly
indistinguishable from masked RMA and Ini-ISTA on this real capture and does
not demonstrate a material learned improvement. The complex relative
differences are `1.1364e-6` between RMIST50 and RMA50 and `1.1364e-6` between
RMIST50 and Ini-ISTA50. No superiority claim is made.

## Governance and model provenance

M7 was inference/evaluation only. There was no optimizer, training,
fine-tuning, adaptation, calibration learning, or parameter update. The model
was loaded read-only from the accepted M6 best-validation checkpoint:

```text
source: results/m6_rmist_training/checkpoints/best_validation.pt
best epoch: 99
SHA-256: 4b1925f2a6d361a848a81b7e70a2dedc2dc3e520726a29b2ac92a6de999c26ce
parameter equality with committed best_parameters: exact
```

The nine `tau` and nine `theta` values are recorded in
`results/review/M7/parameter_values.json`.

## Mask and real-input mapping

The exact accepted M6 mask was regenerated on the 512×512 grid with seed
`20260928`. Its uint8 SHA-256 is
`43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2`.
The centered 200×407 physical aperture occupies 81,400 samples; 40,682 are
observed after intersection with the M6 mask. The effective physical sampling
rate is therefore `0.4997788698`.

The raw data SHA-256 remains
`167c5977d75deddac782710c287d5bd49cffefe04a67e0c88fa45ab8e30a7475`.
Only MATLAB Rx1 / Python index 0 was loaded. M2 motion correction, serpentine
correction, 256-point range FFT, and MATLAB bins 12–24 were unchanged.

## Eq. (32) inference and comparisons

Each range slice was padded from `[z=200,x=407]` to `[512,512]`, multiplied by
the same effective mask, and reconstructed with its own exact physical range
and range-dependent `RM` / `RM_dagger` kernels. Learned scalars were shared
without modification. Outputs are stacked as `[range,z,x]`.

The comparison set is:

- accepted full-data M2 RMA reference;
- RMA50: `RM_dagger` of the masked measurement;
- Ini-ISTA50: nine Eq. (19) iterations with `tau=1e-4`, `theta=1e-3`;
- trained RMIST50: accepted M6 epoch-99 parameters.

M4 tuned diagnostic parameters were not used. Hammer RMA was visualization
reference only and was never a training label.

## Observed-sample consistency

For each sparse estimate, the reported diagnostic is

\[
E_{obs}=\frac{\|Mask_{eff}\odot(RM(\hat\alpha)-P(S_r))\|_2}
{\|Mask_{eff}\odot P(S_r)\|_2+\epsilon}.
\]

Aggregate values are arithmetic statistics over the 13 per-slice values:

| Method | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| RMA50 | 0.0849890322 | 0.0245418705 | 0.0612671338 | 0.1353330165 |
| Ini-ISTA50 | 0.0849890328 | 0.0245418712 | 0.0612671338 | 0.1353330165 |
| RMIST50 | 0.0849889525 | 0.0245418491 | 0.0612670742 | 0.1353328973 |

The near-identical residuals are diagnostic only. They neither establish nor
imply superior image quality. No voxel ground truth exists, so no real-data
NRMSE, PSNR, or TBR was computed.

## Qualitative assessment

All four panels use the same 500 mm x-z ROI, one common amplitude peak, and a
40 dB display range. At 50% sampling, the head and handle remain recognizable
and their accepted down-left orientation is preserved. The principal response
remains around 0.61 m. Compared with full-data M2, all sparse reconstructions
show denser fine-scale sampling texture; the trained model neither
catastrophically degrades the target nor visibly suppresses that texture beyond
the sparse baselines.

Accordingly:

```text
technical_pipeline: PASS
real_data_outcome: SUCCESS
learned improvement over sparse baselines: NOT DEMONSTRATED
```

The last line is an important non-blocking scientific limitation and a likely
manifestation of synthetic-to-real amplitude/distribution mismatch. Resolving
it would require a new authorized milestone; M7 did not rescale inputs or
change parameters.

## Reproduction and artifacts

```bash
PYTHONPATH=src python3 scripts/run_rmist_hammer.py \
  --config configs/rmist_hammer_inference.yaml --preflight-only
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_rmist_hammer.py \
  --config configs/rmist_hammer_inference.yaml
```

Lightweight review artifacts are in `results/review/M7/`. The large complex
volumes and the 3-D comparison are under ignored
`results/m7_rmist_hammer/`. The final repository suite passes 31/31 tests.

M7-R is complete and stops here for independent review. No retraining,
domain adaptation, calibration, or mask redesign is authorized.
