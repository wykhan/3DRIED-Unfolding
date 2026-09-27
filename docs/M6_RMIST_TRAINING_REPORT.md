# M6-R Strict RMIST Synthetic Training Report

Date: 2026-09-27 (Asia/Shanghai)
Status: **PASS — STOP FOR INDEPENDENT REVIEW**

## Result

Strict RMIST Eq. (27)--(31) training completed exactly 100 epochs on exactly
1000 reproducible 512×512 synthetic pairs. A separate 200-pair validation set
shows held-out NRMSE improvement from `0.6148313` at untrained initialization
to `0.5302653` at the best checkpoint (epoch 99). All scientific and
governance gates pass; M7-R remains locked.

## Equation mapping and dataset

- Eq. (28): each complex oracle has 100 unique uniformly located discrete
  impulses, independent phase `Uniform(0,2*pi)`, and amplitude
  `Uniform(0.5,1.0)`.
- Eq. (29): PyTorch complex representation is the accepted M5 equivalent of
  stacked real/imaginary channels.
- Eq. (30): `S = fixed_mask * strict_RM(alpha_gt) + circular_complex_AWGN`.
- Eq. (31): echoes remain complex internally under the accepted M5 choice.
- Eq. (27): optimization uses the mean over the batch of each sample's
  complex L2 reconstruction norm. NMSE is evaluation-only.

Training and validation seeds are disjoint. The mask seed is 20260928; its
realized rate is `0.4999198913574219` and uint8 SHA-256 is
`43f423a8eef01c1efa632b9945cf0f4146dc065d94cee001a3cca6f1b7374de2`.
Requested and achieved mean SNR are both 20 dB. Hammer RMA was not used.

## Training configuration

```text
grid/range:       512×512 / 0.5 m
train/validation: 1000 / 200 pairs
phases:           9
optimizer:        Adam
learning rate:    1e-4
epochs:           100 (no early stopping)
batch size:       8
tau initialization:   1e-4 for all phases
theta initialization: 1e-3 for all phases
```

There was no scheduler, warmup, weight decay, gradient clipping, positivity
transformation, or optimizer grouping. The accepted fixed-range `phi` and
`phi_dagger` values were cached as an **IMPLEMENTATION PERFORMANCE
OPTIMIZATION — algorithm unchanged**; exact equality is tested.

## Preflight

| Gate | Result |
|---|---:|
| identical seed reproduces oracle/echo/noise | PASS |
| mask deterministic and binary | PASS |
| achieved SNR | `20.0` dB |
| one-batch Eq. (27) loss | `4.73281765`, finite |
| all 18 gradients finite | PASS |
| tiny overfit, 20 steps | `4.73281765 → 4.12074471` |
| cached kernels equal accepted M3 values | PASS |
| M3–M5 regression suite before training | PASS |

## Training and validation results

| Metric | Untrained | Trained best | Ini-ISTA, 9 steps |
|---|---:|---:|---:|
| Eq. (27) loss | `4.688689` | `4.043289` | `4.688689` |
| NRMSE | `0.614831` | `0.530265` | `0.614831` |
| PSNR (dB) | `40.72047` | `42.00497` | `40.72047` |
| TBR, Eq. (33) (dB) | `3.28918` | `17.77521` | `3.28918` |

The equality of untrained RMIST-Net and Ini-ISTA is expected from the accepted
M4-to-M5 regression. Train loss decreased from `4.115710` at epoch 1 to
`4.048267` at epoch 100. Final validation loss was `4.043440`; the minimum
was `4.043289` at epoch 99. Training completed all 100 epochs in `228.535`
seconds with `629.004` MiB peak allocated GPU memory.

Initial `tau`/`theta` are the paper values. Best-epoch parameters are recorded
in `results/review/M6/summary_metrics.json`, with complete per-epoch values in
`parameter_history.csv`. All gradients, losses, and parameters remained
finite.

## Paper-unspecified choices

- first model at one sampling rate, SR=0.5;
- 200 separately generated validation pairs;
- exactly 100 scatterers in every training oracle;
- amplitude distribution `Uniform(0.5,1.0)`;
- batch size 8;
- fixed-mask realization and all random seeds;
- deterministic epoch shuffling;
- best checkpoint selected by minimum validation Eq. (27) loss;
- standard complex NRMSE/PSNR and Eq. (33) support-based TBR evaluation.

No result is claimed to reproduce the paper's tables exactly; this run uses
the accepted 3DRIED system parameters and documented choices.

## Reproduction and artifacts

```bash
PYTHONPATH=src python3 scripts/train_rmist.py --config configs/rmist_train.yaml
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Lightweight review artifacts are in `results/review/M6/`. The best-validation
and epoch-100 checkpoints are saved locally in
`results/m6_rmist_training/checkpoints/` and ignored by Git. The dataset is
procedural and was not saved. The final repository suite passes 27/27 tests.

No real Hammer inference was performed. M7-R remains locked.
