# RMIST_M6_START_PROMPT.md

# Long-running Codex task: M6-R strict RMIST synthetic training

The human explicitly authorizes M6-R training after accepted M5-R is merged to main.

This task implements RMIST Eq. (27)-(31).

Do NOT perform real Hammer inference.
Do NOT unlock M7-R.

## 1. Verify branch

The human should create `milestone/rmist-m6` from accepted `main` and commit the M6 specification files before launching this task.

Run:

```bash
git status
git branch --show-current
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git log --oneline --decorate -10
```

Expected:

```text
current branch = milestone/rmist-m6
```

If unexpected, STOP.

## 2. Read source-of-truth

Read completely:

- `AGENTS.md`
- `RESEARCH_SPEC.md`
- `PLAN.md`
- `STATUS.md`
- `DECISIONS.md`
- `docs/RMIST_REPRO_SPEC.md`
- `docs/M5_RMIST_NET_REPORT.md`
- `docs/M6_RMIST_TRAINING_SPEC.md`
- `docs/M6_REVIEW_CHECKLIST.md`

Inspect accepted:

- `src/three_dried/imaging/rmist_kernel.py`
- `src/three_dried/models/rmist_net.py`
- `src/three_dried/solvers/rmist_ista.py`
- all M3-M5 tests.

## 3. Unlock only M6-R

Update governance:

```text
M5-R: externally reviewed PASS
M6-R: UNLOCKED
M7-R: LOCKED
```

State that M6-R is synthetic training only.

Commit governance unlock before implementation.

## 4. Implement Eq. (28)-(31) dataset

Strict first reference config:

```text
grid: 512 x 512
range_m: 0.5
sampling_rate: 0.5
training_pairs: 1000
validation_pairs: 200
scatterers_per_image: 100
amplitude_distribution: Uniform(0.5, 1.0)
phase_distribution: Uniform(0, 2*pi)
noise_snr_db: 20
fixed_mask: true
```

Paper-unspecified choices must be labeled clearly.

Use the accepted strict `RM` operator.

Use one fixed deterministic 50% binary mask for the first model.

Do not use Hammer RMA as training labels.

## 5. Implement Eq. (27) loss

Train using:

```text
mean over batch of per-sample complex L2 reconstruction norm
```

equivalent to displayed Eq. (27).

Do NOT use NMSE as training loss.

NMSE/PSNR/TBR are evaluation only.

## 6. Training hyperparameters

Freeze:

```text
optimizer = Adam
lr = 1e-4
epochs = 100
phases = 9
tau_init = 1e-4
theta_init = 1e-3
batch_size = 8
```

Batch size is paper-unspecified.

If CUDA OOM:

```text
8 -> 4 -> 2 -> 1
```

Record final batch size.

Do not add scheduler, clipping, weight decay, warmup, early stopping, positivity transformations, or optimizer groups.

## 7. Optional regression-safe kernel cache

You may cache non-trainable `phi` / `phi_dagger` for the fixed training range/device/dtype.

Only keep caching if all M3-M5 regression tests still pass.

Do not alter kernel values or make them trainable.

## 8. Preflight gates

Before full training verify:

1. identical sample seed -> identical oracle/echo;
2. mask deterministic/binary and realized rate recorded;
3. achieved SNR near 20 dB;
4. one-batch forward/backward finite;
5. all 18 gradients finite;
6. tiny-overfit diagnostic on a small subset reduces loss;
7. all accepted M3-M5 tests pass.

If strict paper hyperparameters are scientifically unstable, STOP rather than silently changing them.

## 9. Full training

Train exactly 100 epochs on exactly 1000 training pairs.

Use separate deterministic validation data.

Record every epoch:

```text
train_loss
val_loss
tau[9]
theta[9]
tau_grad_norm
theta_grad_norm
learning_rate
epoch_time_seconds
peak_gpu_memory
```

Save locally:

- best validation checkpoint;
- epoch-100 checkpoint.

Do not early-stop.

## 10. Evaluation

On held-out validation set report:

- Eq. (27) loss;
- NRMSE;
- PSNR;
- TBR if definition is reliable.

Compare:

1. untrained RMIST-Net;
2. trained RMIST-Net;
3. M4 Ini-ISTA 9 steps;
4. optional tuned M4 diagnostic, clearly labeled 3DRIED-specific.

Do not claim exact reproduction of RMIST paper tables.

## 11. Required files

Create:

```text
docs/M6_RMIST_TRAINING_REPORT.md
configs/rmist_train.yaml
scripts/train_rmist.py
src/three_dried/training/
tests/test_rmist_training.py
results/review/M6/
```

Commit lightweight:

```text
summary_metrics.json
training_history.csv
parameter_history.csv
config_used.yaml
dataset_spec.json
mask_summary.json
training_curves.png
parameter_curves.png
```

Do not commit full dataset, checkpoints, or large arrays.

## 12. Validate and stop

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Apply every gate in `docs/M6_REVIEW_CHECKLIST.md`.

Update `STATUS.md`, `EXPERIMENTS.md`, and `DECISIONS.md`.

Push branch.

Do not merge.

Return:

```text
M6-R: PASS/FAIL
commit:

training pairs:
validation pairs:
batch size:
mask realized SR:
requested/achieved SNR:

epochs completed:
best epoch:
train loss initial/final:
val loss initial/best/final:

untrained NRMSE:
trained NRMSE:
Ini-ISTA NRMSE:
PSNR:
TBR:

initial tau:
final tau:
initial theta:
final theta:

runtime:
peak GPU memory:
tests:

paper-unspecified choices:
unresolved issues:

M7-R status: LOCKED
```

STOP for independent review.
