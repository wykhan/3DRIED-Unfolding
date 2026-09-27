# M6_RMIST_TRAINING_SPEC.md

# M6-R Strict RMIST Synthetic Training Specification

## 1. Scope

M6-R reproduces RMIST training Eq. (27)-(31) using the independently accepted M3-R/M4-R/M5-R implementation.

M6-R includes:

1. synthetic oracle generation;
2. echo generation through strict RM;
3. sampling-mask application;
4. complex AWGN;
5. Eq. (27) reconstruction loss;
6. Adam training;
7. 100 epochs;
8. checkpoint selection and synthetic validation;
9. training diagnostics.

M6-R does NOT include real Hammer inference. M7-R remains locked.

---

## 2. Paper-supported training facts

The RMIST paper explicitly supports:

- synthetic training because real measured training data are insufficient;
- randomly generated target coordinates;
- random phase uniform on `[0, 2*pi)`;
- randomly distributed amplitudes in the same order of magnitude as RMA references;
- echo generation by `Mask * RM(alpha_gt) + n`;
- complex AWGN;
- default 20 dB training noise;
- 1000 oracle/echo training pairs;
- 9 phases;
- Adam optimizer;
- learning rate `1e-4`;
- `tau_t` initialization `1e-4`;
- `theta_t` initialization `1e-3`;
- 100 epochs.

Do not silently change these values.

---

## 3. Paper-unspecified implementation choices

The paper does not fully specify:

- batch size;
- validation-set size/split;
- exact number of scatterers per training oracle;
- exact amplitude probability distribution;
- exact training-mask realization/schedule;
- random seeds;
- checkpoint-selection rule;
- whether a separate network is trained for each sampling rate.

Every such choice must be labeled:

```text
PAPER UNSPECIFIED IMPLEMENTATION CHOICE
```

---

## 4. First strict 3DRIED training configuration

Use one clearly defined reference condition:

```text
grid size: 512 x 512
range: 0.5 m
sampling rate: 0.5
training pairs: 1000
validation pairs: 200
training noise: 20 dB
phases: 9
epochs: 100
optimizer: Adam
learning rate: 1e-4
tau init: 1e-4
theta init: 1e-3
```

The choices `SR=0.5` as the first single training condition and `validation_pairs=200` are paper-unspecified reproduction choices.

Do not mix multiple sampling rates into the first strict training run.

---

## 5. Oracle generation — Eq. (28)

Required:

- uniform random target coordinates;
- independent phase `U(0, 2*pi)`;
- random amplitudes;
- complex oracle image.

For the first reference run use:

```text
scatterers per image: 100
amplitude ~ Uniform(0.5, 1.0)
```

Both are implementation choices: the paper's quantitative benchmark uses 100 random points, while the training section itself does not explicitly state that every training oracle contains exactly 100 targets; it also does not give an amplitude distribution.

For the ideal point-scattering baseline, use discrete impulses for `p(m,n)` rather than inventing an undocumented blur PSF.

Do not use Hammer RMA as supervised ground truth.

---

## 6. Sampling mask

Generate one deterministic fixed binary 2-D random mask with target sampling rate:

```text
0.5
```

Reuse the same fixed mask for the first training/validation model.

Record:

- mask seed;
- realized sampling rate;
- shape;
- hash if practical.

The exact mask realization is paper-unspecified.

Mask is not trainable.

---

## 7. Range

Use:

```text
range_m = 0.5
```

for all samples in the first M6 run.

This supports same-range batching and matches the paper's point-target quantitative setup.

Do not train a multi-range model in the first M6 run.

---

## 8. Echo generation — Eq. (30)-(31)

For each oracle:

\[
S_{\rm clean}=Mask\odot RM(\alpha_{gt})
\]

\[
S_{\rm cplx}=S_{\rm clean}+n.
\]

Use circular complex AWGN.

Target:

```text
20 dB SNR
```

Record requested and achieved SNR statistics.

Internal PyTorch complex tensors are allowed, consistent with accepted M5-R.

---

## 9. Dataset size

Generate exactly:

```text
1000 training pairs
```

as paper-reported.

Generate a separate deterministic validation set:

```text
200 validation pairs
```

with a disjoint seed.

Do not take the 200 validation samples out of the paper-reported 1000 training pairs.

Dataset generation must be reproducible from config and seeds.

The full dataset must not be committed to Git.

---

## 10. Loss — Eq. (27)

Use the displayed Eq. (27) form:

\[
C(\Theta)=\frac{1}{N_s}\sum_i
\|\hat\alpha(\Theta,S_i)-\alpha_{{label},i}\|_2.
\]

For complex tensors, implement mathematically equivalent per-sample complex L2 reconstruction norm and average across the batch.

Do NOT use NMSE as training loss.

NRMSE, PSNR, and TBR are evaluation metrics only.

The paper text calls this averaged RMSE, while the displayed equation is an averaged L2 norm; the strict baseline follows the displayed equation and records this interpretation.

---

## 11. Optimizer / schedule

Freeze:

```text
optimizer: Adam
learning_rate: 1e-4
epochs: 100
phases: 9
tau_init: 1e-4
theta_init: 1e-3
```

Do not add:

- LR scheduler;
- warmup;
- weight decay;
- gradient clipping;
- positivity reparameterization;
- early stopping that terminates the 100-epoch run;
- separate optimizer parameter groups.

If strict settings become scientifically unstable, STOP and report instead of silently changing them.

---

## 12. Batch size

Paper-unspecified initial choice:

```text
batch_size = 8
```

If CUDA OOM occurs, Codex may reduce:

```text
8 -> 4 -> 2 -> 1
```

and record the final value.

Do not change other scientific hyperparameters to solve OOM.

---

## 13. Optional kernel caching

A non-trainable cache of range-dependent `phi` / `phi_dagger` is allowed for fixed `(range_m, dtype, device)`.

Requirements:

- values exactly match accepted M3-R kernels;
- no learned physics parameter;
- all M3-M5 regression tests remain passing.

Record as:

```text
IMPLEMENTATION PERFORMANCE OPTIMIZATION — algorithm unchanged
```

---

## 14. Checkpointing

The paper says the optimal weights file is chosen for test but does not specify the selection rule.

Use:

- best checkpoint by lowest validation Eq. (27) loss;
- final epoch-100 checkpoint.

This is paper-unspecified.

Do not early-stop.

Do not commit large checkpoints to Git.

---

## 15. Required epoch logging

Record every epoch:

```text
epoch
train_loss
val_loss
tau_1 ... tau_9
theta_1 ... theta_9
tau_grad_norm
theta_grad_norm
learning_rate
epoch_time_seconds
peak_gpu_memory
```

Also record:

```text
initial tau/theta
best epoch
best validation loss
final tau/theta
total runtime
```

---

## 16. Preflight gates before the full 100 epochs

### A. Dataset determinism
Same sample seed gives identical oracle/echo/noise.

### B. SNR
20 dB requested SNR is correctly achieved.

### C. Mask
Mask is deterministic, binary, and realized SR is recorded.

### D. One-batch backward
Loss finite; all 18 parameter gradients finite.

### E. Tiny overfit
Use a tiny subset (for example 8 samples) and a short diagnostic to confirm the training pipeline can reduce loss.

This is debugging only, not the reported strict run.

### F. Regression
All accepted M3-M5 tests still pass.

Only then begin full training.

---

## 17. Post-training synthetic evaluation

Evaluate held-out validation data using:

- Eq. (27) loss;
- NRMSE;
- PSNR;
- TBR if target/background regions are defined reliably.

Compare:

1. untrained RMIST-Net;
2. trained RMIST-Net;
3. 9-step Ini-ISTA;
4. optional M4 tuned diagnostic, clearly labeled 3DRIED-specific.

Do not claim exact reproduction of the paper's Table II values because this project uses 3DRIED system parameters and paper-unspecified choices.

---

## 18. Required files

Create:

```text
docs/M6_RMIST_TRAINING_REPORT.md
configs/rmist_train.yaml
scripts/train_rmist.py
src/three_dried/training/
tests/test_rmist_training.py
results/review/M6/
```

Commit lightweight review artifacts:

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

Do not commit:

```text
1000-pair dataset
large .npy arrays
.pt/.pth checkpoints
```

---

## 19. PASS criteria

M6-R passes only if:

1. exactly 1000 reproducible training pairs;
2. separate deterministic validation set;
3. 20 dB complex AWGN validated;
4. fixed 50% mask documented;
5. Eq. (27) loss used;
6. Adam lr `1e-4`;
7. 100 epochs complete;
8. accepted 9-phase M5 model used;
9. initial tau/theta match paper;
10. all 18 parameters receive finite gradients;
11. train/validation losses stay finite;
12. trained model improves held-out validation reconstruction over untrained initialization;
13. all M3-M5 regressions remain passing;
14. no real Hammer inference occurs;
15. all paper-unspecified choices are explicitly labeled.

After M6-R, STOP for independent review before M7-R.
