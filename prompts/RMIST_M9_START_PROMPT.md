# RMIST_M9_START_PROMPT.md

# Codex task: evaluate scale-normalized RMIST on real Hammer

The human explicitly authorizes M9 after accepted M8 is merged into main.

M9 is inference/evaluation only.

Do NOT train.
Do NOT fine-tune.
Do NOT change learned tau/theta.
Do NOT optimize a scale factor using Hammer labels.
Do NOT change RM/RM_dagger.
Do NOT redesign the M6 mask.

## 1. Verify branch

The human should create:

```text
milestone/rmist-m9
```

from accepted `main` and commit M9 specification files before launching.

Run:

```bash
git status
git branch --show-current
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git log --oneline --decorate -12
```

Expected:

```text
current branch = milestone/rmist-m9
```

If unexpected, STOP.

## 2. Read source-of-truth

Read completely:

- `AGENTS.md`
- `RESEARCH_SPEC.md`
- `PLAN.md`
- `STATUS.md`
- `DECISIONS.md`
- `docs/M7_RMIST_HAMMER_REPORT.md`
- `docs/M8_SIM2REAL_DIAGNOSTIC_REPORT.md`
- `docs/M9_SCALE_NORMALIZED_RMIST_SPEC.md`
- `docs/M9_REVIEW_CHECKLIST.md`

Inspect accepted M3-M8 code and artifacts.

## 3. Unlock M9 only

Update governance:

```text
M8: independently reviewed PASS — SCALE_MISMATCH_SUPPORTED
M9: UNLOCKED — scale-normalized inference feasibility only
```

Explicitly state:

```text
No training, fine-tuning, adaptive learned scale estimator, or parameter
modification is authorized.
```

Commit governance unlock before implementation.

## 4. Reproduce synthetic scale references

Using the accepted M6 validation generator and the same deterministic first 32
samples as M8, regenerate:

```text
synthetic alpha0 RMS median
synthetic alpha0 p95 median
synthetic observed-measurement RMS median
```

The alpha0 RMS median must reproduce M8 approximately:

```text
0.01376872018
```

within numerical tolerance.

If it does not, STOP and diagnose the mismatch.

Save:

```text
results/review/M9/synthetic_reference_scales.json
```

## 5. Implement normalization wrapper

Create:

```text
src/three_dried/imaging/rmist_normalization.py
```

Implement deterministic scale-estimation and normalize/infer/de-normalize
helpers.

For each slice:

```text
S_norm = c * S
alpha_norm = fixed_RMIST(S_norm)
alpha_out = alpha_norm / c
```

No learned parameters may change.

Blocking regression:

```text
c = 1
```

must reproduce ordinary accepted RMIST output.

## 6. Implement fixed variants

Compare exactly:

```text
N0 = no normalization
N1 = one global alpha0-RMS scale across all Hammer slices
N2 = per-slice alpha0-RMS matching
N3 = per-slice alpha0-p95 matching
N4 = per-slice observed-measurement RMS matching
```

Do not add more variants after viewing results.

Record every scale factor for bins 12-24.

For N1, use one global real scale based on median alpha0 RMS across the 13
Hammer slices so relative inter-slice amplitude is preserved.

## 7. Hammer inference

Use the accepted M7:

```text
Rx1 / Python 0
bins 12-24
same M6 50% mask
same epoch-99 tau/theta
same range-dependent kernels
same padding/preprocessing
```

For each method, reconstruct all 13 slices.

No different-range batching through one scalar range.

## 8. Phase activity

Using accepted M8 definitions, record for each method and range slice:

```text
max/median E_grad
max/median E_shrink
max/median E_update
final E_RMA
```

Confirm N0 reproduces M8 inactivity.

Determine whether each normalized method satisfies the M9 `DYNAMICS RESTORED`
criterion.

## 9. Real-data consistency

Compute accepted observed-sample consistency for:

```text
RMA50
raw RMIST50
N1
N2
N3
N4
```

Do not interpret residual alone as image quality.

## 10. M2 reference proxies

M2 full-data RMA is a reference proxy, NOT ground truth.

Within the same ROI compute for each method:

```text
magnitude cosine similarity to M2
normalized magnitude L2 distance to M2
```

Required at:

```text
per-slice
full volume
x-z MIP
```

Clearly label every such number as `M2-reference proxy`.

Do not report them as true NRMSE/PSNR.

## 11. Range-profile preservation

For every method:

```text
p_r = ||alpha[:,:,r]||_2
```

for all 13 slices.

Normalize the 13-element range profile and compute:

```text
cosine similarity to M2 range profile
dominant range bin
```

This is a blocking evaluation for per-slice normalization because it may
artificially equalize weak/strong slices.

## 12. Visual comparison

Use the exact same ROI/orientation and one fair common dB visualization
policy.

Create figures comparing:

```text
M2 full-data
RMA50
raw RMIST50
N1 global RMS
N2 per-slice RMS
N3 per-slice p95
N4 measurement RMS
```

Required:

```text
comparison_mip.png
representative_slices.png   # bins 16,17,18
range_profile.png
```

Do not call a method superior solely because it produces larger updates.

## 13. Outcome classification

Use exactly one:

```text
NORMALIZATION_FEASIBLE_WITH_PROXY_BENEFIT
DYNAMICS_RESTORED_BUT_IMAGE_BENEFIT_UNPROVEN
NORMALIZATION_DEGRADES_REAL_RECONSTRUCTION
GLOBAL_AND_PER_SLICE_TRADEOFF_UNRESOLVED
INCONCLUSIVE
```

Use the criteria in `docs/M9_SCALE_NORMALIZED_RMIST_SPEC.md`.

Do not force the first outcome.

## 14. Required files

Create:

```text
docs/M9_SCALE_NORMALIZED_RMIST_REPORT.md
configs/rmist_scale_normalized.yaml
scripts/run_rmist_scale_normalized.py
src/three_dried/imaging/rmist_normalization.py
tests/test_rmist_normalization.py
results/review/M9/
```

Commit lightweight:

```text
summary_metrics.json
synthetic_reference_scales.json
normalization_scales.csv
phase_activity_by_method.csv
observed_consistency.csv
m2_reference_proxies.csv
range_profile.csv
comparison_mip.png
representative_slices.png
range_profile.png
```

Do not commit large volume arrays.

## 15. Regression tests

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Required:

- `c=1` wrapper regression PASS;
- N0 matches accepted M7 raw RMIST;
- synthetic reference RMS matches M8;
- no optimizer/backward/train calls;
- all prior regressions pass.

## 16. Commit and stop

Update:

- `STATUS.md`
- `EXPERIMENTS.md`
- `DECISIONS.md` only for new fixed implementation choices.

Commit and push:

```bash
git add ...
git commit -m "feat: evaluate scale-normalized RMIST on Hammer"
git push origin milestone/rmist-m9
```

Do not merge.

Return:

```text
M9: PASS/FAIL
commit:

outcome classification:

synthetic references:
alpha0 RMS:
alpha0 p95:
measurement RMS:

normalization scales:
N1 global:
N2 per-slice range:
N3 per-slice range:
N4 per-slice range:

dynamics restored:
N1:
N2:
N3:
N4:

M2-reference proxy summary:
raw RMIST:
N1:
N2:
N3:
N4:

range-profile similarity:
raw RMIST:
N1:
N2:
N3:
N4:

dominant range bins:

observed consistency:
raw RMIST:
N1:
N2:
N3:
N4:

visual conclusion:
best-supported normalization variant:
important limitations:

tests:
working tree clean:

STOP.
```
