# M6_REVIEW_MERGE_PROMPT.md

# Codex task: record accepted M6-R review and merge to main

Independent review result:

```text
M6-R: PASS WITH NON-BLOCKING ISSUES — independently reviewed
Blocking scientific issues: NONE
```

Non-blocking observations:

1. Validation improvement is real but modest: NRMSE `0.6148313 -> 0.5302653`; do not claim paper-table reproduction.
2. M6 used Python 3.11 / PyTorch 2.2.2 / CUDA 12.1 rather than the earlier M3-M5 environment; record this, but it is not blocking because all regression tests pass.
3. The best checkpoint is ignored by Git. Before M7, record checkpoint provenance (SHA-256 if the local file exists) and verify its parameters match committed `best_parameters`.
4. M6 trained one fixed SR=0.5 mask at range 0.5 m; M7 must use the same learned scalar parameters with range-dependent RM kernels and must not silently claim full-data RMIST inference.

This task is repository governance/provenance only.

Do NOT perform Hammer inference.
Do NOT create `milestone/rmist-m7`.
Do NOT retrain M6.
Do NOT change accepted M6 scientific code.
Do NOT force-push or delete branches.

## 1. Verify state

Expected branch:

```text
milestone/rmist-m6
```

Run:

```bash
git status
git branch --show-current
git fetch origin
git log --oneline --decorate --graph -15
git merge-base --is-ancestor origin/main HEAD
```

If unexpected, STOP.

## 2. Record M6 independent review

Update:

- `STATUS.md`
- `PLAN.md`
- `AGENTS.md`
- `REVIEW_CHECKLIST.md`
- `docs/M6_REVIEW_CHECKLIST.md`
- `docs/RMIST_REPRO_SPEC.md`
- `DECISIONS.md`

Active status must become:

```text
M6-R: INDEPENDENTLY REVIEWED PASS
M7-R: LOCKED
```

Append a decision equivalent to:

```markdown
## D030 — M6-R independent review accepted

**Status:** FIXED

- Strict RMIST Eq. (27)-(31) synthetic training accepted.
- 1000 training pairs, 20 dB AWGN, Adam lr=1e-4, 9 phases, 100 epochs verified.
- Held-out validation NRMSE improved from 0.6148313 to 0.5302653.
- The improvement is accepted as a baseline result, not as reproduction of the paper's exact table values.
- M6 paper-unspecified choices remain documented.
- M7-R remains locked pending separate human authorization.
```

## 3. Record best-model provenance

Preferred local checkpoint:

```text
results/m6_rmist_training/checkpoints/best_validation.pt
```

If it exists:

1. compute SHA-256;
2. load it read-only;
3. verify epoch is 99;
4. verify loaded `tau[9]` and `theta[9]` match committed
   `results/review/M6/summary_metrics.json -> best_parameters`.

Write lightweight:

```text
results/review/M6/checkpoint_provenance.json
```

containing:

```text
path
exists
sha256
epoch
parameter_match
fallback_reconstruction_possible_from_committed_best_parameters
```

If checkpoint is absent, do NOT fail M6 acceptance; record `exists=false`.
M7 can reconstruct the inference-only model from the 18 committed best
parameters because they completely define the trainable state.

Narrowly update `.gitignore` only if needed to commit this JSON, not the checkpoint.

## 4. Regression tests

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

If tests fail, STOP.

## 5. Commit review state

Commit only governance/provenance updates:

```bash
git add AGENTS.md PLAN.md STATUS.md DECISIONS.md REVIEW_CHECKLIST.md \
  docs/M6_REVIEW_CHECKLIST.md docs/RMIST_REPRO_SPEC.md \
  results/review/M6/checkpoint_provenance.json .gitignore

git commit -m "chore: record independent M6-R acceptance"
git push origin milestone/rmist-m6
```

Omit unchanged paths from `git add` as needed.

## 6. Merge to main

Proceed only if tests pass and no unexpected main divergence exists:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge --no-ff milestone/rmist-m6 -m "merge: accept strict RMIST M6-R training"
git push origin main
```

On conflict, STOP.

## 7. Verify

Confirm:

- M6 implementation commit `b393082` is in main;
- M6 review-state commit is in main;
- M7 remains locked;
- no Hammer inference was performed;
- M6 branch remains available.

Return:

```text
M6 review-state cleanup: PASS/FAIL
review-state commit:
tests passed:

checkpoint exists:
checkpoint sha256:
checkpoint parameter match:

merge: PASS/FAIL
merge commit:
main push: PASS/FAIL

M6-R status: INDEPENDENTLY REVIEWED PASS
M7-R status: LOCKED

STOP.
```
