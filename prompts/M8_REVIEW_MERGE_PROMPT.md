# M8_REVIEW_MERGE_PROMPT.md

# Codex task: record accepted M8 diagnosis and merge to main

Independent review result:

```text
M8: PASS — independently reviewed
Root-cause classification: SCALE_MISMATCH_SUPPORTED
Blocking implementation issues: NONE
```

Accepted scientific interpretation:

- M8 measured an approximately `2.3104e9` Hammer/synthetic `alpha0` RMS ratio.
- Raw Hammer phase-1 residual/correction activity is comparable to synthetic control.
- Hammer threshold/shrinkage activity is effectively zero at the raw real-data scale.
- Counterfactual RMS scale alignment restores strong phase dynamics.
- This supports scale mismatch as the mechanism of RMIST inactivity on Hammer.
- M8 does NOT establish that normalization improves real Hammer image quality.
- Broader distribution mismatch is not ruled out.

This task is repository governance only.

Do NOT add production normalization.
Do NOT train or fine-tune.
Do NOT change M6 learned parameters.
Do NOT create `milestone/rmist-m9`.
Do NOT start M9 experiments.
Do NOT force-push or delete branches.

## 1. Verify state

Expected branch:

```text
milestone/rmist-m8
```

Run:

```bash
git status
git branch --show-current
git fetch origin
git log --oneline --decorate --graph -15
git merge-base --is-ancestor origin/main HEAD
```

Confirm M8 implementation commit:

```text
77c6d6a
```

is present and the working tree is clean.

If unexpected, STOP.

## 2. Record independent M8 review

Update:

- `STATUS.md`
- `PLAN.md`
- `AGENTS.md`
- `REVIEW_CHECKLIST.md`
- `docs/M8_REVIEW_CHECKLIST.md`
- `DECISIONS.md`

Use precise wording:

```text
M8: INDEPENDENTLY REVIEWED PASS
Root cause supported: SCALE_MISMATCH_SUPPORTED

Important limitation:
M8 establishes restoration of RMIST internal phase dynamics after scale
alignment, but does not establish improved real-data reconstruction quality.
```

Append a decision equivalent to:

```markdown
## D032 — M8 independent review accepted

**Status:** FIXED

- Hammer/synthetic initialization amplitude scale mismatch is accepted as the
  dominant demonstrated mechanism behind M7 RMIST inactivity.
- Real/synthetic `alpha0` RMS differs by approximately `2.3104e9` on the
  primary Hammer slice.
- Counterfactual scale alignment restores threshold/shrinkage and phase-update
  activity without changing learned parameters.
- Residual degeneracy is not the dominant phase-1 explanation.
- Normalization as an image-improvement method is NOT yet validated.
- M9 may test scale-normalized inference without retraining.
```

Do not alter accepted M8 numerical results.

## 3. Regression tests

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

If tests fail unexpectedly, STOP.

## 4. Commit review state

Commit only review/governance updates:

```bash
git add AGENTS.md PLAN.md STATUS.md DECISIONS.md REVIEW_CHECKLIST.md \
  docs/M8_REVIEW_CHECKLIST.md

git commit -m "chore: record independent M8 acceptance"
git push origin milestone/rmist-m8
```

Record the commit hash.

## 5. Merge to main

Proceed only if tests pass and origin/main has no unexpected divergence:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge --no-ff milestone/rmist-m8 -m "merge: accept RMIST M8 sim-to-real diagnosis"
git push origin main
```

If any merge conflict occurs, STOP.

## 6. Verify

Confirm:

- `77c6d6a` is contained in main;
- M8 review-state commit is contained in main;
- main contains accepted M0-M8 history;
- no M9 implementation exists yet.

Return:

```text
M8 review-state cleanup: PASS/FAIL
review-state commit:
tests passed:

merge: PASS/FAIL
merge commit:
main push: PASS/FAIL

M8 status: INDEPENDENTLY REVIEWED PASS
root cause: SCALE_MISMATCH_SUPPORTED
normalization image-quality benefit: NOT YET VALIDATED
M9 status: LOCKED

STOP.
```
