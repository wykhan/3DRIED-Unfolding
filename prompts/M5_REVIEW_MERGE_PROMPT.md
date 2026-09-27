# M5_REVIEW_MERGE_PROMPT.md

# Codex task: record accepted M5-R review and merge to main

Independent review result:

```text
M5-R: PASS — independently reviewed
Blocking scientific issues: NONE
```

This task is repository governance only.

Do NOT implement M6-R.
Do NOT create an optimizer or training loop.
Do NOT create `milestone/rmist-m6`.
Do NOT perform Hammer inference.
Do NOT force-push or delete branches.

## 1. Verify state

Run:

```bash
git status
git branch --show-current
git fetch origin
git log --oneline --decorate --graph -12
git merge-base --is-ancestor origin/main HEAD
```

Expected branch:

```text
milestone/rmist-m5
```

Expected clean working tree and ancestry from accepted `main`.

If unexpected, STOP.

## 2. Record independent review

Update governance only.

### STATUS.md

Set active status:

```text
M0 PASS — human reviewed
M1 PASS — human reviewed
M2 PASS — human reviewed
M3-R PASS — independently reviewed
M4-R PASS — independently reviewed
M5-R PASS — independently reviewed
M6-R LOCKED
M7-R LOCKED
```

Next action: merge M5-R to main; M6 requires separate human authorization.

### PLAN.md

Mark M5-R `COMPLETED + EXTERNALLY REVIEWED PASS`.
Keep M6-R/M7-R locked.

### AGENTS.md

Record M5-R as independently reviewed PASS.
Keep M6-R/M7-R locked.

### REVIEW_CHECKLIST.md

Fix stale branch wording and record:

```text
M5-R external review: PASS
```

Preserve M3-R/M4-R history.

### docs/M5_REVIEW_CHECKLIST.md

Set external verdict to:

```text
PASS — independently reviewed
```

Leave the M6 human-authorization gate closed.

### docs/RMIST_REPRO_SPEC.md

If M5 status still says self-validated/awaiting review, change only that status to externally reviewed PASS. Do not alter equations.

### DECISIONS.md

Append:

```markdown
## D028 — M5-R independent review accepted

**Status:** FIXED

- Strict 9-phase RMIST-Net Eq. (20)-(26) accepted.
- Exactly 18 trainable scalars: tau[9], theta[9].
- M4-to-M5 regression passed.
- Gradient-flow validation passed.
- PyTorch complex internal representation remains a paper-unspecified implementation choice.
- M6-R remains locked until separate human authorization.
```

## 3. Regression tests

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

If tests unexpectedly fail, STOP. Do not modify accepted M5 science merely to force merge.

## 4. Commit review state

```bash
git diff
git status

git add AGENTS.md PLAN.md STATUS.md DECISIONS.md REVIEW_CHECKLIST.md \
  docs/M5_REVIEW_CHECKLIST.md docs/RMIST_REPRO_SPEC.md

git commit -m "chore: record independent M5-R acceptance"
git push origin milestone/rmist-m5
```

Record commit hash.

## 5. Merge to main

Proceed only if tests passed, branch pushed, working tree clean, and main has no unexpected divergent scientific work.

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge --no-ff milestone/rmist-m5 -m "merge: accept strict RMIST M5-R architecture"
git push origin main
```

If merge conflict occurs, STOP.

## 6. Verify

```bash
git status
git log --oneline --decorate --graph -12
git branch --merged main
```

Confirm M5 implementation and review-state commits are in main and M6-R remains locked.

## Final report

Return:

```text
M5 review-state cleanup: PASS/FAIL
review-state commit:
tests passed:

merge: PASS/FAIL
merge commit:
main push: PASS/FAIL

M5-R status: INDEPENDENTLY REVIEWED PASS
M6-R status: LOCKED
M7-R status: LOCKED

STOP.
```
