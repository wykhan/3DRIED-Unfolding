# M7_REVIEW_MERGE_PROMPT.md

# Codex task: record accepted M7-R review and merge to main

Independent review result:

```text
M7-R technical pipeline: PASS
Real-data transfer: NON-CATASTROPHIC / HAMMER RECOGNIZABLE
Learned improvement over sparse RMA/Ini-ISTA: NOT DEMONSTRATED
Overall milestone verdict: PASS WITH IMPORTANT SCIENTIFIC LIMITATION
Blocking implementation issues: NONE
```

This task is repository governance only.

Do NOT retrain.
Do NOT change M6 parameters.
Do NOT start M8 diagnostics.
Do NOT create `milestone/rmist-m8`.
Do NOT modify M7 scientific outputs.
Do NOT force-push or delete branches.

## 1. Verify state

Expected branch:

```text
milestone/rmist-m7
```

Run:

```bash
git status
git branch --show-current
git fetch origin
git log --oneline --decorate --graph -15
git merge-base --is-ancestor origin/main HEAD
```

Confirm implementation commit:

```text
b7bab03
```

is present and the working tree is clean.

If unexpected, STOP.

## 2. Record independent M7 review

Update:

- `STATUS.md`
- `PLAN.md`
- `AGENTS.md`
- `REVIEW_CHECKLIST.md`
- `docs/M7_REVIEW_CHECKLIST.md`
- `docs/RMIST_REPRO_SPEC.md`
- `DECISIONS.md`

Use precise wording:

```text
M7-R: INDEPENDENTLY REVIEWED — TECHNICAL PASS

Real-data finding:
Hammer remains recognizable and range/orientation are preserved.

Scientific limitation:
trained RMIST50 is numerically almost identical to RMA50/Ini-ISTA50.
Learned improvement on real Hammer is NOT DEMONSTRATED.
```

Do not convert this into a superiority claim.

Append a decision equivalent to:

```markdown
## D031 — M7-R independent review accepted

**Status:** FIXED

- Eq. (32) real Hammer inference pipeline is technically accepted.
- The Hammer remains recognizable under the accepted 50% sparse condition.
- RMIST50 differs from RMA50 by only about 1.1364e-6 in complex relative norm.
- Observed-sample consistency is essentially identical across RMA50, Ini-ISTA50, and RMIST50.
- Therefore real-data learned improvement is not demonstrated.
- The next authorized scientific question is diagnostic: determine why synthetic learned gain collapses to an almost identity/RMA mapping on real data.
```

## 3. Regression tests

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

If tests fail unexpectedly, STOP.

## 4. Commit review state

Commit only governance/review-state updates:

```bash
git add AGENTS.md PLAN.md STATUS.md DECISIONS.md REVIEW_CHECKLIST.md \
  docs/M7_REVIEW_CHECKLIST.md docs/RMIST_REPRO_SPEC.md

git commit -m "chore: record independent M7-R acceptance"
git push origin milestone/rmist-m7
```

Record the commit hash.

## 5. Merge to main

Proceed only if tests pass and origin/main has no unexpected divergent work:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git merge --no-ff milestone/rmist-m7 -m "merge: accept strict RMIST M7-R real inference"
git push origin main
```

If any conflict occurs, STOP.

## 6. Verify

Confirm:

- `b7bab03` is contained in main;
- M7 review-state commit is contained in main;
- main contains M0-M7 accepted baseline;
- no M8 code exists yet.

Return:

```text
M7 review-state cleanup: PASS/FAIL
review-state commit:
tests passed:

merge: PASS/FAIL
merge commit:
main push: PASS/FAIL

M7-R technical status: INDEPENDENTLY REVIEWED PASS
real-data learned improvement: NOT DEMONSTRATED
next scientific milestone: M8 SIM-TO-REAL DIAGNOSTIC, still locked

STOP.
```
