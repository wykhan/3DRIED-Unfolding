# RMIST_CONTROL_FILE_PATCHES.md

This file resets repository governance from the discarded generic M3/M4 design to strict RMIST reproduction. Do not erase M0-M2 history.

# 1. AGENTS.md

Replace the current authorization section with:

```markdown
## Current authorization scope

M0-M2 are complete and human-reviewed.

The previous generic `milestone/m3-m4` implementation was independently reviewed and is not the accepted physics basis for learning.

The currently authorized scope is:

- M3-R: strict RMIST RM / RM_dagger kernel reproduction
- M4-R: strict RM-kernel ISTA reproduction

Algorithmic source of truth:

1. RMIST-Net paper;
2. `docs/RMIST_REPRO_SPEC.md`;
3. human-approved 3DRIED M0-M2 geometry and preprocessing.

After M4-R, STOP and request independent review.

M5-R and later work are locked.

Codex must not replace RMIST's `RM` / `RM_dagger` inverse-kernel structure with a Hilbert-adjoint pair, FISTA, LISTA, Gamma-Net, Toe-LISTA, CNN proximal blocks, or another architecture.
```

Add to "Actions requiring human approval":

```markdown
- changing the RMIST `1/ky` forward and `ky` inverse-kernel structure;
- changing the RMIST 9-phase network structure;
- changing RMIST-reported training hyperparameters;
- substituting a mathematically different operator because it has better adjoint properties;
- treating an unspecified paper parameter as if it were reported.
```

# 2. RESEARCH_SPEC.md

Replace the learning/operator sections from M3 onward with:

```markdown
## RMIST reproduction basis

All learning-related work from M3-R onward follows `docs/RMIST_REPRO_SPEC.md`.

The strict baseline is RMIST, not the previous generic ISTA-unfolding design:

- RM forward kernel with `1/ky`;
- RM_dagger inverse kernel with `ky`;
- Eq. (19) RM-kernel ISTA;
- Eq. (20) phase-wise `tau_t`, `theta_t`;
- 9 phases;
- synthetic RM-generated training echoes;
- 1000 training pairs;
- default 20 dB AWGN;
- Adam, lr=1e-4;
- tau init=1e-4;
- theta init=1e-3;
- 100 epochs;
- RMIST reconstruction loss.

For 3DRIED, the propagation sign is mapped so `RM_dagger` reproduces the validated M2 RMA. This is a system-coordinate adaptation, not an algorithm change.
```

Remove any statement that the accepted learning forward operator is the Hilbert adjoint of M2 RMA.

# 3. PLAN.md

Set project state to:

```text
COMPLETED + HUMAN APPROVED: M0, M1, M2
HISTORICAL / NOT MERGED: previous generic M3-M4 branch
UNLOCKED: M3-R, M4-R
LOCKED: M5-R, M6-R, M7-R
```

Define milestones:

```markdown
M3-R — Strict RM / RM_dagger implementation
M4-R — RM-kernel ISTA Eq. (16)-(19)
EXTERNAL REVIEW
M5-R — RMIST-Net Eq. (20)-(26)
M6-R — synthetic training Eq. (27)-(31)
EXTERNAL REVIEW
M7-R — real Hammer slice-wise inference
```

# 4. STATUS.md

At the top, set:

```markdown
M0 PASS — human reviewed
M1 PASS — human reviewed
M2 PASS — human reviewed

Previous generic M3/M4:
NOT ACCEPTED AS LEARNING PHYSICS BASIS
Preserved on `milestone/m3-m4`
Not merged to `main`

Current branch:
`milestone/rmist-m3-m4`

Current authorized execution:
M3-R → M4-R → EXTERNAL REVIEW

M5-R and later milestones are locked.
```

Keep historical M0-M2 entries below.

# 5. DECISIONS.md

Append:

```markdown
## D017 — RMIST paper becomes algorithmic source of truth

**Status:** FIXED

From M3-R onward, the baseline is a method-level reproduction of RMIST-Net.
Do not substitute generic ISTA unfolding.

## D018 — RMIST RM / RM_dagger semantics

**Status:** FIXED

Preserve RMIST reciprocal amplitude kernels:

- forward `RM`: `1/ky`;
- inverse `RM_dagger`: `ky`.

`RM_dagger` is the inverse RM kernel used by RMIST and must not be replaced by a Hilbert adjoint.

## D019 — 3DRIED sign adaptation

**Status:** FIXED — SYSTEM ADAPTATION

M2 validates inverse phase `ky * exp(-j*r*ky)`.
Therefore the mapped RMIST pair is:

- `Phi_dagger = ky * exp(-j*r*ky)`
- `Phi = (1/ky) * exp(+j*r*ky)`

## D020 — RMIST training hyperparameters

**Status:** FIXED

Strict baseline:

- 9 phases;
- Adam;
- lr = 1e-4;
- tau_t init = 1e-4;
- theta_t init = 1e-3;
- 100 epochs;
- 1000 synthetic training pairs;
- default 20 dB complex AWGN.

Batch size is paper-unspecified and must be labeled as an implementation choice.

## D021 — Previous generic M3/M4 branch

**Status:** HISTORICAL / NOT ACCEPTED

`milestone/m3-m4` remains as experimental history but is not the accepted learning-physics baseline and must not be merged into `main`.
```

# 6. EXPERIMENTS.md

Keep old entries. Append:

```markdown
# RMIST reproduction series

The experiments below belong to the strict RMIST reproduction baseline.
Earlier generic M3/M4 experiments remain historical and are not acceptance evidence for M3-R/M4-R.
```

Use IDs such as:

```text
EXP-RMIST-M3-001
EXP-RMIST-M4-001
```

# 7. REVIEW_CHECKLIST.md

Replace it with the RMIST-specific `REVIEW_CHECKLIST.md` from this reset bundle.

# 8. New files

Add:

```text
docs/RMIST_REPRO_SPEC.md
prompts/RMIST_M3_M4_START_PROMPT.md
RMIST_CONTROL_FILE_PATCHES.md
```

After applying patches, commit governance before scientific code:

```bash
git add AGENTS.md RESEARCH_SPEC.md PLAN.md STATUS.md DECISIONS.md EXPERIMENTS.md REVIEW_CHECKLIST.md docs/RMIST_REPRO_SPEC.md prompts/RMIST_M3_M4_START_PROMPT.md RMIST_CONTROL_FILE_PATCHES.md
git commit -m "chore: reset baseline to strict RMIST reproduction"
git push origin milestone/rmist-m3-m4
```
