# AI Research Control Plane v2

This directory turns the repository into an asynchronous research control plane.

## Roles

- Human PI: chooses goals, answers high-level scientific questions, approves major pivots.
- ChatGPT Supervisor: decomposes goals, reviews branch results, accumulates blockers, prepares daily briefs.
- Worker A: learned sensing + reconstruction paper stream.
- Worker B: baseline/library stream.
- GitHub: asynchronous message bus and audit trail.

## Operating rule

Workers do not stop after every task. They repeatedly consume the next READY task from their own queue, execute it, self-test, commit, push, update their stream status, then continue.

Workers stop only for:
- P0 scientific ambiguity;
- an explicit STOP gate;
- inability to satisfy acceptance criteria after reasonable attempts;
- risk of altering accepted physics/data conventions.

Routine coding, CUDA, plotting, path, dtype, and refactoring problems are handled autonomously.

## Human cadence

The PI reviews once per day. Questions are accumulated in `control/QUESTIONS.md` and nightly briefings rather than interrupting daytime execution.
