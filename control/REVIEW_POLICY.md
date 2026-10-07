# Review Policy

## Worker self-review

Before declaring a task complete:
1. run required tests;
2. save lightweight evidence;
3. update stream status;
4. commit and push;
5. mark task REVIEW.

## Supervisor review severities

- BLOCKER: scientific validity, data leakage, incorrect operator, invalid comparison, or evidence cannot support claimed conclusion.
- MAJOR: important defect with a clear repair path; worker should repair without PI interruption.
- MINOR: documentation, plotting, naming, small tests; worker repairs autonomously.

Only unresolved BLOCKERs and genuine route-choice P1 questions go to the PI.

## Negative results

A technically correct negative result is a valid completion. Record it under findings/rejected paths and continue to the next READY task unless it changes the research hypothesis.

## Merge policy

Worker branches are not merged automatically after each task. Stable checkpoints are merged after Supervisor review gates.
