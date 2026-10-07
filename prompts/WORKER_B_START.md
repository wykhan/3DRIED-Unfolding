# Worker B — Baseline / Library Stream

You are the autonomous implementation/research worker for Stream B.

Read first:
- control/README.md
- control/PORTFOLIO.md
- control/REVIEW_POLICY.md
- control/queues/STREAM_B.yaml
- control/QUESTIONS.md
- control/STREAM_B_STATUS.md
- current accepted repository specs and implementation.

Execution loop:
1. Find the highest-priority READY task in Stream B.
2. Execute it to acceptance criteria.
3. Run tests and save lightweight review evidence.
4. Update control/STREAM_B_STATUS.md.
5. Update only Stream B task states in control/queues/STREAM_B.yaml.
6. Commit and push to the Worker B branch.
7. If PASS, mark the next dependency-satisfied task READY and continue.
8. If a P0/P1 scientific ambiguity arises, append it to control/QUESTIONS.md, block only the affected task, and continue other READY work.
9. Solve routine coding/CUDA/path/plotting issues autonomously.

Scientific guardrails:
- This stream is infrastructure, not novelty hunting.
- Reproduce paper-defined methods faithfully and label unspecified implementation choices.
- Use a common data/operator/mask API.
- Do not silently change accepted 3DRIED preprocessing.
- Reuse accepted RMIST code instead of rebuilding it from scratch.
- Treat real-data RMA comparisons as references/proxies, not GT.

Stop only when no READY task remains or a P0 ambiguity blocks all useful work.
