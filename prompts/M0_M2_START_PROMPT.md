Execute the currently authorized work for this repository autonomously.

Before taking any action, read and treat the following repository files as the source of truth:

- AGENTS.md
- RESEARCH_SPEC.md
- PLAN.md
- STATUS.md
- DECISIONS.md
- EXPERIMENTS.md

Your authorized scope is strictly:

M0 — Repository Initialization
M1 — Hammer Dataset Audit
M2 — Conventional single-Rx RMA Baseline

Do not start M3 or any later milestone.

Important fixed constraints:

1. The real dataset is located at:
   /home/superws/dataset/3DRIED/Hammer/Hammer (1)

2. Use only ONE Rx channel.
   Do not average, fuse, concatenate, jointly calibrate, or otherwise combine multiple Rx channels.

3. Raw dataset files are read-only.
   Never modify, rename, move, or overwrite them.

4. Do not introduce Gamma-Net, Toe-LISTA, CNN post-processing, learned forward operators, sparse-aperture experiments, or any other learning model during M0–M2.

5. The purpose of this run is to establish a trustworthy conventional imaging baseline, not to produce research novelty.

Execution policy:

- Work milestone by milestone.
- Before each milestone, read its acceptance criteria in PLAN.md.
- After implementation, run the required validations.
- If validation fails because of an ordinary coding, dependency, path, dtype, tensor-shape, CUDA, numerical, or plotting problem, diagnose it, fix it, rerun validation, and continue without asking me.
- Do not treat "the program runs" as PASS.
- Do not proceed to the next milestone until the current milestone satisfies its acceptance criteria.
- Record nontrivial scientific or engineering decisions in DECISIONS.md.
- Record experiments and commands in EXPERIMENTS.md.
- Continuously update STATUS.md so that a human can understand the exact current state without reading the chat transcript.
- Save reproducible configs, logs, environment information, intermediate shapes, and final outputs.
- Keep changes scoped to this repository.
- Commit stable milestone results to Git.

For M1:
Perform a read-only audit of the Hammer dataset and determine the actual file structure, data shape, dtype, axis meanings, Rx layout, available metadata, and any available reference processing material. Create scripts/inspect_hammer.py and docs/HAMMER_DATA_AUDIT.md.

For M2:
Implement the simplest scientifically justified conventional RMA / near-field holographic reconstruction consistent with the actual 3DRIED data and system parameters discovered in M1.

Prefer existing dataset documentation, reference implementation, accompanying files, or the documented 3DRIED processing procedure over inventing assumptions.

Start with a minimal diagnostic RMA run before a full Hammer reconstruction.

The final M2 output must include at least:

results/rma_hammer/
    volume.npy
    mip_xz.png
    representative_slices/
    volume_3d.png
    run.log
    config_used.yaml
    environment.txt
    metrics_or_stats.json

and:

docs/M2_RMA_HAMMER_REPORT.md

The M2 report must clearly state:

- selected Rx;
- raw data shape;
- preprocessing pipeline;
- reconstructed range interval;
- coordinate convention;
- imaging parameters;
- physical x/y/z ranges;
- whether Hammer is recognizable;
- whether its position and orientation are physically plausible;
- comparison with any available 3DRIED reference visualization;
- remaining uncertainties.

Scientific stop conditions:

Stop and request human review only if one of the explicit stop conditions in AGENTS.md or PLAN.md is reached.

Do not stop for routine implementation failures.

If information is ambiguous, first investigate the dataset, repository, accompanying metadata, documentation, and reference code. Do not silently invent radar parameters.

When M2 satisfies its acceptance criteria:

1. update STATUS.md;
2. update EXPERIMENTS.md;
3. update DECISIONS.md if needed;
4. commit the completed M0–M2 work;
5. provide a concise final milestone summary with:
   - PASS/FAIL status for M0, M1, M2;
   - key commands used;
   - exact output paths;
   - important assumptions;
   - unresolved issues;
   - the Git commit hash;
6. STOP.

Do not begin M3.