# 3DRIED Unfolding

Reproducible physics-based imaging baselines for the 3DRIED dataset. The
currently authorized scope is the single-Rx Hammer audit and conventional RMA
reconstruction (milestones M0--M2). Later learning milestones remain locked.

```bash
python scripts/smoke_test.py
python scripts/inspect_hammer.py --output results/hammer_audit.json
python scripts/run_rma.py --config configs/hammer.yaml
```

The raw dataset is external and is always opened read-only.

