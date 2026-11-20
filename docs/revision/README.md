# AISTATS revision runs

The frozen campaign in `configs/campaign.yaml` is unchanged. New outputs go to `results/revision/`.

Local, no sampling:

```bash
python -m bnn_geometry.revision archive --config configs/revision.yaml
```

This writes `efficiency.csv` (archived bulk ESS; cost still unmeasured), `bounds.csv`, `status.csv`, `initialization.csv`, the Section 2 medians, per-replicate break-even ratios, spectral counts and the three input spectra.

Local smoke, including a prior-only OU check and a 24-step chain:

```bash
python -m bnn_geometry.revision smoke --config configs/revision_smoke.yaml
```

A short timing forecast on the real deep width-32 replicate, still on this machine:

```bash
python -m bnn_geometry.revision run --config configs/revision.yaml --stage local-pilot --task-id 0 --device cpu
```

Cluster, from the repository root. The account, partition, mail and GPU request match `scripts/final_geometry/`:

```bash
mkdir -p logs/revision
bash scripts/revision/submit.sh                         # 24 fixed-step timed runs
STAGE=tune bash scripts/revision/submit.sh              # 12 endpoint step pilots
STAGE=prior-starts bash scripts/revision/submit.sh      # fresh prior starts
STAGE=prior-only bash scripts/revision/submit.sh        # deep endpoints, likelihood drift off
STAGE=geometry-pilot bash scripts/revision/submit.sh    # six shallow endpoint cells, replicate 0
STAGE=relu bash scripts/revision/submit.sh              # then depth4, prior-scale, concentrated, nonlinear, sin
```

`STAGE=geometry` is the full 36-cell block and should wait on the six-cell pilot. Each task resumes from its checkpoint, so a 24 hour timeout is continued by resubmitting the same command.
