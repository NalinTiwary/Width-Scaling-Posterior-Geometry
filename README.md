# Cylinder BNN posterior experiments

Frozen protocol from the design document: two-layer tanh BNN, Gaussian prior,
elliptical slice sampling, theorem cylinder coverage and curvature deficit brackets.

## Setup

CPU:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# or: pip install -r requirements-cpu.txt && pip install -e .
```

CUDA (pick a wheel matching the cluster driver):

```bash
pip install -r requirements-cuda.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Device selection is automatic (`device: auto` in `config.yaml`): CUDA if available, else CPU.
Float64 is required; Apple MPS is not used (no float64 support). Override with `--device cuda|cpu`.

## Run order (design §11.1)

```bash
# 0. Numerical / theorem preflight
python scripts/00_preflight.py --config config.yaml

# 1. Materialize frozen data + centers
python scripts/00_preflight.py --config config.yaml --write-artifacts

# 2. Central target first
python scripts/01_run_target.py --config config.yaml --m 1024 --seed 0
python scripts/01_run_target.py --config config.yaml --m 1024 --seed 0 --kernel pcn

# 3. Remaining grid (or use the helper)
bash scripts/run_paper_grid.sh

# 4. Diagnostics / extension
python scripts/02_extend_if_needed.py --config config.yaml

# 5. Curvature on 128 states / target
python scripts/03_curvature_states.py --config config.yaml

# 6–7. Summaries and figures
python scripts/04_summarize.py --config config.yaml
python scripts/05_make_figures.py --config config.yaml
```

Parallel remote chains (optional):

```bash
python scripts/01_run_target.py --config config.yaml --m 1024 --seed 0 --chain_id 0 --device cuda
# ... chains 1–3, then merge is automatic if you run without --chain_id on a
# completed set, or re-run the full target once all chain shards exist.
```

## Artifacts

See `artifacts/` after runs: `data.npz`, `centers_seed*.npz`, `runs/*/`,
`curvature.csv`, `target_summary.csv`, `figure*.csv`, `figures/`.

## Smoke / figure check (short chains)

```bash
python scripts/run_smoke_grid.py --config config.smoke.yaml --device cpu
pytest -q
# figures in artifacts_smoke/figures/
```
