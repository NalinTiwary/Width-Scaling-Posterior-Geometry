#!/usr/bin/env bash
# Ordered paper grid (§11.1). Run from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CFG="${1:-config.yaml}"
DEVICE_ARG=()
if [[ "${DEVICE:-}" != "" ]]; then
  DEVICE_ARG=(--device "$DEVICE")
fi

python scripts/00_preflight.py --config "$CFG" --write-data --skip-tests "${DEVICE_ARG[@]}"
python -m pytest -q

echo "=== Central target m=1024 seed=0 (ESS) ==="
python scripts/01_run_target.py --config "$CFG" --m 1024 --seed 0 --kernel ess "${DEVICE_ARG[@]}"

echo "=== pCN cross-check ==="
python scripts/01_run_target.py --config "$CFG" --m 1024 --seed 0 --kernel pcn "${DEVICE_ARG[@]}"

echo "=== Remaining ESS targets ==="
for m in 256 1024 4096; do
  for seed in 0 1 2; do
    if [[ "$m" == "1024" && "$seed" == "0" ]]; then
      continue
    fi
    echo "--- m=$m seed=$seed ---"
    python scripts/01_run_target.py --config "$CFG" --m "$m" --seed "$seed" --kernel ess "${DEVICE_ARG[@]}"
  done
done

python scripts/02_extend_if_needed.py --config "$CFG" --extend "${DEVICE_ARG[@]}"
python scripts/03_curvature_states.py --config "$CFG"
python scripts/04_summarize.py --config "$CFG"
python scripts/05_make_figures.py --config "$CFG"

echo "Paper grid complete. Artifacts in artifacts/"
