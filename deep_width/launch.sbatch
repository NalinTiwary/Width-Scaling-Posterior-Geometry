#!/bin/bash
#SBATCH --job-name=bnn_deep_width
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/revision/%x_%A_%a.out
#SBATCH --error=logs/revision/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/revision/common.sh
set -euo pipefail
# The package lives at the repository root. src/ is only the bnn_geometry tree.
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
"$PY" -c 'import h5py,yaml,numpy,pandas,scipy,torch,bnn_geometry,deep_width,deep_width.run' || {
  echo "IMPORT FAILURE: $PY cannot import the project or deep_width"
  exit 1
}
"$PY" -m deep_width.run "${MODE:-pilot}" --task "${SLURM_ARRAY_TASK_ID:?}" \
    --config "${CONFIG:-configs/revision.yaml}" --device cuda \
    --kernel "${KERNEL:-pcnl}" --step "${STEP:-0.16}" --level "${LEVEL:-1}" \
    --pilot-steps "${PILOT_STEPS:-128}" --out "${DEEP_OUT:-results/deep_width}"
