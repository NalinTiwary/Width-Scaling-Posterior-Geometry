#!/bin/bash
#SBATCH --job-name=bnn_next
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
cd "${SLURM_SUBMIT_DIR:?submit from the project root}"
source scripts/revision/common.sh
set -euo pipefail
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
"$PY" -c 'import h5py,yaml,numpy,pandas,scipy,torch,bnn_geometry' || {
  echo "IMPORT FAILURE: $PY cannot import the project packages"
  exit 1
}
"$PY" a10_next/cluster.py "${JOB:?specify JOB}" --scope "${SCOPE:-endpoint}" \
  --task "${SLURM_ARRAY_TASK_ID:?}" --level "${LEVEL:-1}" --config "$CONFIG" \
  --out "${NEXT_OUT:-results/revision_next}" --device cuda
