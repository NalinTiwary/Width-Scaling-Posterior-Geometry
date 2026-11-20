# Sourced by the revision SLURM scripts. Same project, venv and account conventions as final_geometry.
set -uo pipefail

if [ -n "${BNN_DIR:-}" ]; then
    PROJ_DIR="$BNN_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJ_DIR" || { echo "ERROR: cannot cd to $PROJ_DIR"; exit 1; }

if [ -n "${BNN_VENV:-}" ]; then
    source "$BNN_VENV/bin/activate"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$PROJ_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HDF5_USE_FILE_LOCKING=FALSE

CONFIG="${CONFIG:-configs/revision.yaml}"
DEVICE="${DEVICE:-cuda}"
