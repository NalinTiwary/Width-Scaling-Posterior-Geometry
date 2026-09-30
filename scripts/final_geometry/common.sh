# Sourced by the final-geometry SLURM scripts. Sets up the project dir, venv and helpers.
set -uo pipefail

if [ -n "${BNN_DIR:-}" ]; then
    PROJ_DIR="$BNN_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJ_DIR" || { echo "ERROR: cannot cd to $PROJ_DIR"; exit 1; }

if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$PROJ_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export HDF5_USE_FILE_LOCKING=FALSE

CONFIG="${CONFIG:-configs/campaign.yaml}"
BNN_ARGS=(--config "$CONFIG")
[ -n "${OUT:-}" ] && BNN_ARGS+=(--out "$OUT")
DEV_ARGS=()
[ -n "${DEVICE:-}" ] && DEV_ARGS+=(--device "$DEVICE")

OUTPUT_ROOT="$(python3 - "$CONFIG" "${OUT:-}" <<'EOF'
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(open(sys.argv[1]))
out = sys.argv[2] or cfg["output_root"]
p = Path(out)
print(p if p.is_absolute() else Path.cwd() / p)
EOF
)"

bnn() {
    echo "[$(date +%H:%M:%S)] python -m bnn_geometry $*"
    python3 -m bnn_geometry "$@"
}

banner() {
    echo "=== $1 at $(date) ==="
    echo "=== host $(hostname)  job ${SLURM_JOB_ID:-none}  array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} ==="
    echo "=== git $(git rev-parse --short HEAD 2>/dev/null || echo n/a) $(git diff --quiet 2>/dev/null || echo '(dirty)') ==="
    echo "=== config $CONFIG  output $OUTPUT_ROOT ==="
    python3 -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true
    echo "=== disk free: $(df -h "$PROJ_DIR" | tail -1) ==="
}
