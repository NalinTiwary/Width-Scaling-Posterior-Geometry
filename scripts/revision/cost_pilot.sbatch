#!/bin/bash
#SBATCH --job-name=bnn_cost_pilot
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --requeue
#SBATCH --output=logs/revision/%x_%j.out
#SBATCH --error=logs/revision/%x_%j.err
#
# One GPU. Deep widths 32 and 256, replicate 0, alternating.
# Four chains, scalar logging inside each step. Each width keeps its own block length,
# synchronized elapsed time, and transition count. Blocks last at least one second.
# ess_per_second_ratio = 2.068404 / (wide seconds per transition / narrow seconds per transition).
# A value above one favors width 256. A block-rate range that contains 2.068404 is inconclusive.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/revision/common.sh

mkdir -p logs/revision
echo "=== cost pilot at $(date) ==="
echo "=== host $(hostname) job ${SLURM_JOB_ID:-none} ==="
echo "=== git $(git rev-parse --short HEAD 2>/dev/null || echo n/a) ==="
echo "=== config $CONFIG device $DEVICE ==="
python3 -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${ALLOW_CPU:-0}" != "1" ] && ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "FAIL: CUDA is not visible on $(hostname). Refusing the cost pilot (ALLOW_CPU=1 overrides)."
    exit 1
fi

python3 -m bnn_geometry.revision cost-pilot --config "$CONFIG" --device "$DEVICE"
rc=$?
echo "=== cost pilot exit=$rc elapsed=${SECONDS}s ==="
exit $rc
