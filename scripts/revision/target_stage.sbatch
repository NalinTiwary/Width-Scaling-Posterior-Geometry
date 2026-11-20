#!/bin/bash
#SBATCH --job-name=bnn_revision
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --array=0-23
#SBATCH --requeue
#SBATCH --output=logs/revision/%x_%A_%a.out
#SBATCH --error=logs/revision/%x_%A_%a.err
#
# One revision cell per array task. STAGE is fixed-step, tune, prior-starts, prior-only,
# geometry-pilot, geometry, relu, depth4, prior-scale, concentrated, nonlinear, or sin.
# Finished checkpoints are reused, so a timed-out task can be resubmitted.
# Submit with scripts/revision/submit.sh, which overrides the array range.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/revision/common.sh

: "${STAGE:?set STAGE to a revision stage}"
TASK="${SLURM_ARRAY_TASK_ID:?must run as an array task}"

mkdir -p logs/revision
echo "=== revision $STAGE task $TASK at $(date) ==="
echo "=== host $(hostname) job ${SLURM_JOB_ID:-none} array ${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} ==="
echo "=== git $(git rev-parse --short HEAD 2>/dev/null || echo n/a) ==="
echo "=== config $CONFIG device $DEVICE ==="
python3 -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${ALLOW_CPU:-0}" != "1" ] && ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "FAIL: CUDA is not visible on $(hostname). Refusing a full-width CPU run (ALLOW_CPU=1 overrides)."
    exit 1
fi

python3 -m bnn_geometry.revision run --config "$CONFIG" --stage "$STAGE" --task-id "$TASK" --device "$DEVICE"
rc=$?
echo "=== task $TASK stage $STAGE exit=$rc elapsed=${SECONDS}s ==="
exit $rc
