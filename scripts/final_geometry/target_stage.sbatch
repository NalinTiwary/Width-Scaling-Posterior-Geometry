#!/bin/bash
#SBATCH --job-name=bnn_target
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
#SBATCH --output=logs/final_geometry/%x_%A_%a.out
#SBATCH --error=logs/final_geometry/%x_%A_%a.err
#
# One target per array task (task id = index into the campaign's target list, see `plan`).
# Env: STAGE = reference | production | refine (required), CONFIG, OUT, DEVICE.
# Every stage resumes from its atomic checkpoints, so rerunning a task (or a continuation array
# submitted by submit.sh) continues where a time limit stopped it; finished tasks exit at once.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/final_geometry/common.sh

: "${STAGE:?set STAGE=reference|production|refine}"
TASK="${SLURM_ARRAY_TASK_ID:?must run as an array task}"
TID="$(python3 -c "
import sys; sys.path.insert(0, 'src')
from bnn_geometry import config as C
print([t.target_id for t in C.targets(C.load('$CONFIG'))][$TASK])")"

if [ -f "$OUTPUT_ROOT/targets/$TID/stages/$STAGE.done" ]; then
    echo "$TID: stage $STAGE already complete ($(cat "$OUTPUT_ROOT/targets/$TID/stages/$STAGE.done")); nothing to do"
    exit 0
fi
banner "stage $STAGE target $TID"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "${ALLOW_CPU:-0}" != "1" ] && ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "FAIL: CUDA not visible to torch on $(hostname); refusing to run on CPU (ALLOW_CPU=1 overrides)"
    exit 1
fi

bnn run --resume "${BNN_ARGS[@]}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} --stage "$STAGE" --target "$TID"
rc=$?
echo "=== $TID stage $STAGE exit=$rc elapsed=${SECONDS}s ==="
[ -f "$OUTPUT_ROOT/targets/$TID/status.json" ] && cat "$OUTPUT_ROOT/targets/$TID/status.json" | head -c 4000
exit $rc
