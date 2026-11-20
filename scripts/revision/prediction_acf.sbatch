#!/bin/bash
#SBATCH --job-name=bnn_pred_acf
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/revision/prediction_acf_%j.out
#SBATCH --error=logs/revision/prediction_acf_%j.err
#
# Rebuild the deep_m0032_r0 test_prob_0 autocorrelation from the raw stage files.
# If those HDF5 trajectories are not on this checkout, the job records that and exits.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/revision/common.sh
mkdir -p logs/revision
echo "=== prediction acf at $(date) on $(hostname) device $DEVICE ==="
python3 -c "import sys, torch; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" || true
if [ "${ALLOW_CPU:-0}" != "1" ] && ! python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "FAIL: CUDA is not visible on $(hostname). Refusing the prediction rebuild (ALLOW_CPU=1 overrides)."
    exit 1
fi
python3 -m bnn_geometry.revision prediction-acf --config "$CONFIG" --device "$DEVICE"
exit $?
