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

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "$(dirname "$0")/common.sh" ]; then
    source "$(dirname "$0")/common.sh"
elif [ -f scripts/revision/common.sh ]; then
    source scripts/revision/common.sh
else
    echo "FAIL: scripts/revision/common.sh not found from $(pwd) (script $0)" >&2
    exit 1
fi
mkdir -p logs/revision
echo "=== prediction acf at $(date) on $(hostname) python=$(command -v python3) ==="
# Reading the archived chains does not need the allocated GPU.
python3 -m bnn_geometry.revision prediction-acf --config "${CONFIG:-configs/revision.yaml}" --device cpu
status=$?
echo "=== prediction exit=$status ==="
if [ -f results/revision/prediction_acf_check.json ]; then
    cat results/revision/prediction_acf_check.json
fi
exit $status
