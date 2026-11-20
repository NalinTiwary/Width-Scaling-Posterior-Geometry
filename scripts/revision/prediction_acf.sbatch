#!/bin/bash
#SBATCH --job-name=bnn_pred_acf
#SBATCH --time=01:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/revision/prediction_acf_%j.out
#SBATCH --error=logs/revision/prediction_acf_%j.err
#
# Rebuild the deep_m0032_r0 test_prob_0 autocorrelation from the raw stage files.
# If those HDF5 trajectories are not on this checkout, the job records that and exits.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/revision/common.sh
mkdir -p logs/revision
echo "=== prediction acf at $(date) on $(hostname) ==="
python3 -m bnn_geometry.revision prediction-acf --config "$CONFIG"
exit $?
