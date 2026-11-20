#!/bin/bash
#SBATCH --job-name=bnn_rev_archive
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/revision/archive_%j.out
#SBATCH --error=logs/revision/archive_%j.err
#
# No sampling. Rebuilds the Section 2 table, input spectra, spectral counts and bounds.csv
# from the frozen campaign and final_geometry_package.zip. This also runs on a laptop:
#   python -m bnn_geometry.revision archive --config configs/revision.yaml

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/revision/common.sh
mkdir -p logs/revision
echo "=== archive at $(date) on $(hostname) ==="
python3 -m bnn_geometry.revision archive --config "$CONFIG"
exit $?
