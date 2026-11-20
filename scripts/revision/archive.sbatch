#!/bin/bash
#SBATCH --job-name=bnn_rev_archive
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/revision/archive_%j.out
#SBATCH --error=logs/revision/archive_%j.err
#
# No sampling. Rebuilds the Section 2 table, input spectra, spectral counts and bounds.csv
# from the frozen campaign and final_geometry_package.zip. This also runs on a laptop:
#   python -m bnn_geometry.revision archive --config configs/revision.yaml

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
echo "=== archive at $(date) on $(hostname) ==="
python3 -m bnn_geometry.revision archive --config "$CONFIG"
exit $?
