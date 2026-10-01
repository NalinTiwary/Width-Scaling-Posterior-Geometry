#!/bin/bash
#SBATCH --job-name=bnn_loss_export
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/final_geometry/%x_%j.out
#SBATCH --error=logs/final_geometry/%x_%j.err
#
# Paper figure 2 inputs: export the final h=0.01 training-loss (V) traces of all 24 targets, last 102,400 retained
# transitions per chain, to <output_root>/paper_figures/loss_traces/*.npz plus input_map_loss.csv and
# loss_export_log.json. Reads existing HDF5 traces only (no sampling); CPU work, a few minutes.
# Works as `sbatch` or directly on a login node: `bash scripts/final_geometry/export_loss_traces.sbatch`.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/final_geometry/common.sh
mkdir -p logs/final_geometry
banner "export loss traces"
export DEVICE=cpu
bnn export-loss-traces "${BNN_ARGS[@]}" --device cpu "$@"
rc=$?
du -sh "$OUTPUT_ROOT/paper_figures/loss_traces" 2>/dev/null
echo "=== export rc=$rc at $(date) ==="
exit $rc
