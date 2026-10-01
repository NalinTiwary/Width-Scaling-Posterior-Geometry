#!/bin/bash
#SBATCH --job-name=bnn_paper_figures
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
# Two-figure paper package (docs/paper_figures/figure_production_guide.pdf), CPU only, no sampling:
#   1. export-loss-traces: the final h=0.01 V traces of all 24 targets (last 102,400 retained transitions per chain)
#      -> results/final_geometry/paper_figures/loss_traces/*.npz, input_map_loss.csv, loss_export_log.json
#   2. paper-figures: selfcheck, spectral + ACF tables, Figure 1 + Figure 2, integrity sheets, appendix table,
#      captions.tex, figure_results.md, recovery_log.md, input_map.csv, provenance.json
# Runs as `sbatch` or directly on a login node: `bash scripts/final_geometry/paper_figures.sbatch` (a few minutes).
# SKIP_EXPORT=1 rebuilds from already-exported NPZs.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/final_geometry/common.sh
mkdir -p logs/final_geometry
banner "paper figures"
rc=0
if [ -z "${SKIP_EXPORT:-}" ]; then
    bnn export-loss-traces "${BNN_ARGS[@]}" --device cpu || rc=1
fi
if [ $rc -eq 0 ]; then
    bnn paper-figures "${BNN_ARGS[@]}" || rc=1
else
    echo "FAIL: loss-trace export had failures (see paper_figures/loss_export_log.json); building Figure 1 only"
    bnn paper-figures "${BNN_ARGS[@]}" --steps selfcheck,spectral,render,text
fi
du -sh "$OUTPUT_ROOT/paper_figures" 2>/dev/null
echo "=== paper figures rc=$rc at $(date) ==="
exit $rc
