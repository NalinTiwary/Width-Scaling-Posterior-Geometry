#!/bin/bash
#SBATCH --job-name=bnn_paper_supplements
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/final_geometry/%x_%j.out
#SBATCH --error=logs/final_geometry/%x_%j.err
#
# Figure cleanup + supplements S1/S2 (docs/paper_cleanup), CPU only, no sampling:
#   1. export-supplements: test-probability traces of all 24 h=0.01 trajectories (S1, same window as Figure 2),
#      deep-endpoint V traces at h=0.01 and h=0.005 (S2, matched duration 512), and a fresh float64 SVD unit
#      check of T on archived reference W2 matrices -> paper_cleanup/supplement_traces (not committed), exports/
#   2. selfcheck, main, supplements, text: Figures 1-2 (6.75 in), S1/S2 figures and tables, diagnostics,
#      appendix table, recovery_log.json, captions.tex, figure_results.md
# Runs as `sbatch` or directly on a login node: `bash scripts/final_geometry/paper_supplements.sbatch`.
# SKIP_EXPORT=1 rebuilds from already-exported supplement traces.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/final_geometry/common.sh
mkdir -p logs/final_geometry
banner "paper cleanup + supplements"
rc=0
if [ -z "${SKIP_EXPORT:-}" ]; then
    bnn paper-cleanup "${BNN_ARGS[@]}" --device cpu --steps export-supplements || rc=1
fi
if [ $rc -eq 0 ]; then
    bnn paper-cleanup "${BNN_ARGS[@]}" --steps selfcheck,main,supplements,text || rc=1
else
    echo "FAIL: supplement export had failures (see paper_cleanup/supplement_export_log.json); main figures only"
    bnn paper-cleanup "${BNN_ARGS[@]}" --steps selfcheck,main,text
fi
du -sh "$OUTPUT_ROOT/paper_cleanup" "$OUTPUT_ROOT/paper_cleanup/supplement_traces" 2>/dev/null
echo "=== paper cleanup rc=$rc at $(date) ==="
exit $rc
