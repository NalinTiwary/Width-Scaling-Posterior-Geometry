#!/bin/bash
#SBATCH --job-name=bnn_global
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
#SBATCH --output=logs/final_geometry/%x_%j.out
#SBATCH --error=logs/final_geometry/%x_%j.err
#
# Campaign-wide steps between the target arrays. Env STEPS is a space-separated list of:
#   freeze            validate-config, plan, freeze (data, centers, manifest, full pytest), benchmark
#   select-step       architecture-wide pCNL step from all targets' calibration   (job B)
#   controls          iid Gaussian-prior spectral reference                         (job B)
#   endpoint-decision round-1 step-halving decision                                 (job D)
#   refine-decision   round-2 decision after the one allowed refinement             (job F)
#   post              analyze, figures (+SUMMARY.md), audit, export                 (job F)
# Also: CONFIG, OUT, DEVICE.

source "$(dirname "$0")/common.sh" 2>/dev/null || source scripts/final_geometry/common.sh
: "${STEPS:?set STEPS}"
banner "global steps: $STEPS"

rc=0
for step in $STEPS; do
    t0=$SECONDS
    case "$step" in
        freeze)
            bnn validate-config "${BNN_ARGS[@]}" || exit 1
            bnn plan "${BNN_ARGS[@]}"
            bnn freeze "${BNN_ARGS[@]}" || { echo "FAIL: freeze or tests failed; not starting the campaign"; exit 1; }
            bnn benchmark "${BNN_ARGS[@]}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} || rc=1
            ;;
        select-step|controls|endpoint-decision|refine-decision)
            bnn run --resume "${BNN_ARGS[@]}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} --stage "$step" || rc=1
            ;;
        post)
            bnn analyze "${BNN_ARGS[@]}" || rc=1
            bnn figures "${BNN_ARGS[@]}" || rc=1
            bnn audit "${BNN_ARGS[@]}" --strict || rc=1
            bnn export "${BNN_ARGS[@]}" || rc=1
            [ -f "$OUTPUT_ROOT/SUMMARY.md" ] && cat "$OUTPUT_ROOT/SUMMARY.md"
            ;;
        *)
            echo "unknown step $step"; rc=1 ;;
    esac
    echo "=== step $step finished (rc so far $rc) in $((SECONDS - t0))s ==="
done
exit $rc
