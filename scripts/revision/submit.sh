#!/bin/bash
#
# Submit revision stages. Sampling jobs match the final-geometry account, partition and mail.
#
#   bash scripts/revision/submit.sh                  # fixed-step timing, 24 targets
#   STAGE=tune bash scripts/revision/submit.sh       # 12 endpoint pilots
#   STAGE=prior-starts bash scripts/revision/submit.sh
#   STAGE=geometry-pilot bash scripts/revision/submit.sh
#   STAGE=archive bash scripts/revision/submit.sh    # no sampling; also fine locally
#
# Stages: archive, fixed-step, tune, prior-starts, prior-only, geometry-pilot, geometry,
#         relu, depth4, prior-scale, concentrated, nonlinear, sin
#
# Env: CONFIG, DEVICE, TIME_LIMIT, DRY_RUN=1
#
set -euo pipefail
cd "$(dirname "$0")/../.."
STAGE="${STAGE:-fixed-step}"
CONFIG="${CONFIG:-configs/revision.yaml}"
DRY_RUN="${DRY_RUN:-0}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
mkdir -p logs/revision

if [ "$STAGE" = "archive" ]; then
    if [ "$DRY_RUN" = "1" ]; then
        echo "sbatch --export=ALL,CONFIG=$CONFIG scripts/revision/archive.sbatch"
    else
        sbatch --export="ALL,CONFIG=$CONFIG" scripts/revision/archive.sbatch
    fi
    exit 0
fi

# Counts match tasks_for in src/bnn_geometry/revision/runner.py. The login node may not have h5py.
case "$STAGE" in
    fixed-step) N=24 ;;
    tune|prior-starts) N=12 ;;
    prior-only) N=2 ;;
    geometry-pilot) N=6 ;;
    geometry) N=36 ;;
    relu|depth4|prior-scale|concentrated|nonlinear|sin) N=2 ;;
    *)
        echo "stage $STAGE has no tasks"
        exit 1
        ;;
esac
ARRAY="0-$((N - 1))"
EXPORTS="ALL,CONFIG=$CONFIG,STAGE=$STAGE${DEVICE:+,DEVICE=$DEVICE}"
echo "stage $STAGE: $N tasks, array $ARRAY, config $CONFIG"
if [ "$DRY_RUN" = "1" ]; then
    echo "sbatch --array=$ARRAY --time=$TIME_LIMIT --export=$EXPORTS scripts/revision/target_stage.sbatch"
    exit 0
fi
sbatch --array="$ARRAY" --time="$TIME_LIMIT" --export="$EXPORTS" scripts/revision/target_stage.sbatch
