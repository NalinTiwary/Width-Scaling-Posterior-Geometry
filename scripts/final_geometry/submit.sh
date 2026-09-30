#!/bin/bash
#
# Submit the whole final-geometry campaign as a SLURM dependency chain (runbook §15.2):
#
#   0  freeze   validate config, plan, freeze data/centers/manifest, full pytest, benchmark
#   A  array    reference sampling + gates + static + predictive + step calibration (24 tasks)
#   B  global   architecture-wide step selection + prior spectral control
#   C  array    production dynamics at the selected step + h/2 endpoint runs (24 tasks)
#   D  global   endpoint decision round 1
#   E  array    architecture-wide refinement where D requires it (no-op otherwise) (24 tasks)
#   F  global   refinement decision round 2, analyze, figures, SUMMARY, audit, export
#
# All 24 targets of an array run at once. Each array is followed by RETRIES continuation arrays
# (afterany) that resume from checkpoints if a task hit the 24 h limit; finished tasks exit in
# seconds. Downstream global jobs use afterany so one failed target cannot stall the campaign
# (its failure is recorded in targets/<id>/errors.json and reported by the audit).
#
# Usage (from the repository root):
#   bash scripts/final_geometry/submit.sh
# Env:
#   FROM=0|A|B|C|D|E|F   start the chain at this step (earlier outputs must exist)  [0]
#   RETRIES=n            continuation arrays after each array                         [2]
#   CONFIG, OUT, DEVICE  forwarded to python -m bnn_geometry                          [configs/campaign.yaml]
#   DRY_RUN=1            print the sbatch commands only
#
set -euo pipefail
cd "$(dirname "$0")/../.."
FROM="${FROM:-0}"
RETRIES="${RETRIES:-2}"
CONFIG="${CONFIG:-configs/campaign.yaml}"
DRY_RUN="${DRY_RUN:-0}"
D=scripts/final_geometry
mkdir -p logs/final_geometry

N=$(PYTHONPATH=src python3 -c "
from bnn_geometry import config as C
print(len(C.targets(C.load('$CONFIG'))))")
ARRAY="0-$((N - 1))"
EXPORTS="ALL,CONFIG=$CONFIG${OUT:+,OUT=$OUT}${DEVICE:+,DEVICE=$DEVICE}"

order="0 A B C D E F"
started=0
prev=""
JOBLOG="logs/final_geometry/jobs_$(date +%Y%m%d_%H%M%S).txt"

submit() {  # submit <label> <dependency-type> <sbatch args...>; sets $prev
    local label="$1" dep="$2"; shift 2
    local depargs=()
    [ -n "$prev" ] && depargs=(--dependency="$dep:$prev")
    if [ "$DRY_RUN" = "1" ]; then
        echo "sbatch --parsable ${depargs[*]:-} $*"
        prev="DRY_$label"
    else
        prev=$(sbatch --parsable ${depargs[@]+"${depargs[@]}"} "$@")
        prev="${prev%%;*}"
    fi
    printf '%-26s %s\n' "$label" "$prev" | tee -a "$JOBLOG"
}

target_array() {  # target_array <step-letter> <stage>
    submit "$1 $2" afterany --job-name="bnn_$1_$2" --array="$ARRAY" --export="$EXPORTS,STAGE=$2" "$D/target_stage.sbatch"
    for i in $(seq 1 "$RETRIES"); do
        submit "$1 $2 (continuation $i)" afterany --job-name="bnn_$1_$2_c$i" --array="$ARRAY" \
            --export="$EXPORTS,STAGE=$2" "$D/target_stage.sbatch"
    done
}

global_job() {  # global_job <label> <dep> <time> <steps>
    submit "$1" "$2" --job-name="bnn_${1%% *}" --time="$3" --export="$EXPORTS,STEPS=$4" "$D/global_stage.sbatch"
}

echo "Campaign config $CONFIG: $N targets, arrays $ARRAY, $RETRIES continuation(s) per array, starting at $FROM"
for step in $order; do
    [ "$step" = "$FROM" ] && started=1
    [ "$started" = "1" ] || continue
    case "$step" in
        0) global_job "0 freeze" afterok 08:00:00 "freeze" ;;
        A) # the reference arrays must not start if freezing/tests failed
           submit "A reference" afterok --job-name=bnn_A_reference --array="$ARRAY" \
               --export="$EXPORTS,STAGE=reference" "$D/target_stage.sbatch"
           for i in $(seq 1 "$RETRIES"); do
               submit "A reference (continuation $i)" afterany --job-name="bnn_A_reference_c$i" --array="$ARRAY" \
                   --export="$EXPORTS,STAGE=reference" "$D/target_stage.sbatch"
           done ;;
        B) global_job "B select-step+controls" afterany 06:00:00 "select-step controls" ;;
        C) target_array C production ;;
        D) global_job "D endpoint-decision" afterany 04:00:00 "endpoint-decision" ;;
        E) target_array E refine ;;
        F) global_job "F final+post" afterany 12:00:00 "refine-decision post" ;;
    esac
done
[ "$started" = "1" ] || { echo "unknown FROM=$FROM (use one of: $order)"; exit 1; }
echo "Job ids saved to $JOBLOG. Monitor: squeue -u \$USER ; logs in logs/final_geometry/"
echo "Cancel everything: scancel \$(awk '{print \$NF}' $JOBLOG)"
