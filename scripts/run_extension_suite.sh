#!/bin/bash
#SBATCH --job-name=cyl_ext
#SBATCH --time=24:00:00                    # Queue limit; split with FILTER and resubmit to resume
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"  # Email when job starts/finishes/fails
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/cyl_ext/cyl_ext_%j.out
#SBATCH --error=logs/cyl_ext/cyl_ext_%j.err

#
# Larger-sample extension suite (cylinder_larger_sample_extension.md) + all postprocessing.
# Run from the project root. SLURM needs the log dir to exist before submission:
#
#   mkdir -p logs/cyl_ext && sbatch scripts/run_extension_suite.sh
#
# Order (addendum §9): prepare data + theorem table -> numerical validation (fatal) ->
# pytest -> profile n=128,m=4096 -> n=128 -> n=64 -> n=256 -> Fashion-MNIST ->
# diagnostics/extensions -> summaries -> figures -> export.
#
# Heavy per-target arrays stay in artifacts_ext/ (git-ignored). Everything worth keeping
# (CSV/JSON tables, validation/profile reports, per-target metadata + diagnostics + per-state
# curvature brackets + gzipped scalar traces, figures, SUMMARY.md, manifest.json) is exported to
# results/extension/, which is small enough to commit.
#
# Env options (all optional):
#   CONFIG          config file (default: config.ext.yaml; config.ext.smoke.yaml for a quick test)
#   DEVICE          cuda | cuda:0 | cpu (default: config `device`, i.e. auto)
#   FILTER          regex on target names, e.g. 'orth_n128_' or 'fmnist_' (default: all)
#   REALDATA        1 = include Fashion-MNIST appendix targets (default 1)
#   RUN_VALIDATE    1 = numerical validation before sampling; failure aborts (default 1)
#   VALIDATE_LIGHT  1 = shorter matched-sampling validation runs (default 0)
#   RUN_TESTS       1 = run pytest before sampling (default 1)
#   RUN_PROFILE     1 = profile n=128,m=4096 before the grid (default 1; ALL_PROFILE=1 for all settings)
#   EXTEND          1 = extend targets failing diagnostics to 8000 then 16000 (default 1)
#   OVERWRITE       1 = rerun targets from scratch; 0 = skip finished ones (default 0)
#   POST            1 = run summaries/figures/export at the end (default 1; set 0 for parallel split jobs)
#   POST_ONLY       1 = skip sampling, only redo diagnostics/postprocessing/export (default 0)
#   RESULTS_DIR     export dir override (default: config `results_dir`)
#   SETUP_VENV      1 = create .venv and pip install requirements-cuda.txt if missing
#   CYL_DIR         project root override
#   LOG_DIR         directory for the tee'd log (default: logs/cyl_ext)
#
# Examples:
#   mkdir -p logs/cyl_ext && sbatch scripts/run_extension_suite.sh
#   CONFIG=config.ext.smoke.yaml bash scripts/run_extension_suite.sh         # quick local test
#   FILTER='orth_n(64|128)_' sbatch scripts/run_extension_suite.sh            # split: n=64,128 ...
#   FILTER='orth_n256_|fmnist_' RUN_VALIDATE=0 RUN_PROFILE=0 sbatch scripts/run_extension_suite.sh
#   POST_ONLY=1 bash scripts/run_extension_suite.sh                            # regenerate results/extension
#
set -uo pipefail

if [ -n "${CYL_DIR:-}" ]; then
    PROJ_DIR="$CYL_DIR"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJ_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJ_DIR" || { echo "ERROR: Cannot cd to $PROJ_DIR"; exit 1; }

CONFIG="${CONFIG:-config.ext.yaml}"
FILTER="${FILTER:-}"
REALDATA="${REALDATA:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
VALIDATE_LIGHT="${VALIDATE_LIGHT:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_PROFILE="${RUN_PROFILE:-1}"
ALL_PROFILE="${ALL_PROFILE:-0}"
EXTEND="${EXTEND:-1}"
OVERWRITE="${OVERWRITE:-0}"
POST="${POST:-1}"
POST_ONLY="${POST_ONLY:-0}"
SETUP_VENV="${SETUP_VENV:-0}"

LOG_DIR="${LOG_DIR:-logs/cyl_ext}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ext_${TIMESTAMP}${SLURM_JOB_ID:+_job$SLURM_JOB_ID}.log"
echo "Log file: $LOG_FILE" | tee "$LOG_FILE"

log() {
    echo "$@" | tee -a "$LOG_FILE"
}

if [ ! -d ".venv" ] && [ ! -d "venv" ] && [ "$SETUP_VENV" = "1" ]; then
    log "Creating .venv and installing requirements-cuda.txt"
    python3 -m venv .venv 2>&1 | tee -a "$LOG_FILE"
    .venv/bin/pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
    .venv/bin/pip install -r requirements-cuda.txt 2>&1 | tee -a "$LOG_FILE"
fi
if [ -d ".venv" ]; then
    source .venv/bin/activate
    log "Using .venv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    log "Using venv"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CFG_ARGS=(--config "$CONFIG")
DEV_ARGS=()
[ -n "${DEVICE:-}" ] && DEV_ARGS+=(--device "$DEVICE")
EXPORT_ARGS=()
[ -n "${RESULTS_DIR:-}" ] && EXPORT_ARGS+=(--out "$RESULTS_DIR")
FILTER_ARGS=()
[ -n "$FILTER" ] && FILTER_ARGS+=(--filter "$FILTER")
LIST_ARGS=()
[ "$REALDATA" != "1" ] && LIST_ARGS+=(--no-realdata)
if [ "$OVERWRITE" = "1" ]; then
    RESUME_ARGS=(--overwrite)
else
    RESUME_ARGS=(--skip-existing)
fi

cfg_get() {
    python3 -c "
import sys, yaml
cfg = yaml.safe_load(open('$CONFIG'))
v = eval(sys.argv[1], {}, {'cfg': cfg})
print(' '.join(map(str, v)) if isinstance(v, list) else v)
" "$1"
}

FAILED=()
run_step() {
    # run_step <label> <cmd...>: tee output to the log, record failures, never abort.
    local label="$1"; shift
    log ""
    log "=============================================="
    log "[$(date +%H:%M:%S)] $label"
    log "Command: $*"
    log "=============================================="
    local t0=$SECONDS
    "$@" 2>&1 | tee -a "$LOG_FILE"
    local r=${PIPESTATUS[0]}
    log "[$label] exit=$r elapsed=$((SECONDS - t0))s"
    if [ "$r" -ne 0 ]; then
        FAILED+=("$label (exit $r)")
    fi
    return "$r"
}

log "=== Cylinder extension suite started at $(date) ==="
log "=== Working directory: $PROJ_DIR ==="
log "=== Host: $(hostname)  SLURM job: ${SLURM_JOB_ID:-none} ==="
log "=== Git: $(git rev-parse --short HEAD 2>/dev/null || echo n/a) $(git diff --quiet 2>/dev/null || echo '(dirty)') ==="
log "=== CONFIG=$CONFIG FILTER='${FILTER}' REALDATA=$REALDATA EXTEND=$EXTEND OVERWRITE=$OVERWRITE POST=$POST POST_ONLY=$POST_ONLY DEVICE=${DEVICE:-auto} OMP_NUM_THREADS=$OMP_NUM_THREADS ==="
log "=== Python ==="
python3 --version 2>&1 | tee -a "$LOG_FILE" || true
log "=== PyTorch / GPU ==="
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
" 2>&1 | tee -a "$LOG_FILE" || true
command -v nvidia-smi >/dev/null && nvidia-smi 2>&1 | tee -a "$LOG_FILE"
log "=== Disk free: $(df -h . | tail -1) ==="

TARGETS="$(python3 scripts/ext/list_targets.py "${CFG_ARGS[@]}" ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"} ${LIST_ARGS[@]+"${LIST_ARGS[@]}"})" || {
    log "FAIL: could not list targets from $CONFIG"; exit 1; }
N_TARGETS=$(echo "$TARGETS" | grep -c . || true)
log "=== $N_TARGETS target(s): $(echo $TARGETS) ==="

if [ "$POST_ONLY" != "1" ]; then
    if ! run_step "prepare (theorem table + frozen data)" python3 scripts/ext/10_prepare.py "${CFG_ARGS[@]}"; then
        log "FAIL: theorem table or data preparation failed; not sampling."
        exit 1
    fi
    if [ "$RUN_VALIDATE" = "1" ]; then
        VAL_ARGS=()
        [ "$VALIDATE_LIGHT" = "1" ] && VAL_ARGS+=(--light)
        if ! run_step "numerical validation" python3 scripts/ext/11_validate.py "${CFG_ARGS[@]}" ${VAL_ARGS[@]+"${VAL_ARGS[@]}"}; then
            log "FAIL: numerical validation failed (see validation.json); not sampling."
            exit 1
        fi
    fi
    if [ "$RUN_TESTS" = "1" ]; then
        if ! run_step "pytest" python3 -m pytest -q; then
            log "FAIL: test suite failed; not sampling."
            exit 1
        fi
    fi
    if [ "$RUN_PROFILE" = "1" ]; then
        PROF_ARGS=()
        [ "$ALL_PROFILE" = "1" ] && PROF_ARGS+=(--all-settings)
        run_step "profile" python3 scripts/ext/12_profile.py "${CFG_ARGS[@]}" \
            ${DEV_ARGS[@]+"${DEV_ARGS[@]}"} ${PROF_ARGS[@]+"${PROF_ARGS[@]}"}
    fi

    for t in $TARGETS; do
        run_step "target $t" python3 scripts/ext/13_run_target.py "${CFG_ARGS[@]}" --target "$t" \
            ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
    done

    EXT_ARGS=()
    [ "$EXTEND" = "1" ] && EXT_ARGS+=(--extend)
    run_step "diagnostics/extension" python3 scripts/ext/14_extend.py "${CFG_ARGS[@]}" \
        ${EXT_ARGS[@]+"${EXT_ARGS[@]}"} ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
else
    run_step "diagnostics" python3 scripts/ext/14_extend.py "${CFG_ARGS[@]}" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
fi

RESULTS_PATH="${RESULTS_DIR:-$(cfg_get 'cfg.get("results_dir", "results/extension")')}"
ARTIFACTS_PATH="$(cfg_get 'cfg.get("artifacts_dir", "artifacts_ext")')"

if [ "$POST" = "1" ] || [ "$POST_ONLY" = "1" ]; then
    run_step "summarize" python3 scripts/ext/15_summarize.py "${CFG_ARGS[@]}"
    run_step "figures" python3 scripts/ext/16_figures.py "${CFG_ARGS[@]}"
    run_step "export" python3 scripts/ext/17_export.py "${CFG_ARGS[@]}" ${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}
fi

log ""
log "=== Cylinder extension suite finished at $(date) (elapsed ${SECONDS}s) ==="
log "Heavy artifacts: $ARTIFACTS_PATH/ ($(du -sh "$ARTIFACTS_PATH" 2>/dev/null | cut -f1))"
if [ -d "$RESULTS_PATH" ]; then
    log "Git-trackable results: $RESULTS_PATH/ ($(du -sh "$RESULTS_PATH" 2>/dev/null | cut -f1))"
fi
if [ -f "$RESULTS_PATH/SUMMARY.md" ] && { [ "$POST" = "1" ] || [ "$POST_ONLY" = "1" ]; }; then
    log ""
    tee -a "$LOG_FILE" < "$RESULTS_PATH/SUMMARY.md"
fi

if [ ${#FAILED[@]} -ne 0 ]; then
    log ""
    log "FAIL: ${#FAILED[@]} step(s) failed:"
    for f in ${FAILED[@]+"${FAILED[@]}"}; do log "  - $f"; done
    log "Finished targets are kept; resubmit the same command to resume the rest."
    exit 1
fi
log ""
if [ "$POST" = "1" ] || [ "$POST_ONLY" = "1" ]; then
    log "SUCCESS. To store the metrics in git:"
    log "  git add $RESULTS_PATH && git commit -m 'Add extension results ($(hostname -s), job ${SLURM_JOB_ID:-local})' && git push"
else
    log "SUCCESS (POST=0). After all split jobs finish: POST_ONLY=1 bash scripts/run_extension_suite.sh"
fi
exit 0
