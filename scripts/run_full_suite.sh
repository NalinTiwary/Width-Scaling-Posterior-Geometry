#!/bin/bash
#SBATCH --job-name=cyl_suite
#SBATCH --time=24:00:00                    # Full grid incl. extensions; typically a few hours on one GPU
#SBATCH --mail-type=ALL,FAIL
#SBATCH --mail-user="nalint2@illinois.edu"  # Email when job starts/finishes/fails
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=arindamb-cs-eng
#SBATCH --partition=eng-research-gpu
#SBATCH --output=logs/cyl_suite/cyl_suite_%j.out
#SBATCH --error=logs/cyl_suite/cyl_suite_%j.err

#
# Full cylinder experiment suite (design doc §11.1) + all postprocessing.
# Run from the project root. SLURM needs the log dir to exist before submission:
#
#   mkdir -p logs/cyl_suite && sbatch scripts/run_full_suite.sh
#
# Heavy per-run arrays stay in artifacts_dir (artifacts/, git-ignored). Everything
# worth keeping (CSV/JSON metrics, per-run metadata + diagnostics, gzipped scalar
# traces, figures, SUMMARY.md, manifest.json) is exported to results_dir
# (results/paper/ for config.yaml), which is small enough to commit.
#
# Env options (all optional):
#   CONFIG          config file (default: config.yaml; config.smoke.yaml for a quick test)
#   DEVICE          cuda | cuda:0 | cpu (default: config `device`, i.e. auto)
#   WIDTHS          space-separated widths (default: from config)
#   SEEDS           space-separated center seeds (default: from config)
#   RUN_PCN         1 = run the pCN cross-check target (default 1)
#   EXTEND          1 = extend targets failing diagnostics per protocol (default 1)
#   OVERWRITE       1 = rerun targets from scratch; 0 = skip finished ones (default 0)
#   RUN_TESTS       1 = run pytest before sampling (default 1)
#   POST_ONLY       1 = skip sampling, only redo postprocessing/export (default 0)
#   RESULTS_DIR     export dir override (default: config `results_dir`)
#   CLEAN_HEAVY     1 = delete observables.npz after a successful export (default 0;
#                   note a later rerun would then have to resample those targets)
#   SETUP_VENV      1 = create .venv and pip install requirements-cuda.txt if missing
#   CYL_DIR         project root override
#   LOG_DIR         directory for the tee'd log (default: logs/cyl_suite)
#
# Examples:
#   mkdir -p logs/cyl_suite && sbatch scripts/run_full_suite.sh
#   CONFIG=config.smoke.yaml bash scripts/run_full_suite.sh          # quick local test
#   WIDTHS="4096" RUN_PCN=0 sbatch scripts/run_full_suite.sh          # just the m=4096 targets
#   POST_ONLY=1 bash scripts/run_full_suite.sh                        # regenerate results/
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

CONFIG="${CONFIG:-config.yaml}"
RUN_PCN="${RUN_PCN:-1}"
EXTEND="${EXTEND:-1}"
OVERWRITE="${OVERWRITE:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
POST_ONLY="${POST_ONLY:-0}"
CLEAN_HEAVY="${CLEAN_HEAVY:-0}"
SETUP_VENV="${SETUP_VENV:-0}"

LOG_DIR="${LOG_DIR:-logs/cyl_suite}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/suite_${TIMESTAMP}${SLURM_JOB_ID:+_job$SLURM_JOB_ID}.log"
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

DEV_ARGS=()
[ -n "${DEVICE:-}" ] && DEV_ARGS+=(--device "$DEVICE")
EXPORT_ARGS=()
[ -n "${RESULTS_DIR:-}" ] && EXPORT_ARGS+=(--out "$RESULTS_DIR")
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
WIDTHS="${WIDTHS:-$(cfg_get 'cfg["widths"]')}"
SEEDS="${SEEDS:-$(cfg_get 'cfg["prior"]["center_seeds"]')}"
PCN_M="$(cfg_get 'cfg["pcn_crosscheck"]["width"]')"
PCN_SEED="$(cfg_get 'cfg["pcn_crosscheck"]["center_seed"]')"

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

log "=== Cylinder suite started at $(date) ==="
log "=== Working directory: $PROJ_DIR ==="
log "=== Host: $(hostname)  SLURM job: ${SLURM_JOB_ID:-none} ==="
log "=== Git: $(git rev-parse --short HEAD 2>/dev/null || echo n/a) $(git diff --quiet 2>/dev/null || echo '(dirty)') ==="
log "=== CONFIG=$CONFIG WIDTHS='$WIDTHS' SEEDS='$SEEDS' RUN_PCN=$RUN_PCN EXTEND=$EXTEND OVERWRITE=$OVERWRITE POST_ONLY=$POST_ONLY DEVICE=${DEVICE:-auto} OMP_NUM_THREADS=$OMP_NUM_THREADS ==="
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

if [ "$POST_ONLY" != "1" ]; then
    if ! run_step "preflight" python3 scripts/00_preflight.py --config "$CONFIG" --write-data --skip-tests ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}; then
        log "FAIL: preflight failed; theorem/data checks must pass before sampling."
        exit 1
    fi
    if [ "$RUN_TESTS" = "1" ]; then
        if ! run_step "pytest" python3 -m pytest -q; then
            log "FAIL: test suite failed; not sampling."
            exit 1
        fi
    fi

    # Central target first (§11.1), then pCN, then the rest of the grid.
    targets=()
    for m in $WIDTHS; do
        for s in $SEEDS; do
            if [ "$m" = "$PCN_M" ] && [ "$s" = "$PCN_SEED" ]; then
                targets=("$m:$s" ${targets[@]+"${targets[@]}"})
            else
                targets+=("$m:$s")
            fi
        done
    done
    pcn_done=0
    for ts in ${targets[@]+"${targets[@]}"}; do
        m="${ts%%:*}"; s="${ts##*:}"
        run_step "ess m=$m seed=$s" python3 scripts/01_run_target.py --config "$CONFIG" \
            --m "$m" --seed "$s" --kernel ess ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
        if [ "$RUN_PCN" = "1" ] && [ "$pcn_done" = "0" ] && [ "$m" = "$PCN_M" ] && [ "$s" = "$PCN_SEED" ]; then
            run_step "pcn m=$m seed=$s" python3 scripts/01_run_target.py --config "$CONFIG" \
                --m "$m" --seed "$s" --kernel pcn ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
            pcn_done=1
        fi
    done
    if [ "$RUN_PCN" = "1" ] && [ "$pcn_done" = "0" ]; then
        run_step "pcn m=$PCN_M seed=$PCN_SEED" python3 scripts/01_run_target.py --config "$CONFIG" \
            --m "$PCN_M" --seed "$PCN_SEED" --kernel pcn ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
    fi

    EXT_ARGS=()
    [ "$EXTEND" = "1" ] && EXT_ARGS+=(--extend)
    run_step "diagnostics/extension" python3 scripts/02_extend_if_needed.py --config "$CONFIG" ${EXT_ARGS[@]+"${EXT_ARGS[@]}"} ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
else
    run_step "diagnostics" python3 scripts/02_extend_if_needed.py --config "$CONFIG" ${DEV_ARGS[@]+"${DEV_ARGS[@]}"}
fi

run_step "curvature" python3 scripts/03_curvature_states.py --config "$CONFIG"
run_step "summarize" python3 scripts/04_summarize.py --config "$CONFIG"
run_step "figures" python3 scripts/05_make_figures.py --config "$CONFIG"
run_step "export" python3 scripts/06_export_results.py --config "$CONFIG" ${EXPORT_ARGS[@]+"${EXPORT_ARGS[@]}"}
export_rc=$?

RESULTS_PATH="${RESULTS_DIR:-$(cfg_get 'cfg.get("results_dir", "results")')}"
ARTIFACTS_PATH="$(cfg_get 'cfg.get("artifacts_dir", "artifacts")')"

if [ "$CLEAN_HEAVY" = "1" ]; then
    if [ "$export_rc" -eq 0 ] && [ ${#FAILED[@]} -eq 0 ]; then
        log "Removing heavy observables.npz under $ARTIFACTS_PATH/runs"
        find "$ARTIFACTS_PATH/runs" -name observables.npz -print -delete 2>&1 | tee -a "$LOG_FILE"
    else
        log "CLEAN_HEAVY skipped because some steps failed."
    fi
fi

log ""
log "=== Cylinder suite finished at $(date) (elapsed ${SECONDS}s) ==="
log "Heavy artifacts: $ARTIFACTS_PATH/ ($(du -sh "$ARTIFACTS_PATH" 2>/dev/null | cut -f1))"
log "Git-trackable results: $RESULTS_PATH/ ($(du -sh "$RESULTS_PATH" 2>/dev/null | cut -f1))"
if [ -f "$RESULTS_PATH/SUMMARY.md" ]; then
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
log "SUCCESS. To store the metrics in git:"
log "  git add $RESULTS_PATH && git commit -m 'Add cylinder results ($(hostname -s), job ${SLURM_JOB_ID:-local})' && git push"
exit 0
