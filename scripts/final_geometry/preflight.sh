#!/bin/bash
#
# Login-node checks before submitting the campaign (run automatically by submit.sh; ~20 s, no GPU).
# Fails on anything that would make the queued jobs die: missing packages, invalid config,
# failing unit tests, SLURM rejecting the job options, too little disk, or a mismatched output root.
#
#   bash scripts/final_geometry/preflight.sh          # env: CONFIG, OUT, BNN_VENV as for submit.sh
#
source "$(dirname "$0")/common.sh"
fail=0
ok()   { echo "  ok    $*"; }
bad()  { echo "  FAIL  $*"; fail=1; }
warn() { echo "  warn  $*"; }

echo "== Python environment"
python3 - <<'EOF' && ok "imports and versions" || bad "python environment (install: pip install -r requirements-cuda.txt)"
import sys
mods = {}
for name in ["numpy", "scipy", "torch", "h5py", "yaml", "pandas", "matplotlib", "arviz"]:
    mods[name] = __import__(name)
v = {k: getattr(m, "__version__", "?") for k, m in mods.items()}
print("        python", sys.version.split()[0], " ".join(f"{k}={x}" for k, x in v.items()))
assert sys.version_info >= (3, 9), "Python >= 3.9 required"
import re
def V(s):
    return tuple(int(x) for x in re.findall(r"\d+", s.split("+")[0])[:3])
if V(v["arviz"]) < V("0.18"):
    assert V(v["scipy"]) < V("1.13"), "arviz<0.18 needs scipy<1.13"
assert V(v["torch"].split("+")[0]) >= V("2.1"), "torch>=2.1 required"
EOF
cuda_build=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
if [ -n "$cuda_build" ] && [ "$cuda_build" != "None" ]; then
    ok "torch CUDA build $cuda_build (GPU visibility is checked inside the jobs)"
elif [ "${ALLOW_CPU:-0}" = "1" ]; then
    warn "CPU-only torch build (ALLOW_CPU=1)"
else
    bad "CPU-only torch build: jobs would run on CPU. pip install torch --index-url https://download.pytorch.org/whl/cu121"
fi

echo "== Configuration and plan"
bnn validate-config "${BNN_ARGS[@]}" >/dev/null && ok "config $CONFIG valid" || bad "config $CONFIG invalid"
bnn plan "${BNN_ARGS[@]}" | tail -5 | sed 's/^/        /'

echo "== Unit tests (CPU)"
if python3 -m pytest -q -x tests/unit -p no:cacheprovider > /tmp/bnn_preflight_$$.log 2>&1; then
    ok "$(tail -1 /tmp/bnn_preflight_$$.log)"
else
    tail -30 /tmp/bnn_preflight_$$.log; bad "unit tests"
fi
rm -f /tmp/bnn_preflight_$$.log

echo "== Output root $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT" 2>/dev/null && [ -w "$OUTPUT_ROOT" ] && ok "writable" || bad "cannot write $OUTPUT_ROOT"
need=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['limits']['recommended_free_disk_gb'])")
free=$(df -Pk "$OUTPUT_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$free" -ge "$need" ]; then ok "free disk ${free} GB >= ${need} GB"; else warn "free disk ${free} GB < recommended ${need} GB"; fi
command -v quota >/dev/null && quota -s 2>/dev/null | tail -3 | sed 's/^/        /'
if [ -f "$OUTPUT_ROOT/manifest.json" ]; then
    python3 - "$OUTPUT_ROOT/manifest.json" "$CONFIG" <<'EOF' && ok "existing manifest matches config (resume)" || bad "output root holds a different campaign; use OUT=... or remove it"
import sys, json
sys.path.insert(0, "src")
from bnn_geometry import config as C
m = json.load(open(sys.argv[1]))
assert m["config_sha256"] == C.config_sha256(sys.argv[2]), "config sha256 differs"
EOF
fi

echo "== Git"
git fetch -q origin 2>/dev/null
[ -z "$(git status --porcelain -- src configs scripts)" ] && ok "no local edits in src/configs/scripts" \
    || warn "local edits in src/configs/scripts (recorded as dirty in the manifest)"
[ "$(git rev-parse HEAD)" = "$(git rev-parse '@{u}' 2>/dev/null)" ] && ok "up to date with $(git rev-parse --abbrev-ref '@{u}')" \
    || warn "HEAD differs from upstream; run git pull if unintended"

echo "== SLURM accepts the job options (sbatch --test-only, nothing is submitted)"
if command -v sbatch >/dev/null; then
    mkdir -p logs/final_geometry
    for spec in "target_stage.sbatch --array=0-23 --export=ALL,STAGE=reference" \
                "global_stage.sbatch --time=12:00:00 --export=ALL,STEPS=freeze"; do
        set -- $spec
        out=$(sbatch --test-only "${@:2}" "$(dirname "$0")/$1" 2>&1)
        if echo "$out" | grep -qi "error"; then bad "$1: $out"; else ok "$1: $(echo "$out" | tail -1)"; fi
    done
else
    warn "sbatch not found (not on a SLURM login node)"
fi

echo
if [ "$fail" = "0" ]; then echo "PREFLIGHT PASSED"; else echo "PREFLIGHT FAILED"; fi
exit $fail
