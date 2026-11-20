#!/bin/bash
# Login-node check. No GPU and no sbatch. Run from anywhere:
#   bash scripts/revision/login_check.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
echo "=== login check in $PWD ==="
echo "=== python $($PY -c 'import sys; print(sys.executable)') PYTHONPATH=$PYTHONPATH ==="
"$PY" -m bnn_geometry.revision preflight --config configs/revision.yaml
pre=$?
"$PY" -m bnn_geometry.revision prediction-acf --config configs/revision.yaml --device cpu
pred=$?
echo "=== login_check preflight=$pre prediction=$pred ==="
if [ "$pre" -ne 0 ]; then
    exit "$pre"
fi
exit "$pred"
