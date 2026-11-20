#!/bin/bash
# Login-node collector. No GPU. Writes results/revision/tables from summary JSON.
#   bash scripts/revision/collect.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -x .venv/bin/python ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
echo "=== collect in $PWD python=$PY ==="
"$PY" -m bnn_geometry.revision collect --config configs/revision.yaml
status=$?
echo "=== collect exit=$status ==="
exit $status
