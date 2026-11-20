# Source from the repository root. 'go' is the only function that submits.
# The login node's plain python3 does not see the project packages. Prefer the
# virtualenv that already ran recovery and the finalists.
if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
todo() {
    PYTHONPATH="${PYTHONPATH:-$PWD/src}" "$PY" experiment_ops/pending.py "$@"
}
go() {
    local job="$1" scope="$2" want="$3" level="${4:-1}" ids receipt
    ids="$(todo "$job" "$scope" "$want" "$level")" || return 1
    if [ -z "$ids" ]; then
        echo "All requested cells already exist or are listed in active.csv."
        return 0
    fi
    case "$job" in
      reference) limit=24:00:00 ;;
      reuse) limit=02:00:00 ;;
      *) limit=08:00:00 ;;
    esac
    receipt="$(sbatch --parsable --array="$ids" --job-name="bnn_${job}" --time="$limit" \
      --comment="bnn:${job}:${scope}:L${level}" \
      --export="ALL,JOB=$job,SCOPE=$scope,LEVEL=$level,CONFIG=configs/revision.yaml,NEXT_OUT=results/revision_next" \
      a10_next/next.sbatch)" || return 1
    printf '%s,%s,"%s",%s\n' "$job" "$scope" "$ids" "${receipt%%;*}" >> results/revision_next/active.csv
    echo "$job $scope level $level tasks $ids submitted as $receipt"
}
