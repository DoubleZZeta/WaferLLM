#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
    echo "Usage: bash run_chunked.sh P M N group_num repetitions [rows_per_copy] [result_values_per_pe]" >&2
    exit 2
fi

P=$1
M=$2
N=$3
GROUP_NUM=$4
REPETITIONS=$5
ROWS_PER_COPY=${6:-1}
RESULT_VALUES_PER_PE=${7:-}

PROBE_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(cd "$PROBE_DIR/../.." && pwd)

echo "Compiling the existing dense MeshGEMV kernel"
cd "$REPO_DIR/MeshGEMVPatched"
python3 compile.py \
    --mode sim \
    --P "$P" \
    --M "$M" \
    --N "$N" \
    --group_num "$GROUP_NUM"

echo "Running the chunked-transfer diagnostic launcher"
cd "$REPO_DIR"
LAUNCH_ARGS=(
    --P "$P"
    --M "$M"
    --N "$N"
    --group_num "$GROUP_NUM"
    --repetitions "$REPETITIONS"
    --rows-per-copy "$ROWS_PER_COPY"
)
if [ -n "$RESULT_VALUES_PER_PE" ]; then
    LAUNCH_ARGS+=(--result-values-per-pe "$RESULT_VALUES_PER_PE")
fi

cs_python "$PROBE_DIR/launch_sim_chunked.py" \
    "${LAUNCH_ARGS[@]}"
