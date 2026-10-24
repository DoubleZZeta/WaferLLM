#!/usr/bin/env bash
set -euo pipefail

# Start one dense or fused Llama-2 projection benchmark in a detached tmux
# session. Extra arguments are passed directly to bench.py.
#
# Usage:
#   bash run.sh {dense|fused|dense-rectangular|fused-rectangular|fused-chunked|fused-pipelining} [run_name] [bench.py options]
#
# Examples:
#   bash run.sh dense p64 --P 64 --group-num 8
#   bash run.sh fused p64 --retained-work-ratios .125,.25,.5,.75
#   bash run.sh dense p64_fixed --P 64 --group-num 8
#   bash run.sh fused p64_fixed --retained-work-ratios .2,.4,.6,.8
# Dense and fused use the HCF upload fix by default.
#
# Both subcommands of one run share runs/<run_name>/ and may run at the same
# time. Rebuild the summaries directly with:
#   python3 bench.py summary --run-name p64

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")

if [ "$#" -lt 1 ] || { [ "$1" != dense ] && [ "$1" != fused ] && [ "$1" != fused-rectangular ] && [ "$1" != dense-rectangular ] && [ "$1" != fused-chunked ] && [ "$1" != fused-pipelining ]; }; then
    echo "Usage: bash run.sh {dense|fused|dense-rectangular|fused-rectangular|fused-chunked|fused-pipelining} [run_name] [bench.py options]" >&2
    exit 2
fi
SUBCOMMAND=$1
shift

SDK_DIR=${CEREBRAS_SDK_DIR:-"$(dirname "$REPO_DIR")/cs_sdk"}
if [ ! -x "$SDK_DIR/cslc" ] || [ ! -x "$SDK_DIR/cs_python" ]; then
    echo "Cerebras SDK launchers not found under: $SDK_DIR" >&2
    echo "Set CEREBRAS_SDK_DIR to the directory containing cslc and cs_python." >&2
    exit 1
fi

RUN_NAME="llama2_7b_$(date +%Y%m%d_%H%M%S)"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    RUN_NAME=$1
    shift
fi

case "$RUN_NAME" in
    ""|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*)
        echo "run_name must start with a letter or digit and may contain dots, underscores, and dashes" >&2
        exit 2
        ;;
esac

SESSION_NAME="llama_${SUBCOMMAND}_${RUN_NAME//./_}"
RUN_DIR="$SCRIPT_DIR/runs/$RUN_NAME"
RUN_LOG="$RUN_DIR/$SUBCOMMAND.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
printf -v SDK_DIR_QUOTED '%q' "$SDK_DIR"
printf -v RUN_LOG_QUOTED '%q' "$RUN_LOG"
printf -v BENCHMARK_COMMAND '%q ' \
    python3 bench.py "$SUBCOMMAND" --run-name "$RUN_NAME" --resume "$@"
COMMAND="export PATH=$SDK_DIR_QUOTED:\$PATH; $BENCHMARK_COMMAND"
COMMAND+="2>&1 | tee -a $RUN_LOG_QUOTED"

tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" "$COMMAND"

echo "Started Llama-2 projection benchmark ($SUBCOMMAND)."
echo "  run:     $RUN_NAME"
echo "  session: $SESSION_NAME"
echo "  results: $RUN_DIR/$SUBCOMMAND.csv"
if [[ "$SUBCOMMAND" != *-rectangular && "$SUBCOMMAND" != fused-chunked && "$SUBCOMMAND" != fused-pipelining ]]; then echo "  summary: $RUN_DIR/block_summary.csv"; fi
echo "  attach:  tmux attach -t $SESSION_NAME"
echo "  log:     tail -f $RUN_LOG"
