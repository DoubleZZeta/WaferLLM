#!/usr/bin/env bash
set -euo pipefail

# Start one dense or fused parameter sweep in a detached tmux session. Extra
# arguments are passed directly to sweep.py.
#
# Usage:
#   bash run_sweep.sh {dense|fused|fused-rectangular} [run_name] [sweep.py options]
#
# Examples:
#   bash run_sweep.sh dense square --P 4,8 --group-num 2 --M 256,512 --N 256,512
#   bash run_sweep.sh fused square --P 4,8 --group-num 2 --M 256,512 --N 256,512 \
#     --retained-work-ratios .25,.5,.75
#
# Both subcommands of one run share sweeps/<run_name>/ and may run at the same
# time. Rebuild the summary directly with:
#   python3 sweep.py summary --run-name square

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")

if [ "$#" -lt 1 ] || { [ "$1" != dense ] && [ "$1" != fused ] && [ "$1" != fused-rectangular ]; }; then
    echo "Usage: bash run_sweep.sh {dense|fused|fused-rectangular} [run_name] [sweep.py options]" >&2
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

RUN_NAME="sweep_$(date +%Y%m%d_%H%M%S)"
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

SESSION_NAME="sweep_${SUBCOMMAND}_${RUN_NAME//./_}"
RUN_DIR="$SCRIPT_DIR/sweeps/$RUN_NAME"
RUN_LOG="$RUN_DIR/$SUBCOMMAND.log"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
printf -v SDK_DIR_QUOTED '%q' "$SDK_DIR"
printf -v RUN_LOG_QUOTED '%q' "$RUN_LOG"
printf -v SWEEP_COMMAND '%q ' \
    python3 sweep.py "$SUBCOMMAND" --run-name "$RUN_NAME" --resume "$@"
COMMAND="export PATH=$SDK_DIR_QUOTED:\$PATH; $SWEEP_COMMAND"
COMMAND+="2>&1 | tee -a $RUN_LOG_QUOTED"

tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" "$COMMAND"

echo "Started parameter sweep ($SUBCOMMAND)."
echo "  run:     $RUN_NAME"
echo "  session: $SESSION_NAME"
echo "  results: $RUN_DIR/$SUBCOMMAND.csv"
if [ "$SUBCOMMAND" != fused-rectangular ]; then echo "  summary: $RUN_DIR/summary.csv"; fi
echo "  attach:  tmux attach -t $SESSION_NAME"
echo "  log:     tail -f $RUN_LOG"
