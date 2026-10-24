set -euo pipefail
cd -- "$(dirname -- "$0")"

if [ "$#" -lt 7 ]; then
    echo "Usage: bash run_sim.sh PX PY M R N Gx Gy [--profile-phases] [--schedule pipeline|serial]" >&2
    exit 2
fi

PX=$1
PY=$2
M=$3
R=$4
N=$5
GX=$6
GY=$7

shift 7
EXTRA_ARGS=("$@")

SHAPE_ARGS=(
    --PX "$PX" --PY "$PY"
    --M "$M" --R "$R" --N "$N"
    --group_num_x "$GX" --group_num_y "$GY"
)

python3 compile.py --mode sim "${SHAPE_ARGS[@]}" "${EXTRA_ARGS[@]}"
cs_python launch_sim.py "${SHAPE_ARGS[@]}" "${EXTRA_ARGS[@]}"
