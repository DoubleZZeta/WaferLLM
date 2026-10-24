set -euo pipefail
cd -- "$(dirname -- "$0")"

if [ "$#" -lt 7 ] || [ "$#" -gt 8 ]; then
    echo "Usage: bash run_sim.sh PX PY M R N Gx Gy [--profile-phases]" >&2
    exit 2
fi

PX=$1
PY=$2
M=$3
R=$4
N=$5
GX=$6
GY=$7

PROFILE_ARGS=()
if [ "$#" -eq 8 ]; then
    if [ "$8" != "--profile-phases" ]; then
        echo "Unknown option: $8" >&2
        exit 2
    fi
    PROFILE_ARGS+=("--profile-phases")
fi

SHAPE_ARGS=(
    --PX "$PX" --PY "$PY"
    --M "$M" --R "$R" --N "$N"
    --group_num_x "$GX" --group_num_y "$GY"
)

python3 compile.py --mode sim "${SHAPE_ARGS[@]}"
cs_python launch_sim.py "${SHAPE_ARGS[@]}" "${PROFILE_ARGS[@]}"
