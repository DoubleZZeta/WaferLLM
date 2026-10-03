set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -ne 7 ]; then
    echo "Usage: bash run_device.sh PX PY M R N Gx Gy" >&2
    exit 2
fi
SHAPE_ARGS=(--PX "$1" --PY "$2" --M "$3" --R "$4" --N "$5" --group_num_x "$6" --group_num_y "$7")
python3 compile.py --mode device "${SHAPE_ARGS[@]}"
python3 launch_device.py "${SHAPE_ARGS[@]}"
