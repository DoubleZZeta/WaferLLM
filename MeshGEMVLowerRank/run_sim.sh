set -e
# Usage: ./run_sim.sh P M R N group_num [--profile-phases]
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: bash run_sim.sh P M R N group_num [--profile-phases]" >&2
    exit 2
fi
P=$1
M=$2
R=$3
N=$4
G=$5
PROFILE_ARGS=()
if [ "$#" -eq 6 ]; then
    if [ "$6" != "--profile-phases" ]; then
        echo "Unknown option: $6" >&2
        exit 2
    fi
    PROFILE_ARGS+=("--profile-phases")
fi

echo "P=$P, M=$M, R=$R, N=$N, group_num=$G"

python3 compile.py --mode sim --P "$P" --M "$M" --R "$R" --N "$N" --group_num "$G"
cs_python launch_sim.py --P "$P" --M "$M" --R "$R" --N "$N" --group_num "$G" \
    "${PROFILE_ARGS[@]}"
