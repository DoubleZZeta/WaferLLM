set -e
# Usage: ./run_sim.sh P M N group_num [--profile-phases]
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: bash run_sim.sh P M N group_num [--profile-phases]" >&2
    exit 2
fi
P=$1
M=$2
N=$3
G=$4
PROFILE_ARGS=()
if [ "$#" -eq 5 ]; then
    if [ "$5" != "--profile-phases" ]; then
        echo "Unknown option: $5" >&2
        exit 2
    fi
    PROFILE_ARGS+=("--profile-phases")
fi

echo "P=$P, M=$M, N=$N, group_num=$G"

python3 compile.py --mode sim --P "$P" --M "$M" --N "$N" --group_num "$G"
cs_python launch_sim.py --P "$P" --M "$M" --N "$N" --group_num "$G" "${PROFILE_ARGS[@]}"
