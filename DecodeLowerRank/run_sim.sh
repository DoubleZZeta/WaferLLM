set -e
# Usage: ./run_sim.sh <config.json> <rank_config.json> [--profile-projections]
# Small configs only: simulator cost scales with P*P.
CONFIG=$1
RANK_CONFIG=$2
if [ -z "$CONFIG" ] || [ -z "$RANK_CONFIG" ]; then
    echo "usage: ./run_sim.sh <config.json> <rank_config.json>"
    exit 1
fi

python3 compile.py --mode sim --config "$CONFIG" --rank_config "$RANK_CONFIG"
cs_python launch_sim.py --config "$CONFIG" --rank_config "$RANK_CONFIG"

if [ "${3:-}" = "--profile-projections" ]; then
    python3 compile.py --mode sim --config "$CONFIG" --rank_config "$RANK_CONFIG" --profile-projections
    cs_python launch_sim.py --config "$CONFIG" --rank_config "$RANK_CONFIG" --profile-projections
elif [ -n "${3:-}" ]; then
    echo "unknown option: $3"
    exit 1
fi
