#!/usr/bin/env bash
set -euo pipefail

# Direct-factor simulator path used only by benchmark_llama2.py.
if [ "$#" -ne 5 ]; then
    echo "Usage: bash run_sim_on_host_llama.sh P M R N group_num" >&2
    exit 2
fi

P=$1
M=$2
R=$3
N=$4
G=$5

echo "P=$P, M=$M, R=$R, N=$N, group_num=$G"
echo "Compiling stage 1: [1,$M] @ [$M,$R]"
python3 compile.py --mode sim --P "$P" --M "$M" --N "$R" --group_num "$G"
echo "Compiling stage 2: [1,$R] @ [$R,$N]"
python3 compile.py --mode sim --P "$P" --M "$R" --N "$N" --group_num "$G"

STATE_FILE="low_rank_llama_${P}_${M}_${R}_${N}_${G}.npz"
cs_python launch_sim_on_host_llama.py --stage 1 \
    --P "$P" --M "$M" --R "$R" --N "$N" --group_num "$G" \
    --state-file "$STATE_FILE"
cs_python launch_sim_on_host_llama.py --stage 2 \
    --P "$P" --M "$M" --R "$R" --N "$N" --group_num "$G" \
    --state-file "$STATE_FILE"
