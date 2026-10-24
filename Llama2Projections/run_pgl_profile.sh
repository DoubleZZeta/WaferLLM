#!/usr/bin/env bash
# Full-GEMV phase profiles: 39 matched cases plus one dense Y L=172 case.
# Usage: bash run_pgl_profile.sh RUN_PREFIX [--dry-run] [--build-root PATH]
set -euo pipefail
cd -- "$(dirname -- "$0")"

PREFIX=${1:?Usage: bash run_pgl_profile.sh RUN_PREFIX [sweep options]}
shift

# Use existing kernels. All grids are square. G is group count, not size.
# Dense: Mt=64, Nt=L. Low-rank Y: Mt=Nt=L+1, Rt=L.
# Low-rank X: Mt=64, Rt=8, Nt=L.
# Y shapes are communication probes, not compression/speedup experiments.
while read -r P G L; do
    echo "P=$P G=$G L=$L"
    python3 sweep.py dense --run-name "${PREFIX}_dense" \
        --P "$P" --group-num "$G" --M "$((64*P))" --N "$((L*P))" \
        --profile-phases --suppress-simfab-trace --resume "$@"

    python3 sweep.py fused-rectangular --run-name "${PREFIX}_y" \
        --PX "$P" --PY "$P" --group-num-x "$G" --group-num-y "$G" \
        --M "$(((L+1)*P))" --N "$(((L+1)*P))" --ranks "$((L*P))" \
        --profile-phases --suppress-simfab-trace --resume "$@"

    python3 sweep.py fused-rectangular --run-name "${PREFIX}_x" \
        --PX "$P" --PY "$P" --group-num-x "$G" --group-num-y "$G" \
        --M "$((64*P))" --N "$((L*P))" --ranks "$((8*P))" \
        --profile-phases --suppress-simfab-trace --resume "$@"
done <<'CASES'
16 8 64
32 8 64
64 8 64
96 8 64
128 8 64
64 2 64
64 4 64
64 16 64
64 32 64
64 8 16
64 8 33
64 8 48
64 8 80
CASES

# Additional gate/up-sized payload; only dense Y, whose tile fits in memory.
echo "Dense Y only: P=64 G=8 L=172"
python3 sweep.py dense --run-name "${PREFIX}_dense" \
    --P 64 --group-num 8 --M 4096 --N 11008 \
    --profile-phases --suppress-simfab-trace --resume "$@"
