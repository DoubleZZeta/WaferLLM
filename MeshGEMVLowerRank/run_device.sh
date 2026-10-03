set -e
# Usage: ./run_device.sh P M R N group_num
if [ "$#" -ne 5 ]; then
    echo "Usage: bash run_device.sh P M R N group_num" >&2
    exit 2
fi
P=$1
M=$2
R=$3
N=$4
G=$5

echo "P=$P, M=$M, R=$R, N=$N, group_num=$G"

python compile.py --mode device --P "$P" --M "$M" --R "$R" --N "$N" --group_num "$G"
python launch_device.py --P "$P" --M "$M" --R "$R" --N "$N" --group_num "$G"
