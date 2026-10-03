set -e
# Usage: ./run_device.sh <config.json> <rank_config.json>
CONFIG=$1
RANK_CONFIG=$2
if [ -z "$CONFIG" ] || [ -z "$RANK_CONFIG" ]; then
    echo "usage: ./run_device.sh <config.json> <rank_config.json>"
    exit 1
fi

python3 compile.py --mode device --config "$CONFIG" --rank_config "$RANK_CONFIG"
python3 launch_device.py --config "$CONFIG" --rank_config "$RANK_CONFIG"
