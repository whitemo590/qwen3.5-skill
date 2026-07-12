#!/usr/bin/env bash
set -euo pipefail

ASCEND_ENV_SCRIPT="${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/cann/set_env.sh}"
if [[ ! -f "$ASCEND_ENV_SCRIPT" ]]; then
    echo "[ERROR] Ascend environment script not found: $ASCEND_ENV_SCRIPT" >&2
    exit 1
fi
source "$ASCEND_ENV_SCRIPT"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# 删除triton的cache
# export TRITON_CACHE_DIR=./triton_cache
# rm -rf $TRITON_CACHE_DIR/*

NPUS_PER_NODE="${NPUS_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MINDSPEED_MM_ROOT="${MINDSPEED_MM_ROOT:-$PROJECT_ROOT/third_party/MindSpeed-MM}"
CONFIG_PATH="${1:-$SCRIPT_DIR/qwen3_5_0.8B_config.yaml}"
TRAINER="$MINDSPEED_MM_ROOT/mindspeed_mm/fsdp/train/trainer.py"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config not found: $CONFIG_PATH" >&2
    exit 1
fi
if [[ ! -f "$TRAINER" ]]; then
    echo "[ERROR] MindSpeed-MM trainer not found: $TRAINER" >&2
    echo "[HINT] Set MINDSPEED_MM_ROOT to the MindSpeed-MM 26.0.0 checkout." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"
logfile="$(date +%Y%m%d)_$(date +%H%M%S)"
log_path="$LOG_DIR/train_${logfile}.log"

torchrun $DISTRIBUTED_ARGS "$TRAINER" \
    "$CONFIG_PATH" \
    2>&1 | tee "$log_path"

STEP_TIME=$(grep "elapsed time per iteration" "$log_path" | awk -F 'elapsed time per iteration [(]ms[)]:' '{print$2}' | awk -F '|' '{print$1}' | head -n 200 | tail -n 100 | awk '{sum+=$1} END {if (NR != 0) printf("%.1f",sum/NR)}' || true)
GBS=$(grep "global batch size" "$log_path" | awk -F 'global batch size:' '{print$2}' | awk -F '|' '{print$1}' | head -n 1 | awk '{print $1}' || true)

if [[ -n "$STEP_TIME" && -n "$GBS" && "$STEP_TIME" != "0" ]]; then
    SAMPLES_PER_SECOND=$(awk -v gbs="$GBS" -v step="$STEP_TIME" 'BEGIN{printf "%.3f", gbs*1000/step}')
    echo "Elapsed Time Per iteration (ms): $STEP_TIME" | tee -a "$log_path"
    echo "Average Samples per Second: $SAMPLES_PER_SECOND" | tee -a "$log_path"
else
    echo "[WARN] Could not calculate throughput from training log." | tee -a "$log_path"
fi
