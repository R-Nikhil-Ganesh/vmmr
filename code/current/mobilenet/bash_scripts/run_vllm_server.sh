#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server for Qwen3-VL models (default: Qwen/Qwen3-VL-32B-Instruct-FP8)
# Runs in a detached tmux session by default (session name: vllm_server).
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"
source "${SCRIPT_DIR}/lib/tmux_wrap.sh"
run_in_tmux vllm_server "${SCRIPT_DIR}/run_vllm_server.sh" "$@"

MODEL="${VLLM_MODEL:-Qwen/Qwen3-VL-32B-Instruct-FP8}"
PORT="${VLLM_PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"

VLLM_BIN=""
if [ -x "$(dirname "${PYTHON_BIN}")/vllm" ]; then
    VLLM_BIN="$(dirname "${PYTHON_BIN}")/vllm"
elif command -v vllm >/dev/null 2>&1; then
    VLLM_BIN="$(command -v vllm)"
else
    VLLM_BIN="${PYTHON_BIN} -m vllm.entrypoints.openai.api_server"
fi

echo "=================================================================="
echo "  Starting vLLM Server              $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Model:       ${MODEL}"
echo "  Endpoint:    http://${HOST}:${PORT}/v1"
echo "  Max Context: ${MAX_MODEL_LEN}"
echo "  GPU Util:    ${GPU_UTIL}"
echo "=================================================================="

exec ${VLLM_BIN} serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --limit-mm-per-prompt image=1 \
    --gpu-memory-utilization "${GPU_UTIL}" \
    "$@"

