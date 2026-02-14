#!/bin/bash
#
# Start vLLM server with OpenAI-compatible API
#
# Usage:
#   ./start_vllm_server.sh <model_name> [num_gpus] [port] [gpu_memory_utilization]
#
# Examples:
#   # Start with default settings (1 GPU, port 8000)
#   ./start_vllm_server.sh openai/gpt-oss-120b
#
#   # Start with 4 GPUs on port 8080
#   ./start_vllm_server.sh openai/gpt-oss-120b 4 8080
#
#   # Start with custom memory utilization
#   ./start_vllm_server.sh openai/gpt-oss-120b 1 8000 0.95
#

set -e

MODEL_NAME=${1:-"openai/gpt-oss-120b"}
NUM_GPUS=${2:8}
PORT=${3:-8009}
GPU_MEMORY_UTILIZATION=${4:-0.8}

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Number of GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "=========================================="
echo ""
echo "Once the server is running, you can use it with:"
echo "  python generate.py \\"
echo "    --generator-model $MODEL_NAME \\"
echo "    --vllm-server-url http://localhost:$PORT/v1 \\"
echo "    --num-workers 8 \\"
echo "    --n_seeds 1000"
echo ""
echo "=========================================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$NUM_GPUS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --port "$PORT" \
    --trust-remote-code
