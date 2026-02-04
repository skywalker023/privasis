#!/bin/bash
#
# Run generate.py with a vLLM server for parallel processing
#
# This script:
# 1. Starts a vLLM server in the background
# 2. Waits for the server to be ready
# 3. Runs generate.py with parallel workers
# 4. Cleans up the server on exit (including Ctrl+C)
#
# Usage:
#   ./run_generate.sh [options]
#
# Examples:
#   # Basic usage with defaults
#   ./run_generate.sh
#
#   # Custom configuration
#   ./run_generate.sh \
#       --model openai/gpt-oss-120b \
#       --num-gpus 4 \
#       --num-workers 8 \
#       --n-seeds 1000 \
#       --run-id my-experiment
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MODEL="openai/gpt-oss-120b"
NUM_GPUS=8
PORT=8009
NUM_WORKERS=5
N_SEEDS=100
RUN_ID="privasis--$(date +%Y%m%d-%H%M%S)"
GPU_MEM_UTIL=0.8
MAX_MODEL_LEN=""
EXTRA_ARGS=""
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-0.6B"

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model MODEL           Model name/path (default: openai/gpt-oss-120b)"
    echo "  --num-gpus N            Number of GPUs for tensor parallelism (default: 1)"
    echo "  --port PORT             Port for vLLM server (default: 8000)"
    echo "  --num-workers N         Number of parallel worker threads (default: 4)"
    echo "  --n-seeds N             Number of seeds to generate (default: 10)"
    echo "  --run-id ID             Run ID for output files (default: timestamped)"
    echo "  --gpu-mem-util FLOAT    GPU memory utilization 0.0-1.0 (default: 0.9)"
    echo "  --max-model-len N       Maximum model context length (optional)"
    echo "  --print                 Print generated outputs to console"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --num-gpus 4 --num-workers 8 --n-seeds 1000"
    echo "  $0 --model meta-llama/Llama-3.1-8B-Instruct --num-gpus 2"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"
            shift 2
            ;;
        --run-id)
            RUN_ID="$2"
            shift 2
            ;;
        --gpu-mem-util)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --print)
            EXTRA_ARGS="$EXTRA_ARGS --print"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Server PID (will be set when server starts)
SERVER_PID=""

# Cleanup function - kills the server on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}

# Set trap to call cleanup on exit, interrupt, or termination
trap cleanup EXIT INT TERM

echo "=========================================="
echo "  Parallel Generation with vLLM Server"
echo "=========================================="
echo "Model: $MODEL"
echo "Number of GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "Number of Workers: $NUM_WORKERS"
echo "Number of Seeds: $N_SEEDS"
echo "Run ID: $RUN_ID"
echo "GPU Memory Utilization: $GPU_MEM_UTIL"
if [[ -n "$MAX_MODEL_LEN" ]]; then
    echo "Max Model Length: $MAX_MODEL_LEN"
fi
echo "=========================================="
echo ""

# Build vLLM server command
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --tensor-parallel-size $NUM_GPUS \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --trust-remote-code"

if [[ -n "$MAX_MODEL_LEN" ]]; then
    VLLM_CMD="$VLLM_CMD --max-model-len $MAX_MODEL_LEN"
fi

# Setup server log file
SERVER_LOG_DIR="$SCRIPT_DIR/outputs/privasis/server_logs"
mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG_FILE="$SERVER_LOG_DIR/${RUN_ID}.log"

# Start vLLM server in background with logging
echo "Starting vLLM server (logs: $SERVER_LOG_FILE)..."
$VLLM_CMD > "$SERVER_LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (quietly, with periodic status updates)
MAX_WAIT=600  # Maximum wait time in seconds (10 minutes)
WAIT_TIME=0
POLL_INTERVAL=3
STATUS_INTERVAL=30  # Only print status every 30 seconds

# Function to check if server is ready using Python (since curl may not be available)
check_server_ready() {
    python -c "
import sys
import urllib.request
import json
try:
    req = urllib.request.urlopen('http://localhost:${PORT}/v1/models', timeout=5)
    data = json.loads(req.read().decode())
    sys.exit(0 if 'data' in data else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# Use /v1/models endpoint - it only succeeds when the model is fully loaded
printf "Waiting for server to be ready"
while ! check_server_ready; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "Error: vLLM server process died unexpectedly. Check: $SERVER_LOG_FILE"
        exit 1
    fi
    
    if [[ $WAIT_TIME -ge $MAX_WAIT ]]; then
        echo ""
        echo "Error: Timeout waiting for server to be ready"
        exit 1
    fi
    
    # Print dots to show progress, status every STATUS_INTERVAL seconds
    if [[ $((WAIT_TIME % STATUS_INTERVAL)) -eq 0 ]] && [[ $WAIT_TIME -gt 0 ]]; then
        printf " (%ds)" "$WAIT_TIME"
    else
        printf "."
    fi
    sleep $POLL_INTERVAL
    WAIT_TIME=$((WAIT_TIME + POLL_INTERVAL))
done

echo ""
echo "Server ready! Starting generation..."
echo ""

cd "$SCRIPT_DIR"

python generate.py \
    --generator-model "$MODEL" \
    --vllm-server-url "http://localhost:$PORT/v1" \
    --num-workers "$NUM_WORKERS" \
    --n_seeds "$N_SEEDS" \
    --run-id "$RUN_ID" \
    --embedding-model "$EMBEDDING_MODEL" \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Generation complete!"
echo "Output saved to: outputs/privasis/${RUN_ID}.jsonl"
echo "Server logs saved to: $SERVER_LOG_FILE"
echo "=========================================="
