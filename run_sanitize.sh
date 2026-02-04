#!/bin/bash
#
# Run sanitize.py with a vLLM server for parallel processing
#
# This script:
# 1. Starts a vLLM server in the background
# 2. Waits for the server to be ready
# 3. Runs sanitize.py with parallel workers
# 4. Cleans up the server on exit (including Ctrl+C)
#
# Usage:
#   ./run_sanitize.sh [options]
#
# Examples:
#   # Basic usage with defaults
#   ./run_sanitize.sh --privasis-data-id my-dataset
#
#   # Custom configuration
#   ./run_sanitize.sh \
#       --privasis-data-id my-dataset \
#       --model openai/gpt-oss-120b \
#       --num-gpus 4 \
#       --num-workers 8 \
#       --run-id sanitized-v1
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
MODEL="openai/gpt-oss-120b"
NUM_GPUS=8
PORT=8009
NUM_WORKERS=2
PRIVASIS_DATA_ID="[the name of the run id from the generation phase]"
RUN_ID="sanitized--$(date +%Y%m%d-%H%M%S)"
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=""
RETRY_LIMIT=1
ATTR_WEIGHTING="sensitivity"
EXTRA_ARGS=""

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Required:"
    echo "  --privasis-data-id ID   Run ID from generation phase (loads from outputs/privasis/)"
    echo ""
    echo "Options:"
    echo "  --model MODEL           Model name/path (default: openai/gpt-oss-120b)"
    echo "  --num-gpus N            Number of GPUs for tensor parallelism (default: 1)"
    echo "  --port PORT             Port for vLLM server (default: 8000)"
    echo "  --num-workers N         Number of parallel worker threads (default: 4)"
    echo "  --run-id ID             Run ID for output files (default: timestamped)"
    echo "  --gpu-mem-util FLOAT    GPU memory utilization 0.0-1.0 (default: 0.9)"
    echo "  --max-model-len N       Maximum model context length (optional)"
    echo "  --retry-limit N         Number of retries if sanitization fails (default: 1)"
    echo "  --attr-weighting TYPE   Attribute selection: uniform or sensitivity (default: sensitivity)"
    echo "  --print                 Print sanitized outputs to console"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --privasis-data-id my-dataset --num-gpus 4 --num-workers 8"
    echo "  $0 --privasis-data-id my-dataset --model meta-llama/Llama-3.1-8B-Instruct"
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
        --privasis-data-id)
            PRIVASIS_DATA_ID="$2"
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
        --retry-limit)
            RETRY_LIMIT="$2"
            shift 2
            ;;
        --attr-weighting)
            ATTR_WEIGHTING="$2"
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

# Check required arguments
if [[ -z "$PRIVASIS_DATA_ID" ]]; then
    echo "Error: --privasis-data-id is required"
    echo "Use --help to see available options"
    exit 1
fi

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
echo "  Parallel Sanitization with vLLM Server"
echo "=========================================="
echo "Privasis Data ID: $PRIVASIS_DATA_ID"
echo "Model: $MODEL"
echo "Number of GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "Number of Workers: $NUM_WORKERS"
echo "Run ID: $RUN_ID"
echo "GPU Memory Utilization: $GPU_MEM_UTIL"
echo "Retry Limit: $RETRY_LIMIT"
echo "Attribute Weighting: $ATTR_WEIGHTING"
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
SERVER_LOG_DIR="$SCRIPT_DIR/outputs/sanitized_privasis/server_logs"
mkdir -p "$SERVER_LOG_DIR"
SERVER_LOG_FILE="$SERVER_LOG_DIR/${PRIVASIS_DATA_ID}_${RUN_ID}.log"

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
echo "Server ready! Starting sanitization..."
echo ""

cd "$SCRIPT_DIR"

python sanitize.py \
    --privasis-data-id "$PRIVASIS_DATA_ID" \
    --sanitization-model "$MODEL" \
    --vllm-server-url "http://localhost:$PORT/v1" \
    --num-workers "$NUM_WORKERS" \
    --run-id "$RUN_ID" \
    --retry-limit "$RETRY_LIMIT" \
    --attr-selection-weighting "$ATTR_WEIGHTING" \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Sanitization complete!"
echo "Output saved to: outputs/sanitized_privasis/${PRIVASIS_DATA_ID}_${RUN_ID}.jsonl"
echo "Server logs saved to: $SERVER_LOG_FILE"
echo "=========================================="
