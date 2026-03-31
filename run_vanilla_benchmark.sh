#!/bin/bash
#
# Test & evaluate sanitization model on Privasis-Zero vanilla test set
#
# Two-phase pipeline:
#   Phase 1 - Sanitization: spins up a vLLM server for the testee model,
#             runs run_benchmark.py to generate sanitized texts.
#   Phase 2 - Evaluation:   stops the testee server, starts a vLLM server
#             for the evaluator model, runs evaluate.py.
#
# Downloads nvidia/Privasis-Zero vanilla/test from HuggingFace on first run.
#
# Usage:
#   ./run_vanilla_benchmark.sh
#
#   # Override defaults:
#   MODEL=openai/gpt-oss-120b ./run_vanilla_benchmark.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================
# CONFIGURATION - Override via env vars
# ============================================

# Testee model
MODEL="${MODEL:-openai/gpt-oss-120b}"

# Evaluator model
EVALUATOR_MODEL="${EVALUATOR_MODEL:-openai/gpt-oss-120b}"

# Test data (nvidia/Privasis-Zero from HuggingFace)
HF_SUBSET="${HF_SUBSET:-vanilla}"
HF_SPLIT="${HF_SPLIT:-test}"
SANITIZED_OUTPUT="${SANITIZED_OUTPUT:-${MODEL##*/}_vanilla_test.jsonl}"
EVALUATION_OUTPUT="${EVALUATION_OUTPUT:-${MODEL##*/}_vanilla_test_eval_outputs.jsonl}"

# Generation parameters
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-9024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_EXAMPLES="${NUM_EXAMPLES:-}"

# vLLM server settings
NUM_GPUS="${NUM_GPUS:-8}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

# Extra flags (space-separated, e.g. "--use_smoothed_instruction")
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ============================================
# JOB INFORMATION
# ============================================

echo "=========================================="
echo "  Test Model (Vanilla)"
echo "=========================================="
echo "Testee Model:    ${MODEL}"
echo "Evaluator Model: ${EVALUATOR_MODEL}"
echo "=========================================="
echo "HF Dataset:        nvidia/Privasis-Zero (${HF_SUBSET}/${HF_SPLIT})"
echo "Sanitized Output:  ${SANITIZED_OUTPUT}"
echo "Evaluation Output: ${EVALUATION_OUTPUT}"
echo "=========================================="
echo "Batch Size:   ${BATCH_SIZE}"
echo "Max Length:    ${MAX_LENGTH}"
echo "Temperature:  ${TEMPERATURE}"
echo "Num GPUs:     ${NUM_GPUS}"
if [ -n "${NUM_EXAMPLES}" ]; then
echo "Num Examples: ${NUM_EXAMPLES}"
fi
if [ -n "${MAX_MODEL_LEN}" ]; then
echo "Max Model Len: ${MAX_MODEL_LEN}"
fi
echo "=========================================="
echo ""

# Create directories
mkdir -p outputs/benchmark_predictions/server_logs

# ============================================
# HELPERS
# ============================================

SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
}
trap cleanup EXIT INT TERM

check_server_ready() {
    python -c "
import sys, urllib.request, json
try:
    req = urllib.request.urlopen('http://localhost:${PORT}/v1/models', timeout=5)
    data = json.loads(req.read().decode())
    sys.exit(0 if 'data' in data else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

wait_for_server() {
    local MAX_WAIT=1000
    local WAIT_TIME=0
    local POLL_INTERVAL=3
    local STATUS_INTERVAL=30

    printf "Waiting for server to be ready"
    while ! check_server_ready; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            echo "Error: vLLM server process died unexpectedly. Check: $SERVER_LOG_FILE"
            exit 1
        fi
        if [ $WAIT_TIME -ge $MAX_WAIT ]; then
            echo ""
            echo "Error: Timeout waiting for server after ${MAX_WAIT}s"
            exit 1
        fi
        if [ $((WAIT_TIME % STATUS_INTERVAL)) -eq 0 ] && [ $WAIT_TIME -gt 0 ]; then
            printf " (%ds)" "$WAIT_TIME"
        else
            printf "."
        fi
        sleep $POLL_INTERVAL
        WAIT_TIME=$((WAIT_TIME + POLL_INTERVAL))
    done
    echo ""
    echo "Server ready!"
}

start_vllm_server() {
    local MODEL_PATH="$1"
    local LOG_FILE="$2"

    VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
        --model ${MODEL_PATH} \
        --port ${PORT} \
        --tensor-parallel-size ${NUM_GPUS} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --trust-remote-code"

    if [ -n "${MAX_MODEL_LEN}" ]; then
        VLLM_CMD="${VLLM_CMD} --max-model-len ${MAX_MODEL_LEN}"
    fi

    SERVER_LOG_FILE="$LOG_FILE"
    echo "Launching vLLM server..."
    echo "vLLM command: $VLLM_CMD"
    $VLLM_CMD > "$SERVER_LOG_FILE" 2>&1 &
    SERVER_PID=$!
    echo "vLLM server started (PID: $SERVER_PID)"

    wait_for_server
}

stop_server() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped."
    fi
    SERVER_PID=""
}

# ==========================================
# Detect if testee model is an API model
# ==========================================
is_api_model() {
    local model="$1"
    case "$model" in
        gpt-4*|gpt-5*|o1-*|o3*|nvdev/*|nvidia/*) return 0 ;;
        *) return 1 ;;
    esac
}

# ==========================================
# PHASE 1: Sanitization with testee model
# ==========================================
echo ""
echo "=========================================="
echo "  PHASE 1: Sanitization"
echo "=========================================="

PHASE1_CMD="python run_benchmark.py \
    --model_path '${MODEL}' \
    --evaluation_output_filename '${EVALUATION_OUTPUT}' \
    --batch_size ${BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --temperature ${TEMPERATURE} \
    --evaluator_model_name '${EVALUATOR_MODEL}' \
    --hf_subset '${HF_SUBSET}' \
    --hf_split '${HF_SPLIT}' \
    --skip-evaluation"

if is_api_model "${MODEL}"; then
    echo "API model detected — skipping vLLM server for testee model"
    PHASE1_CMD="${PHASE1_CMD} --no_vllm"
else
    start_vllm_server "${MODEL}" "outputs/benchmark_predictions/server_logs/${EVALUATION_OUTPUT%.jsonl}.log"
    PHASE1_CMD="${PHASE1_CMD} --vllm-server-url 'http://localhost:${PORT}/v1'"
fi

if [ -n "${SANITIZED_OUTPUT}" ]; then
    PHASE1_CMD="${PHASE1_CMD} --sanitized_output_filename '${SANITIZED_OUTPUT}'"
fi
if [ -n "${NUM_EXAMPLES}" ]; then
    PHASE1_CMD="${PHASE1_CMD} --num_examples ${NUM_EXAMPLES}"
fi
PHASE1_EXTRA=$(echo "${EXTRA_ARGS}" | sed 's/--use_existing_sanitized_output//g')
if [ -n "${PHASE1_EXTRA}" ]; then
    PHASE1_CMD="${PHASE1_CMD} ${PHASE1_EXTRA}"
fi

echo "Running Phase 1 (sanitization)..."
eval ${PHASE1_CMD}

echo ""
echo "Phase 1 complete! Sanitized results: outputs/benchmark_predictions/${SANITIZED_OUTPUT}"

# ==========================================
# Stop testee model server (if started)
# ==========================================
if ! is_api_model "${MODEL}"; then
    echo ""
    echo "Stopping testee model server..."
    stop_server

    echo "Waiting for GPU memory to be released..."
    sleep 10
fi

# ==========================================
# PHASE 2: Evaluation with evaluator model
# ==========================================
echo ""
echo "=========================================="
echo "  PHASE 2: Evaluation"
echo "=========================================="
echo "Evaluator Model: ${EVALUATOR_MODEL}"

PHASE2_CMD="python evaluate.py \
    --input-file 'outputs/benchmark_predictions/${SANITIZED_OUTPUT}' \
    --output-file '${EVALUATION_OUTPUT}' \
    --evaluator-model-name '${EVALUATOR_MODEL}'"

if is_api_model "${EVALUATOR_MODEL}"; then
    echo "API model detected — skipping vLLM server for evaluator model"
else
    # Strip hf/ prefix for vLLM
    EVALUATOR_MODEL_PATH="${EVALUATOR_MODEL}"
    if [[ "${EVALUATOR_MODEL}" == hf/* ]]; then
        EVALUATOR_MODEL_PATH="${EVALUATOR_MODEL#hf/}"
    fi

    start_vllm_server "${EVALUATOR_MODEL_PATH}" "outputs/benchmark_predictions/server_logs/${EVALUATION_OUTPUT%.jsonl}_evaluator.log"
    PHASE2_CMD="${PHASE2_CMD} --vllm-server-url 'http://localhost:${PORT}/v1'"
fi

if [ -n "${NUM_EXAMPLES}" ]; then
    PHASE2_CMD="${PHASE2_CMD} --n-records ${NUM_EXAMPLES}"
fi

echo "Running Phase 2 (evaluation)..."
eval ${PHASE2_CMD}

echo ""
echo "=========================================="
echo "Test model pipeline complete!"
echo "Sanitized results: outputs/benchmark_predictions/${SANITIZED_OUTPUT}"
echo "Evaluation results: outputs/benchmark_predictions/${EVALUATION_OUTPUT}"
echo "=========================================="
