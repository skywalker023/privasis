# 🏝️ Privasis

Welcome! This is the official repository for our paper ["Privasis: Synthesizing the Largest 'Public' Private Dataset from Scratch"](https://arxiv.org/abs/2602.03183). Please visit our website for an [overview of the project](https://privasis.github.io). We will soon release our dataset and models. Stay tuned!

## Environment Setup

```bash
conda env create -f environment.yml; conda activate privasis
huggingface-cli login
```

---

## Quick Start

You can use one of the shell scripts for easy launch
```
./run_generate.sh  # builds Privasis
./run_sanitize.sh  # builds Privasis-Sanitization on top of the generated Privasis. You first need to generate some Privasis data
```

This section covers the two main ways to run models for the Privasis pipeline:
1. **OpenAI API** - simplest setup, pay-per-use
2. **vLLM Server** (local models) - run models on your own GPUs

### Option 1: OpenAI API

Set your API key and run directly:

```bash
export OPENAI_API_KEY="your-api-key"

# Generate Privasis with GPT-4.1
python generate.py \
    --run-id my-run \
    --n_seeds 100 \
    --generator-model gpt-4.1

# Generate Privasis-Sanitization with GPT-4.1 on the generated Privasis data
python sanitize.py \
    --privasis-data-id my-run \
    --run-id my-run-sanitized \
    --sanitization-model gpt-4.1
```

### Option 2: vLLM Server (Local Models)

For local models, start a vLLM server and use parallel workers for optimal throughput.

**Step 1: Start the vLLM server**

```bash
# Basic usage (1 GPU, port 8000)
./start_vllm_server.sh openai/gpt-oss-120b

# With 4 GPUs on port 8080
./start_vllm_server.sh openai/gpt-oss-120b 4 8080

# With custom memory utilization
./start_vllm_server.sh openai/gpt-oss-120b 1 8000 0.95
```

**Step 2: Run generation with parallel workers**

```bash
python generate.py \
    --generator-model openai/gpt-oss-120b \
    --vllm-server-url http://localhost:8000/v1 \
    --num-workers 8 \
    --n_seeds 1000 \
    --run-id parallel-generation
```

**Step 3: Run sanitization with parallel workers**

```bash
python sanitize.py \
    --privasis-data-id parallel-generation \
    --sanitization-model openai/gpt-oss-120b \
    --vllm-server-url http://localhost:8000/v1 \
    --num-workers 8 \
    --run-id parallel-sanitization
```

#### vLLM Server Architecture

```
┌─────────────────────────────────────────────────────────┐
│  vLLM Server (single instance, holds model in GPU)      │
│  http://localhost:8000/v1                               │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP requests
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │ Thread 1│         │ Thread 2│         │ Thread N│
   └─────────┘         └─────────┘         └─────────┘
                            │
                    ┌───────▼───────┐
                    │ Single output │
                    │ file (locked) │
                    └───────────────┘
```

#### Tips for vLLM Server

- **Number of workers**: Start with 4-8 workers and adjust based on GPU utilization
- **Memory**: Reduce `--gpu_memory_utilization` if you see OOM errors
- **Model name consistency**: Use the same model name for both the server and client scripts
- **Monitoring**: Watch the vLLM server output for request queuing and throughput metrics

---

## Supported Providers

| Provider | Model Patterns | Environment Variable |
|----------|---------------|---------------------|
| **OpenAI** | `gpt-4*`, `gpt-5*`, `o1-*`, `o3*` | `OPENAI_API_KEY` |
| **NVIDIA NIM** | `nvdev/*` | `NVDEV_API_KEY` |
| **vLLM Server** | Any model + `--vllm-server-url` parameter | N/A |

---

## Pipeline Overview

The pipeline has two stages:
1. **Generate Privasis** (`generate.py`) - Generate synthetic records with PII
2. **Sanitize Privasis** (`sanitize.py`) - Abstract or remove target information

### 1. Build Privasis (`generate.py`)

Generates synthetic records containing rich privacy-sensitive information

#### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--run-id` | `v0` | Run ID (used as output filename under `outputs/privasis/`) |
| `--n_seeds` | `10` | Number of seeds to generate profiles for |
| `--generator-model` | **(required)** | Model to use for generation |
| `--seeds_path` | `None` | Path to custom seeds file for profile generation |
| `--num_events` | `1` | Number of events per profile |
| `--tolerance` | `3` | Number of retries before accepting failure |
| `--print` | `False` | Print generated outputs to console |
| `--vllm-server-url` | `None` | URL of vLLM server (e.g., `http://localhost:8000/v1`) |
| `--num-workers` | `1` | Number of parallel worker threads |
| `--embedding-model` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model for diversity scoring (HuggingFace or OpenAI) |
| `--embedding-device` | `cuda` | Device for HF embedding model (`cpu`, `cuda`, `cuda:0`, etc.) |

> **Note**: The embedding model is used for computing diversity scores (Vendi score) during Metropolis-Hastings sampling. Supported models include HuggingFace models (e.g., `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Embedding-4B`) or OpenAI models (e.g., `text-embedding-3-small`).

#### Output

Results are saved to `outputs/privasis/{run-id}/`:
- `{run-id}.jsonl` - Generated records with PII

---

### 2. Build Privasis-Sanitization (`sanitize.py`)

Sanitizes (abstracts or removes) the target information in the generated records.

#### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--privasis-data-id` | `mark07` | Run ID from generation phase |
| `--run-id` | `v0` | Run ID for this sanitization run |
| `--sanitization-model` | `openai/gpt-oss-120b` | Model to use for sanitization |
| `--retry-limit` | `1` | Number of retries if sanitization fails |
| `--attr-selection-weighting` | `sensitivity` | Attribute selection strategy (`uniform` or `sensitivity`) |
| `--print` | `False` | Print sanitized outputs to console |
| `--vllm-server-url` | `None` | URL of vLLM server |
| `--num-workers` | `1` | Number of parallel worker threads |

#### Output

Results are saved to `outputs/sanitized_privasis/`:
- `{privasis-data-id}_{run-id}.jsonl` - Sanitized records
- `{privasis-data-id}_{run-id}_errors.jsonl` - Failed instances (if any)

---

## Evaluation

The evaluation pipeline measures how effectively a sanitization model removes or abstracts private information. It uses a two-phase approach: (1) the testee model sanitizes benchmark texts, then (2) an evaluator model attempts to infer the original PII from the sanitized output via string matching and LLM-based inference attacks.

### Quick Start

Use the provided shell scripts to run the full test-and-evaluate pipeline on the [Privasis-Zero](https://huggingface.co/datasets/nvidia/Privasis-Zero) benchmark:

```bash
# Test on the vanilla benchmark
./run_vanilla_benchmark.sh

# Test on the hard benchmark
./run_hard_benchmark.sh

# Override defaults via environment variables
MODEL=my-model NUM_GPUS=4 ./run_vanilla_benchmark.sh
```

These scripts automatically handle vLLM server lifecycle (start/stop) for both the testee and evaluator models. API models (e.g., `gpt-4*`) are detected automatically and skip the vLLM server.

### Step-by-Step Usage

#### Step 1: Generate sanitized outputs with the testee model

```bash
python run_benchmark.py \
    --model_path openai/gpt-oss-120b \
    --hf_subset vanilla \
    --hf_split test \
    --sanitized_output_filename my_model_vanilla_test.jsonl \
    --evaluation_output_filename my_model_vanilla_test_eval.jsonl \
    --vllm-server-url http://localhost:8000/v1 \
    --batch_size 16 \
    --skip-evaluation
```

#### Step 2: Evaluate sanitization effectiveness

```bash
python evaluate.py \
    --input-file outputs/benchmark_predictions/my_model_vanilla_test.jsonl \
    --output-file my_model_vanilla_test_eval.jsonl \
    --evaluator-model-name gpt-4.1
```

### Benchmark Arguments (`run_benchmark.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | **(required)** | Path to the sanitization model |
| `--test_data_path` | `None` | Path to local test data JSONL file |
| `--hf_subset` | `None` | Privasis-Zero dataset subset (`vanilla` or `hard`) |
| `--hf_split` | `test` | Privasis-Zero dataset split (`test` or `validation`) |
| `--sanitized_output_filename` | `None` | Filename for sanitized results |
| `--evaluation_output_filename` | **(required)** | Filename for evaluation results |
| `--batch_size` | `1` | Batch size for inference |
| `--max_length` | `9024` | Maximum generation length |
| `--temperature` | `0.0` | Sampling temperature |
| `--no_vllm` | `False` | Disable vLLM, use HuggingFace Transformers |
| `--num_gpus` | `1` | Number of GPUs for vLLM tensor parallelism |
| `--vllm-server-url` | `None` | URL of external vLLM server |
| `--evaluator_model_name` | `hf/openai/gpt-oss-120b` | Evaluator model name |
| `--evaluator-vllm-server-url` | `None` | URL of vLLM server for evaluator |
| `--skip-evaluation` | `False` | Only run sanitization, skip evaluation |
| `--num_examples` | `None` | Limit number of examples (for quick testing) |
| `--use_smoothed_instruction` | `False` | Use smoothed sanitization instructions |

### Evaluate Arguments (`evaluate.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | `None` | Path to sanitized results JSONL (or use `--hf-subset`) |
| `--hf-subset` | `None` | Privasis-Zero dataset subset (`vanilla` or `hard`) |
| `--hf-split` | `test` | Privasis-Zero dataset split (`test` or `validation`) |
| `--output-file` | **(required)** | Path to save evaluation report |
| `--evaluator-model-name` | `nvdev/openai/gpt-oss-120b` | LLM model for evaluation |
| `--vllm-server-url` | `None` | URL of vLLM server |
| `--n-records` | `None` | Number of records to evaluate |
| `--faster` | `False` | Use only exact string matching (no LLM inference) |
| `--resume` | `False` | Resume from existing checkpoint |


### Shell Script Configuration

Both `run_vanilla_benchmark.sh` and `run_hard_benchmark.sh` accept the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `openai/gpt-oss-120b` | Testee model |
| `EVALUATOR_MODEL` | `openai/gpt-oss-120b` | Evaluator model |
| `NUM_GPUS` | `8` | Number of GPUs for vLLM |
| `BATCH_SIZE` | `16` (vanilla) / `8` (hard) | Batch size |
| `PORT` | `8000` | vLLM server port |
| `NUM_EXAMPLES` | *(all)* | Limit number of examples |
| `MAX_MODEL_LEN` | *(auto)* | Maximum model context length |
| `EXTRA_ARGS` | *(none)* | Additional flags for `run_benchmark.py` |


---

## BibTeX

Please cite our work if you find this repo useful.

```bib
@article{kim2026privasis,
    title={Privasis: Synthesizing the Largest 'Public' Private Dataset from Scratch},
    author={Kim, Hyunwoo and Mireshghallah, Niloofar and Duan, Michael and Xin, Rui and Li, Shuyue Stella and Jung, Jaehun and Acuna, David and Pang, Qi and Xiao, Hanshen and Suh, G. Edward and Oh, Sewoong and Tsvetkov, Yulia and Koh, Pang Wei and Choi, Yejin},
    booktitle ={arXiv preprint arXiv:2602.03183},
    year=2026
}
```