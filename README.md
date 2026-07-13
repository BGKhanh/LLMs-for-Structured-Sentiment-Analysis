# Benchmarking Large Language Models for Structured Sentiment Analysis

---

A research repository for benchmarking Large Language Models (LLMs) on Structured Sentiment Analysis (SSA) using **LM Evaluation Harness**. The repository provides benchmark tasks, configurable prompting strategies, support for multiple inference backends, and reproducible experimental settings for evaluating LLMs across multilingual SSA datasets.

## Overview

Structured Sentiment Analysis (SSA) extends traditional sentiment analysis by extracting complete opinion structures, including opinion holders, opinion targets, sentiment expressions, and their corresponding polarities. Although recent Large Language Models (LLMs) have demonstrated impressive capabilities in structured information extraction, evaluating their performance on SSA remains challenging due to complex structured outputs, diverse annotation schemes across datasets, and the lack of standardized evaluation protocols.

This repository extends **LM Evaluation Harness** by providing benchmark tasks, prompt templates, output parsers, and evaluation configurations for Structured Sentiment Analysis. It is designed to facilitate reproducible benchmarking of LLMs across multiple datasets, languages, and inference backends while maintaining a consistent evaluation workflow.

The repository supports configurable prompting strategies, structured output validation, multilingual benchmark datasets, and reproducible experimental settings. It is compatible with multiple inference backends—including **vLLM**, **llama.cpp**, and OpenAI-compatible APIs—allowing experiments to be conducted consistently on both high-performance servers and local deployments.

### Key Features

* Structured Sentiment Analysis benchmark tasks built on **LM Evaluation Harness**.
* Support for multilingual SSA datasets.
* Multiple inference backends, including **vLLM**, **llama.cpp**, and OpenAI-compatible APIs.
* Configurable prompt templates and inference settings.
* Automatic parsing and validation of structured outputs.
* Reproducible experiment configurations with Docker support.
* Modular task organization for extending datasets, prompts, and evaluation settings.

This repository is intended for researchers and practitioners who wish to evaluate, compare, and analyze the capabilities of Large Language Models on Structured Sentiment Analysis under consistent and reproducible experimental settings.

---

## 📂 Repository Structure

The repository is organized as follows:

```text
.
├── data/                               # Original and processed SSA datasets
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   └── vitoed_new/                     # Cleaned and corrected dataset version
│
├── divided_data/                       # Dataset partitions for parallel inference
│
├── semeval22_structured_sentiment/     # SemEval-2022 Task 10 multilingual datasets and preprocessing scripts
│   ├── data/
│   └── process_*.py
│
├── notebook/                           # Example notebooks and experimental workflows
│   ├── Gemma3_(4B).ipynb
│   ├── dspy_GEPA.ipynb
│   ├── dspy_GEPA_local_host_api.ipynb
│   └── vastai_notebook_lm_eval.ipynb
│
├── src/
│   └── tasks/                          # SSA task definitions for LM Evaluation Harness
│
├── Dockerfile                          # Docker environment for reproducible experiments
├── requirements.txt                    # Python dependencies
└── .gitignore
```

### Directory Overview

| Directory                         | Description                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `data/`                           | Original and processed Structured Sentiment Analysis datasets.                                        |
| `divided_data/`                   | Dataset partitions for parallel or distributed inference.                                             |
| `semeval22_structured_sentiment/` | Multilingual SSA benchmark datasets and preprocessing utilities based on SemEval-2022.                |
| `notebook/`                       | Example notebooks for model inference, prompt optimization, and experiment workflows.                 |
| `src/tasks/`                      | Task definitions, prompt templates, parsers, and evaluation configurations for LM Evaluation Harness. |
| `Dockerfile`                      | Docker configuration for creating reproducible execution environments.                                |
| `requirements.txt`                | Python package dependencies required by the repository.                                               |

## Installation

This repository supports two installation methods:

* **Docker (Recommended):** A pre-built Docker image with all required dependencies pre-installed for reproducible experiments.
* **Local Environment:** Install the required dependencies manually on your system.

---

### Option 1. Docker (Recommended)

A pre-built Docker image is available on Docker Hub:

```bash
docker pull kgb0630/lm-eval-vastai:v6
```

Launch the container with GPU support:

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    kgb0630/lm-eval-vastai:v6
```

The Docker image includes the runtime environment required for running experiments in this repository and is the recommended option for reproducible benchmarking.

---

### Option 2. Local Environment

Clone this repository:

```bash
git clone https://github.com/BGKhanh/LLMs-for-Structured-Sentiment-Analysis.git
cd LLMs-for-Structured-Sentiment-Analysis
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

#### Install LM Evaluation Harness

This project is built on top of **LM Evaluation Harness**. Please follow the official installation instructions provided by the project:

* **Repository:** https://github.com/EleutherAI/lm-evaluation-harness

The recommended installation method from the official repository is:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git

cd lm-evaluation-harness

pip install -e ".[api,vllm]"
```

> Additional installation options (e.g., `hf`, `multimodal`, `dev`) are available in the official LM Evaluation Harness documentation.

---

### Optional: FlashAttention

FlashAttention is optional but recommended for supported NVIDIA GPUs to improve inference throughput.

```bash
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE 
pip install <flash-attention-wheel>
```

Please refer to the official projects for installation instructions and compatible pre-built wheels:

* **FlashAttention:** https://github.com/Dao-AILab/flash-attention
* **Pre-built Wheels:** https://github.com/mjun0812/flash-attention-prebuild-wheels

Since FlashAttention depends on your **CUDA**, **PyTorch**, **Python**, and **GPU architecture**, it is recommended to follow the compatibility matrix provided by the official repositories rather than using fixed installation commands.

---

### Verify the Installation

Verify that LM Evaluation Harness has been installed successfully:

```bash
lm_eval --help
```

If the installation is successful, the command above will display the available command-line options.

### Requirements

| Component | Version |
|----------|---------|
| Python | >= 3.12 |
| CUDA | >= 13.0 (recommended) |
| Docker | Optional |
| NVIDIA Driver | Compatible with installed CUDA |

## Running Evaluation

This repository supports two workflows for running benchmark experiments:

* **Notebook Workflow (Recommended):** Execute the provided Jupyter notebook, which includes the environment setup and example evaluation commands.
* **Command-Line Workflow:** Run `lm_eval` directly from the terminal for scripting, automation, or large-scale experiments.

Both workflows are built on **LM Evaluation Harness** and support two inference modes:

* **Native vLLM backend**, where `lm_eval` loads the model directly.
* **OpenAI-compatible API**, where `lm_eval` communicates with an external inference server (e.g., **llama-server**, **vLLM Serve**, **SGLang**, or **Text Generation Inference**).

---

### Starting an OpenAI-Compatible Inference Server

This step is only required when using the `local-chat-completions` backend.

#### vLLM OpenAI Server

Launch a vLLM server exposing an OpenAI-compatible API:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <MODEL_NAME_OR_HF_REPO> \
    --host 127.0.0.1 \
    --port 8000 \
    --gpu-memory-utilization 0.975 \
    --max-model-len 16384 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser <PARSER_NAME>
```

Example:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model nvidia/Gemma-4-26B-A4B-NVFP4 \
    --host 127.0.0.1 \
    --port 8000 \
    --gpu-memory-utilization 0.975 \
    --max-model-len 16384 \
    --default-chat-template-kwargs '{"enable_thinking": true}' \
    --reasoning-parser gemma4
```

#### llama.cpp OpenAI Server

Launch `llama-server` with a local GGUF model:

```bash
llama-server \
    -m <MODEL_PATH.gguf> \
    --host 127.0.0.1 \
    --port 8000 \
    -c 640000 \
    -fa on \
    --reasoning on \
    --reasoning-format deepseek \
    -ngl 99 \
    -np 128 \
    -cb
```

> **Note**
>
> The provided Docker image (`kgb0630/lm-eval-vastai:v6`) includes `llama-server`, allowing GGUF models to be served through an OpenAI-compatible API without additional installation.

---

### Option 1. Notebook Workflow (Recommended)

Example notebook:

```text
notebook/
└── vastai_notebook_lm_eval.ipynb
```

The notebook contains the environment setup together with example `lm_eval` commands for reproducing the benchmark experiments.

---

### Option 2. Command-Line Workflow

#### Native vLLM Backend

Use the native vLLM integration provided by **LM Evaluation Harness**.

```bash
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
lm_eval \
  --model vllm \
  --model_args pretrained=<MODEL_NAME>,max_model_len=16384,gpu_memory_utilization=0.95 \
  --tasks <TASK_NAME> \
  --include_path ./src/tasks \
  --apply_chat_template \
  --log_samples \
  --batch_size 32 \
  --output_path results/lm_eval/<experiment_name> \
  --metadata '{"technique":"few_shot","language":"vi","n_shot":3}' \
  --confirm_run_unsafe_code
```

#### OpenAI-Compatible API

Use this command after starting an OpenAI-compatible inference server.

```bash
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
lm_eval \
  --model local-chat-completions \
  --model_args model=<MODEL_NAME>,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_length=16384 \
  --tasks <TASK_NAME> \
  --include_path ./src/tasks \
  --apply_chat_template \
  --log_samples \
  --output_path results/lm_eval/<experiment_name> \
  --metadata '{"technique":"few_shot","language":"vi","n_shot":3}' \
  --confirm_run_unsafe_code
```

Replace `<TASK_NAME>` with any benchmark task provided by this repository (e.g., `vietnamese_ssa`).

## Supported Datasets

This repository currently supports benchmark datasets from both public Structured Sentiment Analysis benchmarks and custom datasets. Most multilingual datasets are adapted from the official **SemEval-2022 Task 10** benchmark, while additional datasets can be integrated through the task definitions provided in this repository.

| Dataset | Language(s) | Source | Description | Status |
|----------|-------------|--------|-------------|:------:|
| ViToED | Vietnamese | Internal | Structured sentiment dataset for Vietnamese | ✅ |
| NoReC Fine | Norwegian | SemEval-2022 | Review domain | ✅ |
| OpeNER | English, Spanish | SemEval-2022 | Hotel reviews | ✅ |
| MultiBooked | Catalan, Basque | SemEval-2022 | Hotel reviews | ✅ |

> **Note**
>
> The multilingual benchmark datasets and their evaluation protocol are adapted from the official **SemEval-2022 Task 10: Structured Sentiment Analysis** repository. Dataset preprocessing scripts are available under `semeval22_structured_sentiment/`.

---

## Evaluation Metrics

Model predictions are evaluated following the official **SemEval-2022 Task 10** evaluation protocol.

Each predicted opinion is represented as a structured tuple consisting of:

- **Holder**
- **Target**
- **Sentiment Expression**
- **Polarity**

The evaluation compares the predicted opinion structures against the gold annotations and reports the following metrics:

| Metric | Description |
|---------|-------------|
| **Precision** | Fraction of predicted opinion structures that are correct. |
| **Recall** | Fraction of gold opinion structures successfully recovered by the model. |
| **Sentiment Graph F1** | Official SemEval evaluation metric measuring the overall quality of structured sentiment prediction. |

Unlike traditional sentiment classification, Structured Sentiment Analysis requires correctly predicting both the sentiment polarity and the relationships among opinion components. Consequently, the official evaluation is performed on complete opinion graphs rather than isolated spans.

### Evaluation Pipeline

```text
Gold Opinion Graphs
          │
          ▼
Model Predictions
          │
          ▼
Official SemEval Evaluation Script
          │
          ▼
 Precision
 Recall
 Sentiment Graph F1
```

The evaluation implementation follows the official **SemEval-2022 Task 10** benchmark to ensure fair and reproducible comparison with previously published methods.