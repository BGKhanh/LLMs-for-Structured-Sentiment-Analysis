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
│   ├── tasks/
│   │   └── vietnamese_ssa/             # SSA task definitions for LM Evaluation Harness
│   │       ├── _common_yaml            # Shared config (metrics, filters, generation_kwargs)
│   │       ├── few_shot.yaml           # tag: vietnamese_ssa
│   │       ├── re_reading.yaml         # tag: vietnamese_ssa
│   │       ├── few_shot_cot.yaml       # tag: vietnamese_ssa
│   │       ├── plan_and_solve.yaml     # tag: vietnamese_ssa
│   │       └── utils.py                # load_dataset, doc_to_text, process_results, metric aggregation
│   ├── prompt_templates/               # Reusable prompt-building blocks (system prompts, example pools)
│   └── utils/                          # Response postprocessing / structured-output extraction
│
├── results/                            # lm_eval output (results_*.json, samples_*.jsonl)
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
| `src/tasks/`                      | Per-technique task configs for LM Evaluation Harness — one `.yaml` per prompting technique, grouped under `tag: vietnamese_ssa`; sample selection for few-shot is handled natively by the harness (`fewshot_config`), not hardcoded in Python. |
| `src/prompt_templates/`           | Stateless, reusable prompt-building blocks (system prompts per language, hand-written CoT example pools) shared across techniques.                    |
| `src/utils/`                      | Postprocessing utilities for parsing and validating structured model outputs.                         |
| `results/`                        | Evaluation outputs (`results_*.json` summaries, `samples_*.jsonl` per-sample logs).                    |
| `Dockerfile`                      | Docker configuration for creating reproducible execution environments.                                |
| `requirements.txt`                | Python package dependencies required by the repository.                                               |

## Installation

This repository supports two installation methods:

* **Docker (Recommended):** A pre-built Docker image with the core runtime (CUDA, PyTorch, `llama-server`) ready to use. `lm-eval` is installed at container **runtime** (not baked into the image) so its version can be pinned per experiment 
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

The Docker image includes the base runtime environment (CUDA, PyTorch, `llama-server` binary) required for running experiments in this repository. 
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

> **Note — Gated models**
>
> If the model you are downloading (HF repo) is gated, set one of `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HUGGINGFACE_API_KEY` as an environment variable before starting the server, e.g. `export HF_TOKEN=<your_token>`.

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
  --num_fewshot 3 \
  --fewshot_random_seed 42 \
  --apply_chat_template \
  --log_samples \
  --batch_size 32 \
  --output_path results/lm_eval/<experiment_name> \
  --metadata '{"language":"vi"}' \
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
  --num_fewshot 3 \
  --fewshot_random_seed 42 \
  --apply_chat_template \
  --log_samples \
  --output_path results/lm_eval/<experiment_name> \
  --metadata '{"language":"vi"}' \
  --confirm_run_unsafe_code
```

Replace `<TASK_NAME>` with either `vietnamese_ssa` (runs all four techniques below in one call, each scored separately) or one specific technique task:

| `<TASK_NAME>` | Technique | Notes |
|---|---|---|
| `vietnamese_ssa_few_shot` | Plain few-shot | `--num_fewshot` sampled live from `train` split (seeded) |
| `vietnamese_ssa_re_reading` | Re-reading | Question repeated twice; same live sampling as above |
| `vietnamese_ssa_few_shot_cot` | Few-shot Chain-of-Thought | Examples come from a small hand-written pool (`src/prompt_templates/shared.py`), not `train` — max `--num_fewshot` is bounded by that pool's size |
| `vietnamese_ssa_plan_and_solve` | Plan-and-Solve | Instruction-only, no examples — leave `--num_fewshot 0` (default); pass `--metadata '{"language":"vi","plus_mode":true}'` for the PS+ variant |

## ⚠️ Troubleshooting: `ValueError: Tasks not found: <task_name>`

If `lm_eval` reports a task as "not found" even though its `.yaml` clearly
exists under `--include_path`, the cause is almost always a **silently
skipped file**, not a missing task. When lm-eval-harness scans
`--include_path` to build its task index, any YAML file that fails to parse
(missing `include:` target, malformed key, etc.) is dropped **without a
visible error** — it only logs at `DEBUG` level, which isn't shown by
default. To reveal the real reason, run:

```bash
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from lm_eval.tasks import TaskManager
tm = TaskManager(include_path='./src/tasks')
print('found:', 'vietnamese_ssa_few_shot' in tm.all_tasks)
" 2>&1 | grep -i "skip.*vietnamese_ssa"
```

The most common root cause for this repo specifically: `_common_yaml`
(shared config included by every technique `.yaml` via `include:
_common_yaml`) is deliberately named **without** a `.yaml` extension, so it
isn't picked up as a standalone task by the file scanner. Some tools (file
browsers, certain upload/download flows) silently append an extension when
saving a dot-less filename — if that happens, every technique task in
`src/tasks/vietnamese_ssa/` will fail to index. Verify with:

```bash
ls -la src/tasks/vietnamese_ssa/
# must show a file literally named `_common_yaml` (no extension),
# in the same directory as few_shot.yaml / re_reading.yaml / etc.
```

Other things to check if the task is still not found:
- `--include_path` must point to the **parent** `tasks` directory (`./src/tasks`), not `./src/tasks/vietnamese_ssa`.
- Task `.yaml` files must literally end in `.yaml` (not `.yml`).

## ⚠️ Reproducibility Notes: Reasoning/Thinking Mode

For reasoning-capable models (e.g. the Gemma-4 family), whether **"thinking" mode is
enabled** has a **major impact on SF1** — in our own experiments we observed swings of
roughly 2–3x on the same prompt purely from this setting. Always set it **explicitly**
and record the value used alongside your reported numbers:

| Backend | Flag |
|---|---|
| Native vLLM / vLLM OpenAI server | `--default-chat-template-kwargs '{"enable_thinking": true\|false}'` (+ `--reasoning-parser <name>` for the OpenAI server) |
| `llama.cpp` (`llama-server`) | `--reasoning on\|off` (optionally `--reasoning-budget N` to cap thinking length, `--reasoning-format deepseek` to separate `reasoning_content` from `content`) |

## Supported Datasets

This repository currently supports benchmark datasets from both public Structured Sentiment Analysis benchmarks and custom datasets. Most multilingual datasets are adapted from the official **SemEval-2022 Task 10** benchmark, while additional datasets can be integrated through the task definitions provided in this repository.

| Dataset | Language(s) | Source | Description | Status |
|----------|-------------|--------|-------------|:------:|
| ViToED | Vietnamese | Internal | Structured sentiment dataset for Vietnamese | ✅ |
| NoReC Fine | Norwegian | SemEval-2022 | Review domain | ✅ |
| OpeNER | English, Spanish | SemEval-2022 | Hotel reviews | ✅ |
| MultiBooked | Catalan, Basque | SemEval-2022 | Hotel reviews | ✅ |
| Darmstadt_unis | English | SemEval-2022 | University-related user reviews annotated for structured sentiment analysis | ✅ |
| MPQA | English | MPQA Corpus | News articles annotated with private states, opinions, sentiment expressions, and opinion holders/targets | ✅ |

> **Note**
>
> The multilingual benchmark datasets and their evaluation protocol are adapted from the official **SemEval-2022 Task 10: Structured Sentiment Analysis** repository. Dataset preprocessing scripts are available under `semeval22_structured_sentiment/`.

---

## Evaluation Metrics

Model predictions are evaluated following the official **SemEval-2022 Task 10** evaluation protocol.

Each predicted opinion is represented as a structured tuple consisting of:

- **Holder** (Source)
- **Target**
- **Sentiment Expression** (Polar expression)
- **Polarity**

Predicted and gold tuples are matched using weighted span overlap, following the official SemEval-2022 Task 10 protocol. The following metrics are computed and reported for every run:

| Metric | Description |
|---------|-------------|
| **SF1** | Sentiment Graph F1 — the official metric. Requires overlap on Holder, Target, and Expression *and* an exact Polarity match. |
| **NSF1** | Same as SF1 but polarity-agnostic (Holder/Target/Expression overlap only). |
| **Holder F1** | Span F1 on the opinion Holder only. |
| **Target F1** | Span F1 on the opinion Target only. |
| **Exp F1** | Span F1 on the Polar Expression only. |
| **Targeted F1** | Requires an *exact* (non-weighted) Target span match, plus Polarity match. |

Unlike traditional sentiment classification, Structured Sentiment Analysis requires correctly predicting both the sentiment polarity and the relationships among opinion components. Consequently, the official evaluation is performed on complete opinion graphs rather than isolated spans.

---

## Results

This repository is part of ongoing research. Benchmark results will be added to this
section as experiments are completed.

---

## License

TBD. A license has not yet been selected for this repository — until one is added,
all rights are reserved by the authors and the code should not be reused or
redistributed without permission. Note that the bundled datasets
(`semeval22_structured_sentiment/`, `data/`) are governed by their own original
licenses/terms, independent of whatever license is eventually chosen for the code
in this repository; please refer to the SemEval-2022 Task 10 organizers for dataset
terms.

---

## Citation

If you use this repository, please cite the following:

```bibtex
@inproceedings{barnes-etal-2022-semeval,
    title = "{S}em{E}val-2022 Task 10: Structured Sentiment Analysis",
    author = "Barnes, Jeremy and
              Oberl{\"a}nder, Laura Ana Maria and
              Troiano, Enrica and
              Kutuzov, Andrey and
              Buchmann, Jan and
              Agerri, Rodrigo and
              {\O}vrelid, Lilja  and
              Velldal, Erik",
    booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022)",
    month = july,
    year = "2022",
    address = "Seattle",
    publisher = "Association for Computational Linguistics"
}

@misc{barnes2021structuredsentimentanalysisdependency,
      title={Structured Sentiment Analysis as Dependency Graph Parsing}, 
      author={Jeremy Barnes and Robin Kurtz and Stephan Oepen and Lilja Øvrelid and Erik Velldal},
      year={2021},
      eprint={2105.14504},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2105.14504}, 
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}

@inproceedings{ovrelid-etal-2020-fine,
    title = "A Fine-grained Sentiment Dataset for {N}orwegian",
    author = "{\O}vrelid, Lilja  and
      M{\ae}hlum, Petter  and
      Barnes, Jeremy  and
      Velldal, Erik",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.618",
    pages = "5025--5033",
    abstract = "We here introduce NoReC{\_}fine, a dataset for fine-grained sentiment analysis in Norwegian, annotated with respect to polar expressions, targets and holders of opinion. The underlying texts are taken from a corpus of professionally authored reviews from multiple news-sources and across a wide variety of domains, including literature, games, music, products, movies and more. We here present a detailed description of this annotation effort. We provide an overview of the developed annotation guidelines, illustrated with examples and present an analysis of inter-annotator agreement. We also report the first experimental results on the dataset, intended as a preliminary benchmark for further experiments.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}

@inproceedings{barnes-etal-2018-multibooked,
    title = "{M}ulti{B}ooked: A Corpus of {B}asque and {C}atalan Hotel Reviews Annotated for Aspect-level Sentiment Classification",
    author = "Barnes, Jeremy  and
      Badia, Toni  and
      Lambert, Patrik",
    booktitle = "Proceedings of the Eleventh International Conference on Language Resources and Evaluation ({LREC} 2018)",
    month = may,
    year = "2018",
    address = "Miyazaki, Japan",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L18-1104",
}

@inproceedings{Agerri2013,
    author = {Agerri, Rodrigo and Cuadros, Montse and Gaines, Sean and Rigau, German},
    booktitle = {Sociedad Espa{\~{n}}ola para el Procesamiento del Lenguaje Natural},
    pages = {215--218},
    title = {{OpeNER: Open polarity enhanced named entity recognition.}},
    volume = {51},
    year = {2013}
}

@article{Wiebe2005b,
author = {Wiebe, Janyce
        and Wilson, Theresa
        and Cardie, Claire},
journal = {Language Resources and Evaluation},
number = {2-3},
pages = {165--210},
title = {{Annotating expressions of opinions and emotions in language}},
volume = {39},
year = {2005}
}

@inproceedings{toprak-etal-2010-sentence,
    title = "Sentence and Expression Level Annotation of Opinions in User-Generated Discourse",
    author = "Toprak, Cigdem  and
      Jakob, Niklas  and
      Gurevych, Iryna",
    booktitle = "Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2010",
    address = "Uppsala, Sweden",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P10-1059",
    pages = "575--584",
}
```