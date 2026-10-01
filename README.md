# Benchmarking Large Language Models for Structured Sentiment Analysis

---

A research repository for benchmarking Large Language Models (LLMs) on Structured Sentiment Analysis (SSA) using **LM Evaluation Harness**. The repository provides benchmark tasks, configurable prompting strategies, YAML-driven experiment and server configs, support for multiple inference backends, and reproducible experimental settings for evaluating LLMs across multilingual SSA datasets.

## Overview

Structured Sentiment Analysis (SSA) extends traditional sentiment analysis by extracting complete opinion structures, including opinion holders, opinion targets, sentiment expressions, and their corresponding polarities. Although recent Large Language Models (LLMs) have demonstrated impressive capabilities in structured information extraction, evaluating their performance on SSA remains challenging due to complex structured outputs, diverse annotation schemes across datasets, and the lack of standardized evaluation protocols.

This repository extends **LM Evaluation Harness** by providing benchmark tasks, prompt templates, output parsers, and evaluation configurations for Structured Sentiment Analysis. It is designed to facilitate reproducible benchmarking of LLMs across multiple datasets, languages, and inference backends while maintaining a consistent evaluation workflow.

Experiments are **config-driven**: evaluation settings live in YAML under `src/config/`, and Python entrypoints (`src/script/run.py`, `src/script/host.py`) only load a config and run. The same tasks remain usable from the `lm_eval` CLI and from the example notebook.

The repository is compatible with multiple inference backends—including **vLLM** (in-process or OpenAI server), **llama.cpp** (`llama-server`), and other OpenAI-compatible APIs—allowing experiments to be conducted consistently on both high-performance servers and local deployments.

### Key Features

- Structured Sentiment Analysis benchmark tasks built on **LM Evaluation Harness**.
- Four prompting techniques as separate tasks (grouped under tag `vietnamese_ssa`).
- YAML experiment configs: load the model **once**, then run many datasets × scenarios.
- Independent YAML configs to host **vLLM** or **llama-server**.
- Support for multilingual SSA datasets (language is inferred from the dataset name).
- Automatic parsing and validation of structured outputs.
- Reproducible experiment configurations with Docker support.

This repository is intended for researchers and practitioners who wish to evaluate, compare, and analyze the capabilities of Large Language Models on Structured Sentiment Analysis under consistent and reproducible experimental settings.

---



## Repository Structure

```text
.
├── data/
│   └── vitoed/                         # Vietnamese SSA dataset (train/dev/test.json)
├── semeval22_structured_sentiment/     # SemEval-2022 Task 10 data, preprocessing, official eval
│   ├── data/                           # opener_en, opener_es, mpqa, norec, multibooked_*, darmstadt_unis
│   └── evaluation/
├── notebook/
│   └── vastai_notebook_lm_eval.ipynb   # Setup + example lm_eval / server cells
├── src/
│   ├── config/                         # YAML configs for evaluation and model hosting
│   │   ├── local.yaml                  # vLLM in-process (mode: local_load)
│   │   ├── api_server.yaml             # HTTP client to an OpenAI-compatible server
│   │   ├── host_vllm.yaml              # Start vLLM OpenAI server
│   │   └── host_llama.yaml             # Start llama-server (GGUF)
│   ├── script/
│   │   ├── run.py                      # CLI: load experiment YAML and evaluate
│   │   ├── lm_eval_runner.py           # Library: Model/Run/Scenario configs + LMEvalRunner
│   │   ├── host.py                     # CLI: load server YAML and keep the process alive
│   │   └── server_host.py              # Library: vLLM / llama.cpp process lifecycle
│   ├── tasks/
│   │   └── vietnamese_ssa/             # LM Evaluation Harness task YAMLs + utils.py
│   │       ├── _common_yaml            # Shared metrics, filters, generation_kwargs (no .yaml suffix)
│   │       ├── few_shot.yaml
│   │       ├── re_reading.yaml
│   │       ├── few_shot_cot.yaml
│   │       ├── plan_and_solve.yaml
│   │       └── utils.py
│   ├── prompt_templates/               # System prompts, CoT example pools, prompt blocks
│   └── utils/                          # Postprocessing / JSON extraction
├── Dockerfile
├── requirements.txt
└── README.md
```



### Directory Overview


| Path                              | Description                                                                                                |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `data/vitoed/`                    | Vietnamese SSA splits (`train.json`, `dev.json`, `test.json`). Registered dataset name: `vitoed`.          |
| `semeval22_structured_sentiment/` | Multilingual SemEval-2022 SSA datasets and official evaluation code.                                       |
| `src/config/`                     | Experiment YAMLs (`local.yaml`, `api_server.yaml`) and server YAMLs (`host_vllm.yaml`, `host_llama.yaml`). |
| `src/script/`                     | Config loaders and runners. Evaluation (`run.py`) is independent of hosting (`host.py`).                   |
| `src/tasks/vietnamese_ssa/`       | One YAML per prompting technique; few-shot sampling is native harness `fewshot_config`.                    |
| `src/prompt_templates/`           | System prompts per language and static CoT example pools.                                                  |
| `src/utils/`                      | Postprocessing for parsing structured model outputs.                                                       |
| `notebook/`                       | Example notebook for Vast.ai / Docker workflows.                                                           |


---



## Installation

This repository supports two installation methods:

- **Docker (Recommended):** CUDA, PyTorch, vLLM, and a pinned `llama-server` binary. **LM Evaluation Harness is installed at container runtime** (not baked into the image) so its version can be pinned per experiment.
- **Local Environment:** Install the required dependencies manually on your system.

---



### Option 1. Docker (Recommended)

Build from this repository (Dockerfile tracks CUDA 13.0, PyTorch 2.11, vLLM 0.23.0, and llama.cpp `v0.4.0` for `llama-server`):

```bash
docker build -t kgb0630/lm-eval-vastai:v12 .
```

Or pull a published tag if you already pushed one to Docker Hub:

```bash
docker pull kgb0630/lm-eval-vastai:<tag>
```

Launch the container with GPU support (mount the repo into `/workspace`):

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    kgb0630/lm-eval-vastai:v12
```

Inside the container, install LM Evaluation Harness (see below), then run evaluation from the **repository root**.

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

This project is built on top of **LM Evaluation Harness**. Follow the official installation instructions:

- **Repository:** [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

Recommended extras for this repo:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[api,vllm,hf]"
```

> Extra `[api]` is required for `model.mode: api_server` (`local-chat-completions`). Extra `[vllm]` is required for in-process vLLM (`model.mode: local_load`).

---



### Optional: FlashAttention

FlashAttention is optional but recommended for supported NVIDIA GPUs to improve inference throughput.

```bash
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install <flash-attention-wheel>
```

Please refer to the official projects for installation instructions and compatible pre-built wheels:

- **FlashAttention:** [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **Pre-built Wheels:** [https://github.com/mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)

Since FlashAttention depends on your **CUDA**, **PyTorch**, **Python**, and **GPU architecture**, it is recommended to follow the compatibility matrix provided by the official repositories rather than using fixed installation commands.

**FlashInfer JIT (NVFP4 / vLLM):** if host RAM is exhausted during kernel compile (`nvcc` killed, exit 137), set `MAX_JOBS=1` (see `src/config/host_vllm.yaml` and [FlashInfer #3634](https://github.com/flashinfer-ai/flashinfer/issues/3634)). Increase to `2` or `4` only if RAM allows.

---



### Verify the Installation

```bash
lm_eval --help
```

If the installation is successful, the command above will display the available command-line options.

### Requirements


| Component     | Version                                       |
| ------------- | --------------------------------------------- |
| Python        | >= 3.12                                       |
| CUDA          | >= 13.0 (recommended; image uses CUDA 13.0.3) |
| Docker        | Optional                                      |
| NVIDIA Driver | Compatible with installed CUDA                |


Pinned stack in `requirements.txt` / `Dockerfile`: PyTorch **2.11.0** (cu130), vLLM **0.23.0**, Transformers `>=5.10,<5.15`.

---



## Running Evaluation

Run **all evaluation commands from the repository root**. Scripts under `src/script/` import sibling modules, so put that directory on `PYTHONPATH`:

```bash
# Linux / macOS / Docker
export PYTHONPATH=src/script

# Windows PowerShell
$env:PYTHONPATH = "src/script"
```

There are three equivalent workflows:

1. **YAML +** `run.py` **(recommended)** — one config, model loaded once, many scenarios/datasets.
2. **Notebook** — `notebook/vastai_notebook_lm_eval.ipynb`.
3. `lm_eval` **CLI** — one task per invocation (reloads the model each time unless you use an API server).

Dataset selection is **not** `--metadata '{"language":"vi"}'`. Pass a registered **dataset name**; language is looked up in `DATASET_LANGUAGES` (`src/prompt_templates/shared.py`).

---



### Config-driven evaluation (`run.py`)

`src/script/lm_eval_runner.py` builds the LM **once**, then loops over `run.datasets` × `scenarios`. Output layout:

```text
<output_root>/<model_tag>/<dataset>/<scenario_label>/
```

Example: `results/lm_eval/google__gemma-4-E2B-it/vitoed/3_shot/`

#### In-process vLLM (`src/config/local.yaml`)

Edit `model.args.pretrained` and `run.datasets`, then:

```bash
python src/script/run.py --config src/config/local.yaml
python src/script/run.py --config src/config/local.yaml --dry-run
python src/script/run.py --config src/config/local.yaml --only re2,3_shot
python src/script/run.py --config src/config/local.yaml --dataset mpqa,opener_en
```

`--dataset` and `--only` override YAML. `--only` filters by scenario **label** (`re2`, `pas`, `3_shot`, `5_shot_cot`, …).

#### HTTP client to a hosted model (`src/config/api_server.yaml`)

Start a server first (`host.py` below, or any OpenAI-compatible endpoint). Point `model.args.base_url` at `http://127.0.0.1:8000/v1/chat/completions`, then:

```bash
python src/script/run.py --config src/config/api_server.yaml
```

`batch_size` should stay `1` for chat completions; concurrency is `args.num_concurrent`.

#### Scenario preset `default_ssa`

If the YAML has `scenarios: default_ssa`, the runner uses 11 standard scenarios:


| Label                       | Task                            | `--num_fewshot`   |
| --------------------------- | ------------------------------- | ----------------- |
| `re2`                       | `vietnamese_ssa_re_reading`     | 0                 |
| `pas`                       | `vietnamese_ssa_plan_and_solve` | 0                 |
| `0_shot` … `10_shot`        | `vietnamese_ssa_few_shot`       | 0, 1, 3, 5, 7, 10 |
| `1_shot_cot` … `5_shot_cot` | `vietnamese_ssa_few_shot_cot`   | 1, 3, 5           |


You can instead list scenarios explicitly in YAML (`label`, `task`, `num_fewshot`, optional `gen_kwargs`). Per-scenario `gen_kwargs` override `run.gen_kwargs`, which override the task YAML.

Optional `run.extra_metadata` is merged into the TaskManager metadata (for example `plus_mode: true` for Plan-and-Solve PS+, or `clean_data: false` to skip gold-span cleaning).

---



### Hosting an inference server (`host.py`)

Use a **separate** terminal (or notebook cell). Evaluation (`run.py`) does not start the server.

```bash
python src/script/host.py --config src/config/host_vllm.yaml
python src/script/host.py --config src/config/host_llama.yaml
```

The process waits until the health endpoint is ready, then stays up until Ctrl+C. Logs go under `log_dir` (`vllm_server.log` or `llama_server.log`).

**Gated Hugging Face models:** set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / `HUGGINGFACE_API_KEY` before starting the host.

You can still launch servers by hand (`python -m vllm.entrypoints.openai.api_server` or `llama-server`); `host.py` is the same flags, parameterized.

---



### Notebook workflow

```text
notebook/vastai_notebook_lm_eval.ipynb
```

The notebook covers environment setup, a sample `lm_eval` command, and cells to host vLLM / llama-server. Prefer aligning new runs with `src/config/*.yaml` so CLI and notebook share the same knobs.

---



### Command-line `lm_eval` (single task)



#### Native vLLM backend

```bash
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
lm_eval \
  --model vllm \
  --model_args pretrained=<MODEL_NAME>,max_model_len=16384,gpu_memory_utilization=0.95 \
  --tasks <TASK_NAME> \
  --include_path ./src/tasks \
  --num_fewshot 3 \
  --seed 0,1234,1234,42 \
  --apply_chat_template \
  --log_samples \
  --batch_size 32 \
  --output_path results/lm_eval/<experiment_name> \
  --metadata '{"dataset":"vitoed"}' \
  --confirm_run_unsafe_code
```



#### OpenAI-compatible API

After a server is listening:

```bash
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
lm_eval \
  --model local-chat-completions \
  --model_args model=<MODEL_NAME>,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_length=16384 \
  --tasks <TASK_NAME> \
  --include_path ./src/tasks \
  --num_fewshot 3 \
  --seed 0,1234,1234,42 \
  --apply_chat_template \
  --log_samples \
  --output_path results/lm_eval/<experiment_name> \
  --metadata '{"dataset":"vitoed"}' \
  --confirm_run_unsafe_code
```

`--include_path` must be the parent `./src/tasks`, not `./src/tasks/vietnamese_ssa`.

Use `--tasks vietnamese_ssa` to run all four technique tasks in one CLI call, or a single task:


| `<TASK_NAME>`                   | Technique                 | Notes                                                                                                                                     |
| ------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `vietnamese_ssa_few_shot`       | Plain few-shot            | `--num_fewshot` sampled from the `train` split (seeded)                                                                                   |
| `vietnamese_ssa_re_reading`     | Re-reading                | Question repeated twice; same live sampling as few-shot                                                                                   |
| `vietnamese_ssa_few_shot_cot`   | Few-shot Chain-of-Thought | Examples come from a static pool in `src/prompt_templates/shared.py`, not `train`. Max `--num_fewshot` is the pool size for that language |
| `vietnamese_ssa_plan_and_solve` | Plan-and-Solve            | Instruction-only; YAML locks `num_fewshot: 0`. PS+: `--metadata '{"dataset":"vitoed","plus_mode":true}'`                                  |


---



## Troubleshooting: `ValueError: Tasks not found: <task_name>`

If `lm_eval` reports a task as "not found" even though its `.yaml` exists under `--include_path`, the cause is almost always a **silently skipped file**, not a missing task. When lm-eval-harness scans `--include_path` to build its task index, any YAML file that fails to parse (missing `include:` target, malformed key, etc.) is dropped **without a visible error** — it only logs at `DEBUG` level. To reveal the real reason, run:

```bash
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from lm_eval.tasks import TaskManager
tm = TaskManager(include_path='./src/tasks')
print('found:', 'vietnamese_ssa_few_shot' in tm.all_tasks)
" 2>&1 | grep -i "skip.*vietnamese_ssa"
```

The most common root cause in this repo: `_common_yaml` (included by every technique YAML) is named **without** a `.yaml` extension so the scanner does not treat it as a standalone task. Some tools silently append an extension when saving a dot-less filename — if that happens, every technique task under `src/tasks/vietnamese_ssa/` will fail to index. Verify with:

```bash
ls -la src/tasks/vietnamese_ssa/
# must show a file literally named `_common_yaml` (no extension)
```

Other checks:

- `--include_path` must point to `./src/tasks`, not `./src/tasks/vietnamese_ssa`.
- Task files must end in `.yaml` (not `.yml`).
- `--metadata` must include `"dataset":"<registered_name>"` (see table below). Technique is already set in each task YAML.

---



## Reproducibility Notes: Reasoning/Thinking Mode

For reasoning-capable models (e.g. the Gemma-4 family), whether **"thinking" mode is enabled** has a **major impact on SF1**. Always set it **explicitly** and record the value used alongside reported numbers.


| Backend                          | Flag                                                                                                                                                                                       |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Native vLLM / vLLM OpenAI server | `--default-chat-template-kwargs '{"enable_thinking": true|false}'` (plus `--reasoning-parser <name>` on the OpenAI server). In `host_vllm.yaml`: `enable_thinking` and `reasoning_parser`. |
| `llama.cpp` (`llama-server`)     | `--reasoning on|off` (optional `--reasoning-budget N`, `--reasoning-format`). In `host_llama.yaml`: `reasoning`.                                                                           |


Seeds used by `run.py` / the documented CLI: `random=0`, `numpy=1234`, `torch=1234`, `fewshot=42`, plus `PYTHONHASHSEED=42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

---



## Supported Datasets

Registered names must match `DATASET_LANGUAGES` in `src/prompt_templates/shared.py` (and `_DATASET_ROOTS` in `src/tasks/vietnamese_ssa/utils.py`). Use these strings in `run.datasets`, `--dataset`, and `--metadata '{"dataset":"..."}'`.


| Dataset key      | Language   | On disk                                      | Source       |
| ---------------- | ---------- | -------------------------------------------- | ------------ |
| `vitoed`         | Vietnamese | `data/vitoed/`                               | Internal     |
| `norec`          | Norwegian  | `semeval22_structured_sentiment/data/norec/` | SemEval-2022 |
| `opener_en`      | English    | `.../data/opener_en/`                        | SemEval-2022 |
| `opener_es`      | Spanish    | `.../data/opener_es/`                        | SemEval-2022 |
| `multibooked_ca` | Catalan    | `.../data/multibooked_ca/`                   | SemEval-2022 |
| `multibooked_eu` | Basque     | `.../data/multibooked_eu/`                   | SemEval-2022 |
| `darmstadt_unis` | English    | `.../data/darmstadt_unis/`                   | SemEval-2022 |
| `mpqa`           | English    | `.../data/mpqa/`                             | MPQA Corpus  |


Override the directory with metadata `dataset_dir` if needed. Gold opinions are cleaned by default (`clean_data: true`); set `clean_data: false` in `run.extra_metadata` or CLI `--metadata` to disable.

> The multilingual datasets and evaluation protocol are adapted from **SemEval-2022 Task 10: Structured Sentiment Analysis**. Preprocessing scripts live under `semeval22_structured_sentiment/`.

---



## Evaluation Metrics

Model predictions are evaluated following the official **SemEval-2022 Task 10** evaluation protocol (wired in `_common_yaml` via `utils.process_results` and metric aggregators).

Each predicted opinion is a structured tuple:

- **Holder** (Source)
- **Target**
- **Sentiment Expression** (Polar expression)
- **Polarity**

Predicted and gold tuples are matched using weighted span overlap.


| Metric          | Description                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| **SF1**         | Sentiment Graph F1 — official metric. Overlap on Holder, Target, and Expression *and* exact Polarity match. |
| **NSF1**        | Same as SF1 but polarity-agnostic.                                                                          |
| **Holder F1**   | Span F1 on the opinion Holder only.                                                                         |
| **Target F1**   | Span F1 on the opinion Target only.                                                                         |
| **Exp F1**      | Span F1 on the Polar Expression only.                                                                       |
| **Targeted F1** | Exact (non-weighted) Target span match, plus Polarity match.                                                |


Unlike traditional sentiment classification, SSA requires correctly predicting both polarity and the relationships among opinion components. Official evaluation is performed on complete opinion graphs rather than isolated spans.

---



## Results

This repository is part of ongoing research. Benchmark results will be added to this section as experiments are completed.

---



## License

TBD. A license has not yet been selected for this repository — until one is added, all rights are reserved by the authors and the code should not be reused or redistributed without permission. Note that the bundled datasets (`semeval22_structured_sentiment/`, `data/`) are governed by their own original licenses/terms, independent of whatever license is eventually chosen for the code in this repository; please refer to the SemEval-2022 Task 10 organizers for dataset terms.

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

@INPROCEEDINGS{11685817,
  author={Vo, Chanh and Luu, Son T. and Luu-Thuy Nguyen, Ngan},
  booktitle={2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={ViTOED: A Dataset for Target-Oriented Emotion Detection on Vietnamese Social Media Texts}, 
  year={2026},
  volume={},
  number={},
  pages={1-6},
  keywords={Modeling;Head;Labeling;Social networking (online);Printing;Training;Sentiment analysis;Conferences;Multilingual;Signal detection;targeted sentiment analysis;dataset;social media texts},
  doi={10.1109/MAPR72750.2026.11685817}}

```

