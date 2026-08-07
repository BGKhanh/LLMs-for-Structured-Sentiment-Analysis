"""
Bridge module between lm-evaluation-harness and the project's existing code.

Reorg note (v2): this file backs ALL FOUR per-technique yaml configs
(`few_shot.yaml`, `re_reading.yaml`, `few_shot_cot.yaml`,
`plan_and_solve.yaml`), each sharing this same `utils.py` + `_common_yaml`.
All four are on lm-eval's *native* few-shot mechanism now (or, for
`plan_and_solve`, deliberately have none). `src/prompt_templates/contents.py`,
`creator.py`, and `factory.py` (the old generic `PromptCreator` /
`ContentProvider` pipeline) have been REMOVED — nothing in this pipeline
imports them anymore, and their responsibilities are now split between
per-doc callables here and the trimmed-down `blocks.py`. `blocks.py` and
`shared.py` are still used directly (see imports below) — pure, reusable
string-building helpers with no lm-eval-specific coupling.

Key change vs the legacy single-file version: `load_dataset()` no longer
pre-computes the full `user_prompt` string. It now returns RAW fields only
(`text`, `opinions_json`, `sent_id`, `language`, `system_prompt`,
`plus_mode`). Prompt rendering is done lazily, per-doc, by the
`*_doc_to_text()` / `*_fewshot_doc_to_text()` / `*_fewshot_doc_to_target()`
functions below — each a plain function of a single `doc` dict, no
reliance on a global "current technique" variable. That's what lets all 4
technique yamls (grouped under one `tag`) run safely in the same process
without leaking state into each other. The ONE exception is
`get_cot_pool()` (see its docstring), which needs a tiny bit of
module-level state due to an lm-eval API constraint, not a design choice.

Few-shot sampling itself (which examples to pick, with what seed, whether
to exclude the eval doc) is handled natively by lm-eval-harness via
`fewshot_config` in each yaml — never by hand-rolled `random.sample()`.

Per-technique function map (see each yaml for the exact wiring):
    few_shot        -> doc_to_text                / fewshot_doc_to_text            / fewshot_doc_to_target
    re_reading      -> re_reading_doc_to_text      / re_reading_fewshot_doc_to_text / re_reading_fewshot_doc_to_target
    few_shot_cot    -> cot_doc_to_text             / cot_fewshot_doc_to_text        / cot_fewshot_doc_to_target (+ get_cot_pool)
    plan_and_solve  -> plan_and_solve_doc_to_text  (no fewshot_config)

Shared across all 4: load_dataset, extract_and_postprocess, process_results,
<metric>_agg.

YAML references (see few_shot.yaml for the canonical example):
    custom_dataset: !function utils.load_dataset
    description: "{{system_prompt}}"                    (Jinja, per-doc field)
    doc_to_text: !function utils.doc_to_text             (eval question)
    doc_to_target: "{{opinions_json}}"                   (Jinja, per-doc field)
    fewshot_config:
        split: train
        doc_to_text: !function utils.fewshot_doc_to_text
        doc_to_target: !function utils.fewshot_doc_to_target
    process_results: !function utils.process_results
    filter_fn: !function utils.extract_and_postprocess
    aggregation: !function utils.<metric>_agg
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import datasets

# ---------------------------------------------------------------------------
# Resolve project root so we can import from src/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prompt_templates.shared import get_system_prompt, load_examples_pool, get_language, DATASET_LANGUAGES
from src.prompt_templates.blocks import (
    base_question_block,
    simplify_opinions,
    rereading_block,
    pas_instruction_block,
)
from src.utils.postprocessing import (
    extract_json_from_response,
    postprocess_response,
    clean_gold_data,
)
from semeval22_structured_sentiment.evaluation.evaluate import (
    convert_opinion_to_tuple,
    sent_tuples_in_list,
    weighted_score,
    set_tokenizer,
)

_EPSILON = 1e-16

_DATASET_ROOTS = {
    "vitoed": "data",  # special-cased top-level folder, not under semeval22_structured_sentiment/
    "opener_en": "semeval22_structured_sentiment/data",
    "mpqa": "semeval22_structured_sentiment/data",
    "darmstadt_unis": "semeval22_structured_sentiment/data",
    "opener_es": "semeval22_structured_sentiment/data",
    "norec": "semeval22_structured_sentiment/data",
    "multibooked_eu": "semeval22_structured_sentiment/data",
    "multibooked_ca": "semeval22_structured_sentiment/data",
}
assert set(_DATASET_ROOTS) == set(DATASET_LANGUAGES), (
    "utils._DATASET_ROOTS and shared.DATASET_LANGUAGES have drifted apart — "
    "every dataset must be registered in BOTH (dataset->root here, "
    "dataset->language there)."
)
 
# Cache system prompts per language (cheap, static strings; safe to share
# across techniques/tasks running in the same process).
_SYS_PROMPT_CACHE: Dict[str, str] = {}
 
 
def _get_system_prompt(language: str) -> str:
    if language not in _SYS_PROMPT_CACHE:
        _SYS_PROMPT_CACHE[language] = get_system_prompt(language)
    return _SYS_PROMPT_CACHE[language]
 
 
_CURRENT_DATASET = {"value": "vitoed"}


def _resolve_dataset_paths(dataset: str, dataset_dir: Optional[str]) -> Dict[str, Path]:
    if dataset_dir:
        base = Path(dataset_dir)
        if not base.is_absolute():
            base = _PROJECT_ROOT / base
    elif dataset not in _DATASET_ROOTS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported: {sorted(_DATASET_ROOTS.keys())}"
        )
    else:
        base = _PROJECT_ROOT / _DATASET_ROOTS[dataset] / dataset
    return {
        "train": base / "train.json",
        "dev": base / "dev.json",
        "test": base / "test.json",
    }


# =========================================================================
# Dataset loading (receives --metadata as **kwargs)
# =========================================================================
def load_dataset(**kwargs) -> datasets.DatasetDict:
    """Load local JSON and return RAW fields for every document.

    Called by lm-eval via ``custom_dataset: !function utils.load_dataset``.
    Receives ``--metadata`` CLI values (merged with the yaml's own
    ``metadata:`` block, e.g. ``technique: few_shot`` set in few_shot.yaml)
    as keyword arguments.

    Required metadata keys:
        technique (str): informational only for now (which yaml is
            calling this); not used to branch behavior here since prompt
            rendering has moved to doc_to_text / fewshot_config callables.

    Optional metadata keys:
        language (str): 'vi' or 'en'/'es'/'nor'/'eu'/'ca' (default: 'vi').
        dataset_dir (str): Override dataset directory.
        clean_data (bool): Run clean_gold_data() before returning (default: true).

    NOTE: `n_shot` / `examples_pool_path` are intentionally gone here —
    for `few_shot`, sampling is now handled natively by lm-eval via
    `fewshot_config.split: train` in the yaml.
    """
    technique = kwargs.get("technique")
    if not technique:
        raise ValueError(
            "Missing 'technique' in --metadata (or in the yaml's metadata: block). "
            "Example: --metadata '{\"technique\":\"few_shot\",\"dataset\":\"vitoed_new\"}'"
        )
 
    dataset = kwargs.get("dataset")
    if not dataset:
        raise ValueError(
            "Missing 'dataset' in --metadata. Example: "
            "--metadata '{\"technique\":\"few_shot\",\"dataset\":\"mpqa\"}'. "
            f"Available datasets: {sorted(DATASET_LANGUAGES.keys())}"
        )
    language = get_language(dataset)  # raises ValueError if `dataset` is unregistered
 
    dataset_dir = kwargs.get("dataset_dir", None)
    clean_data = bool(kwargs.get("clean_data", True))
    plus_mode = bool(kwargs.get("plus_mode", False))  # only meaningful for plan_and_solve
 
    _CURRENT_DATASET["value"] = dataset
    set_tokenizer(language)
 
    dataset_paths = _resolve_dataset_paths(dataset, dataset_dir)
    system_prompt = _get_system_prompt(language)
 
    splits = {}
    for split_name, path in dataset_paths.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
 
        if clean_data:
            raw, removed_ids, _ = clean_gold_data(raw, language=language, verbose=True)
 
        records = []
        for sample in raw:
            records.append({
                "sent_id": sample["sent_id"],
                "text": sample["text"],
                "opinions_json": json.dumps(sample.get("opinions", []), ensure_ascii=False),
                "dataset": dataset,
                "language": language,
                "system_prompt": system_prompt,
                "plus_mode": plus_mode,
            })
        splits[split_name] = datasets.Dataset.from_list(records)
 
    return datasets.DatasetDict(splits)


# =========================================================================
# doc_to_text — eval question. Pure function of `doc`, no external state.
# =========================================================================
def doc_to_text(doc: Dict[str, Any]) -> str:
    return base_question_block(doc["text"], str(doc["sent_id"]), doc.get("language", "vi"))


# =========================================================================
# Native few-shot rendering (fewshot_config.doc_to_text / doc_to_target).
# Called ONCE PER SELECTED EXAMPLE by lm-eval's ContextSampler — sampling,
# seeding (--fewshot_random_seed) and excluding the eval doc from the pool
# are all handled by the harness, not by this module.
# =========================================================================
def fewshot_doc_to_text(doc: Dict[str, Any]) -> str:
    """Render the 'Input:' half of one few-shot example."""
    text = doc["text"]
    return f'Input: "{text}"\nOutput:'


def fewshot_doc_to_target(doc: Dict[str, Any]) -> str:
    """Render the 'Output:' half of one few-shot example (simplified JSON,
    same format as legacy `few_shot_block()`: text arrays only, no spans)."""
    opinions = json.loads(doc["opinions_json"])
    payload = {"opinions": simplify_opinions(opinions)}
    return json.dumps(payload, ensure_ascii=False, indent=2)


# =========================================================================
# re_reading technique — question asked twice (base_q + "read again" + base_q).
# Pure re-reading only (legacy add_method="none"): no CoT/PaS trigger combined
# in, per current scope. Fewshot demos are natively sampled from `train`,
# each one also rendered with the doubled-question format.
# =========================================================================
def re_reading_doc_to_text(doc: Dict[str, Any]) -> str:
    return rereading_block(doc["text"], str(doc["sent_id"]), doc.get("language", "vi"))


def re_reading_fewshot_doc_to_text(doc: Dict[str, Any]) -> str:
    core = rereading_block(doc["text"], str(doc["sent_id"]), doc.get("language", "vi"))
    return f"{core}\nOutput:"


def re_reading_fewshot_doc_to_target(doc: Dict[str, Any]) -> str:
    # Same simplified-JSON target format as plain few_shot.
    return fewshot_doc_to_target(doc)


# =========================================================================
# few_shot_cot technique — hand-written reasoning demos (from
# `shared._HARDCODED_POOL`, NOT from train.json — real data has no
# `reasoning` field). Uses `fewshot_config.samples` (static pool) instead of
# `fewshot_config.split`, since the pool isn't a dataset split.
# =========================================================================
def get_cot_pool() -> List[Dict[str, Any]]:
    """Zero-arg callable required by `fewshot_config.samples` — see the
    `_CURRENT_DATASET` note above for why this needs module-level state.
    Looked up by DATASET, not language, so e.g. `mpqa` and `opener_en`
    (both "en") can have independent pools (`load_examples_pool` falls
    back to a same-language sibling dataset if the requested one is
    empty — see shared.py)."""
    return load_examples_pool(None, _CURRENT_DATASET["value"])


def cot_doc_to_text(doc: Dict[str, Any]) -> str:
    text = doc["text"]
    if doc.get("language", "vi") == "en":
        return f'Input: "{text}"\n\nThink step by step, then return the final JSON.'
    return f'Input: "{text}"'


def cot_fewshot_doc_to_text(doc: Dict[str, Any]) -> str:
    # `doc` here is one entry from `_HARDCODED_POOL`: {"text", "reasoning", "output"}
    return f'Input: "{doc["text"]}"'


def cot_fewshot_doc_to_target(doc: Dict[str, Any]) -> str:
    reasoning = doc.get("reasoning", "").strip()
    output = doc.get("output", "").strip()
    return f"Reasoning:\n{reasoning}\n\nOutput:\n{output}"


# =========================================================================
# plan_and_solve technique — instruction-only, NO examples (matches legacy
# PlanAndSolveContent.setup(), which always discards the pool). `plus_mode`
# (PS+ variant) is read from --metadata, threaded through as a doc field.
# =========================================================================
def plan_and_solve_doc_to_text(doc: Dict[str, Any]) -> str:
    language = doc.get("language", "vi")
    instruction = pas_instruction_block(plus=bool(doc.get("plus_mode", False)), language=language)
    if language == "en":
        question = f'Analyze the sentiment for the following text:\n"{doc["text"]}"'
    else:
        question = f'Phân tích cảm xúc cho văn bản sau:\n"{doc["text"]}"'
    return f"{instruction}\n{question}"


# =========================================================================
# Filter: extract JSON from model response  (unchanged from legacy)
# =========================================================================
def extract_and_postprocess(resps: List[List[str]], docs: List[Dict]) -> List[List[str]]:
    """Custom filter: extract JSON then normalize to SemEval format.

    Input:  resps = [[raw_response], ...] (one list per doc)
    Output: [[processed_json], ...]       (keep list-of-list for pipeline)
    """
    filtered = []
    for resp_list, doc in zip(resps, docs):
        raw = resp_list[0] if resp_list else "{}"
        extracted = extract_json_from_response(raw)
        processed = postprocess_response(extracted, doc["text"], doc["sent_id"])
        filtered.append([processed])
    return filtered


# =========================================================================
# process_results: per-doc metric computation  (unchanged from legacy)
# =========================================================================
def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Compute per-document raw scores for corpus-level aggregation.

    Returns dict mapping metric_name ->
        (weighted_tp_prec, num_pred, weighted_tp_rec, num_gold).

    Precision and recall require separate weighted-TP accumulators because
    ``weighted_score(tuple, list)`` is asymmetric: the denominator is always
    the span length of the *first* argument.
    """
    response = results[0] if results else "{}"

    # Gold
    gold_sample = {
        "sent_id": str(doc["sent_id"]),
        "text": doc["text"],
        "opinions": json.loads(doc["opinions_json"]),
    }
    gold_tuples = convert_opinion_to_tuple(gold_sample)

    # Pred
    try:
        pred_data = json.loads(response)
        pred_sample = {
            "sent_id": str(doc["sent_id"]),
            "text": doc["text"],
            "opinions": pred_data.get("opinions", []),
        }
        pred_tuples = convert_opinion_to_tuple(pred_sample)
    except (json.JSONDecodeError, KeyError, TypeError):
        pred_tuples = []

    modes = {
        "SF1": ("all", True, True),
        "NSF1": ("all", False, True),
        "Holder_F1": ("holder", False, True),
        "Target_F1": ("target", False, True),
        "Exp_F1": ("expression", False, True),
        "Targeted_F1": ("targeted_strict", True, False),
    }
    scores = {}
    for name, (mode, keep_polarity, use_weighted) in modes.items():
        # Precision numerator: iterate predictions (tp) only.
        w_tp_prec = 0.0
        for pt in pred_tuples:
            if sent_tuples_in_list(pt, gold_tuples, keep_polarity=keep_polarity, mode=mode):
                w_tp_prec += weighted_score(pt, gold_tuples, mode=mode) if use_weighted else 1.0

        # Recall numerator: iterate gold (tp) only.
        w_tp_rec = 0.0
        for gt in gold_tuples:
            if sent_tuples_in_list(gt, pred_tuples, keep_polarity=keep_polarity, mode=mode):
                w_tp_rec += weighted_score(gt, pred_tuples, mode=mode) if use_weighted else 1.0

        scores[name] = (w_tp_prec, len(pred_tuples), w_tp_rec, len(gold_tuples))

    return scores


# =========================================================================
# Corpus-level aggregation functions  (unchanged from legacy)
# =========================================================================
def _corpus_f1(items: List[Tuple]) -> float:
    """Corpus-level F1 from per-doc tuples.

    Supports both:
    - (weighted_tp, num_pred, num_gold)  [legacy; precision/recall share numerator]
    - (weighted_tp_prec, num_pred, weighted_tp_rec, num_gold) [SemEval-2022-correct]
    """
    if not items:
        return 0.0

    if len(items[0]) == 3:
        total_wtp = sum(x[0] for x in items)
        total_pred = sum(x[1] for x in items)
        total_gold = sum(x[2] for x in items)
        precision = total_wtp / (total_pred + _EPSILON)
        recall = total_wtp / (total_gold + _EPSILON)
        return 2 * precision * recall / (precision + recall + _EPSILON)

    if len(items[0]) != 4:
        raise ValueError(f"Unexpected per-doc score tuple length: {len(items[0])}")

    total_wtp_prec = sum(x[0] for x in items)
    total_pred = sum(x[1] for x in items)
    total_wtp_rec = sum(x[2] for x in items)
    total_gold = sum(x[3] for x in items)

    precision = total_wtp_prec / (total_pred + _EPSILON)
    recall = total_wtp_rec / (total_gold + _EPSILON)
    return 2 * precision * recall / (precision + recall + _EPSILON)


def sf1_agg(items):
    return _corpus_f1(items)

def nsf1_agg(items):
    return _corpus_f1(items)

def holder_f1_agg(items):
    return _corpus_f1(items)

def target_f1_agg(items):
    return _corpus_f1(items)

def exp_f1_agg(items):
    return _corpus_f1(items)

def targeted_f1_agg(items):
    return _corpus_f1(items)