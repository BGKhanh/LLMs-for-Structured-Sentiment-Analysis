"""
Bridge module between lm-evaluation-harness and the project's existing code.

All configuration is driven by ``--metadata`` passed via CLI:
    lm-eval run --tasks vietnamese_ssa \
        --metadata '{"technique":"few_shot","language":"vi","n_shot":3}'

The load_dataset() function receives metadata as **kwargs, builds prompt creator
to setup prompt templates, and pre-computes system_prompt + user_prompt for
every document. This eliminates module-level state issues entirely.

YAML references:
- custom_dataset: !function utils.load_dataset
- description: "{{system_prompt}}"       (Jinja2, renders from doc field)
- doc_to_text: "{{user_prompt}}"         (Jinja2, renders from doc field)
- doc_to_target: "{{opinions_json}}"     (Jinja2, renders from doc field)
- process_results: !function utils.process_results
- filter_fn: !function utils.extract_and_postprocess
- aggregation: !function utils.<metric>_agg
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import datasets

# ---------------------------------------------------------------------------
# Resolve project root so we can import from src/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prompt_templates import build_prompt_creator
from src.utils.postprocessing import extract_json_from_response, postprocess_response
from semeval22_structured_sentiment.evaluation.evaluate import (
    convert_opinion_to_tuple,
    sent_tuples_in_list,
    weighted_score,
)

_EPSILON = 1e-16


# =========================================================================
# Dataset loading (receives --metadata as **kwargs)
# =========================================================================
def load_dataset(**kwargs) -> datasets.DatasetDict:
    """Load local JSON and pre-compute prompts for every document.

    Called by lm-eval via ``custom_dataset: !function utils.load_dataset``.
    Receives ``--metadata`` CLI values as keyword arguments.

    Required metadata keys:
        technique (str): Prompt technique name.

    Optional metadata keys:
        language (str): 'vi' or 'en' (default: 'vi').
        n_shot (int): Number of few-shot examples (default: 0).
        plus_mode (bool): PS+ mode for plan_and_solve (default: false).
        add_method (str): Enhancement for rereading (default: 'none').
        examples_pool_path (str): Path to examples pool JSON.
        dataset_dir (str): Override dataset directory.
    """
    technique = kwargs.get("technique")
    if not technique:
        raise ValueError(
            "Missing 'technique' in --metadata. Example: "
            "--metadata '{\"technique\":\"few_shot\",\"language\":\"vi\",\"n_shot\":3}'"
        )

    language = kwargs.get("language", "vi")
    n_shot = int(kwargs.get("n_shot", 0))
    plus_mode = bool(kwargs.get("plus_mode", False))
    add_method = str(kwargs.get("add_method", "none"))
    dataset_dir = kwargs.get("dataset_dir", None)

    # Resolve dataset paths
    if dataset_dir:
        base = Path(dataset_dir)
        if not base.is_absolute():
            base = _PROJECT_ROOT / base
    else:
        base = _PROJECT_ROOT / "data" / "vitoed_new"

    dataset_paths = {
        "train": base / "train.json",
        "dev": base / "dev.json",
        "test": base / "test.json",
    }

    # Resolve examples pool
    examples_pool_path = kwargs.get("examples_pool_path", None)
    if examples_pool_path is None and n_shot > 0:
        examples_pool_path = str(dataset_paths["train"])

    # Build and prepare prompt creator (new prompt_templates refactor API)
    creator = build_prompt_creator(
        technique=technique,
        language=language,
        n_shot=n_shot,
        plus_mode=plus_mode,
        add_method=add_method,
        examples_pool_path=examples_pool_path,
    )
    system_prompt = creator._system_prompt_cache

    # Load each split, pre-compute prompts
    splits = {}
    for split_name, path in dataset_paths.items():
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        records = []
        for sample in raw:
            text = sample["text"]
            sent_id = str(sample["sent_id"])
            _, user_prompt = creator.get_prompt(text, sent_id)

            records.append({
                "sent_id": sample["sent_id"],
                "text": text,
                "opinions_json": json.dumps(sample.get("opinions", []), ensure_ascii=False),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            })
        splits[split_name] = datasets.Dataset.from_list(records)

    return datasets.DatasetDict(splits)


# =========================================================================
# Filter: extract JSON from model response
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
# process_results: per-doc metric computation
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
# Corpus-level aggregation functions
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
        # Legacy behavior (kept only for backward compatibility).
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
