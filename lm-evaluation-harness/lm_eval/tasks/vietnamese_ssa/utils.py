"""
Bridge module between lm-evaluation-harness and the project's existing code.

Provides all !function references used in vietnamese_ssa.yaml:
- configure(): module-level setup for prompt template (called from wrapper script)
- load_dataset(): load local JSON data as HuggingFace DatasetDict
- doc_to_text(): build user prompt per document
- doc_to_target(): serialize ground truth
- process_results(): compute per-doc SemEval raw scores
- corpus-level aggregation functions for 6 metrics
- extract/filter functions for JSON extraction from model responses
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import datasets

# ---------------------------------------------------------------------------
# Resolve project root so we can import from src/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]  # up from tasks/vietnamese_ssa/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.prompt_templates import (
    FewShotPrompt,
    FewShotCoTPrompt,
    ReReadingPrompt,
    PlanAndSolvePrompt,
    Re2PaSCoTPrompt,
)
from src.utils.postprocessing import extract_json_from_response, postprocess_response

# Evaluation helpers from SemEval scripts
from semeval22_structured_sentiment.evaluation.evaluate import (
    convert_opinion_to_tuple,
    sent_tuples_in_list,
    weighted_score,
)

# =========================================================================
# Module-level state (set once via configure(), read by !function callables)
# =========================================================================
_template = None
_dataset_paths: Dict[str, str] = {
    "train": str(_PROJECT_ROOT / "data" / "vitoed_new" / "train.json"),
    "dev": str(_PROJECT_ROOT / "data" / "vitoed_new" / "dev.json"),
    "test": str(_PROJECT_ROOT / "data" / "vitoed_new" / "test.json"),
}

_EPSILON = 1e-16


# =========================================================================
# configure() - called from wrapper script before simple_evaluate()
# =========================================================================
def configure(
    technique: str,
    language: str = "vi",
    n_shot: int = 0,
    plus_mode: bool = False,
    add_method: str = "none",
    examples_pool_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
) -> str:
    """Setup prompt template and dataset paths. Must be called once before eval.

    Args:
        technique: One of few_shot, few_shot_cot, rereading, plan_and_solve, re2_pas_cot.
        language: 'vi' or 'en'.
        n_shot: Number of few-shot examples.
        plus_mode: PS+ mode for plan_and_solve.
        add_method: Enhancement method for rereading (none, 0_CoT, FewShot, etc.).
        examples_pool_path: Path to examples pool JSON for few-shot techniques.
        dataset_dir: Override dataset directory (default: data/vitoed_new).

    Returns:
        System prompt string (to pass as system_instruction to simple_evaluate).
    """
    global _template, _dataset_paths

    eng = language == "en"

    # Override dataset paths if provided
    if dataset_dir:
        base = Path(dataset_dir)
        if not base.is_absolute():
            base = _PROJECT_ROOT / base
        _dataset_paths = {
            "train": str(base / "train.json"),
            "dev": str(base / "dev.json"),
            "test": str(base / "test.json"),
        }

    # Resolve examples_pool_path for few-shot techniques
    if examples_pool_path is None and n_shot > 0:
        examples_pool_path = _dataset_paths["train"]

    # Factory: create prompt template instance
    if technique == "few_shot":
        _template = FewShotPrompt(
            eng=eng, n_shot=n_shot, examples_pool_path=examples_pool_path
        )
    elif technique == "few_shot_cot":
        _template = FewShotCoTPrompt(eng=eng, n_shot=n_shot)
    elif technique == "rereading":
        _template = ReReadingPrompt(
            eng=eng,
            add_method=add_method,
            n_shot=n_shot,
            examples_pool_path=examples_pool_path,
        )
    elif technique in ("plan_and_solve", "plan_solve"):
        _template = PlanAndSolvePrompt(eng=eng, plus=plus_mode, n_shot=n_shot)
    elif technique == "re2_pas_cot":
        _template = Re2PaSCoTPrompt(eng=eng, n_shot=n_shot)
    else:
        raise ValueError(
            f"Unknown technique: {technique}. "
            "Supported: few_shot, few_shot_cot, rereading, plan_and_solve, re2_pas_cot"
        )

    _template.prepare()
    return _template._system_prompt_cache


# =========================================================================
# Dataset loading (!function reference)
# =========================================================================
def load_dataset(**kwargs) -> datasets.DatasetDict:
    """Load local JSON files as a HuggingFace DatasetDict.

    Called by lm-eval via ``custom_dataset: !function utils.load_dataset``.
    """
    splits = {}
    for split_name, path in _dataset_paths.items():
        p = Path(path)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Flatten opinions to JSON string for HF Dataset compatibility
        records = []
        for sample in raw:
            records.append(
                {
                    "sent_id": sample["sent_id"],
                    "text": sample["text"],
                    "opinions_json": json.dumps(
                        sample.get("opinions", []), ensure_ascii=False
                    ),
                }
            )
        splits[split_name] = datasets.Dataset.from_list(records)

    return datasets.DatasetDict(splits)


# =========================================================================
# doc_to_text / doc_to_target (!function references)
# =========================================================================
def doc_to_text(doc: Dict[str, Any]) -> str:
    """Build user prompt for a single document.

    System prompt is handled separately via ``system_instruction`` parameter.
    """
    if _template is None:
        raise RuntimeError("utils.configure() must be called before evaluation.")
    _, user_prompt = _template.get_prompt(doc["text"], str(doc["sent_id"]))
    return user_prompt


def doc_to_target(doc: Dict[str, Any]) -> str:
    """Return ground truth as a JSON string for logging/reference."""
    return doc["opinions_json"]


# =========================================================================
# Filter: extract JSON from model response
# =========================================================================
def extract_and_postprocess(resps: List[List[str]], docs: List[Dict]) -> List[str]:
    """Custom filter function: extract JSON then normalize to SemEval format.

    Called via filter_list -> function: !function utils.extract_and_postprocess
    """
    filtered = []
    for resp_list, doc in zip(resps, docs):
        raw = resp_list[0] if resp_list else "{}"
        extracted = extract_json_from_response(raw)
        processed = postprocess_response(extracted, doc["text"], doc["sent_id"])
        filtered.append(processed)
    return filtered


# =========================================================================
# process_results: per-doc metric computation
# =========================================================================
def _build_opinion_dict(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a full sample dict from doc fields for evaluate.py functions."""
    return {
        "sent_id": str(doc["sent_id"]),
        "text": doc["text"],
        "opinions": json.loads(doc["opinions_json"]),
    }


def _compute_per_doc_scores(
    gold_tuples: List, pred_tuples: List
) -> Dict[str, Tuple[float, int, int]]:
    """Compute per-doc raw scores for all 6 metrics.

    Returns dict mapping metric_name -> (sum_weighted_tp, num_pred, num_gold).
    """
    modes = {
        "SF1": ("all", True, True),
        "NSF1": ("all", False, True),
        "Holder_F1": ("holder", False, True),
        "Target_F1": ("target", False, True),
        "Exp_F1": ("expression", False, True),
        "Targeted_F1": ("targeted_strict", True, False),
    }

    scores = {}
    for metric_name, (mode, keep_polarity, use_weighted) in modes.items():
        w_tp = 0.0
        for pt in pred_tuples:
            if sent_tuples_in_list(pt, gold_tuples, keep_polarity=keep_polarity, mode=mode):
                if use_weighted:
                    w_tp += weighted_score(pt, gold_tuples, mode=mode)
                else:
                    w_tp += 1.0

        scores[metric_name] = (w_tp, len(pred_tuples), len(gold_tuples))

    return scores


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Any]:
    """Compute per-document raw scores for corpus-level aggregation.

    Called by lm-eval after filtering. Each metric key maps to a tuple
    ``(weighted_tp, num_pred, num_gold)`` that the custom aggregation
    functions will sum across all documents.
    """
    response = results[0] if results else "{}"

    # Build gold sample
    gold_sample = _build_opinion_dict(doc)
    gold_tuples = convert_opinion_to_tuple(gold_sample)

    # Build pred sample from model response
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

    return _compute_per_doc_scores(gold_tuples, pred_tuples)


# =========================================================================
# Corpus-level aggregation functions (registered in metric_list via !function)
# =========================================================================
def _corpus_f1(items: List[Tuple[float, int, int]]) -> float:
    """Generic corpus-level F1 from per-doc (weighted_tp, num_pred, num_gold) tuples."""
    total_wtp = sum(x[0] for x in items)
    total_pred = sum(x[1] for x in items)
    total_gold = sum(x[2] for x in items)

    precision = total_wtp / (total_pred + _EPSILON)
    recall = total_wtp / (total_gold + _EPSILON)
    f1 = 2 * precision * recall / (precision + recall + _EPSILON)
    return f1


def sf1_agg(items):
    """Corpus-level Sentiment Graph F1 (SF1)."""
    return _corpus_f1(items)


def nsf1_agg(items):
    """Corpus-level Non-polar SF1 (NSF1)."""
    return _corpus_f1(items)


def holder_f1_agg(items):
    """Corpus-level Holder F1."""
    return _corpus_f1(items)


def target_f1_agg(items):
    """Corpus-level Target F1."""
    return _corpus_f1(items)


def exp_f1_agg(items):
    """Corpus-level Expression F1."""
    return _corpus_f1(items)


def targeted_f1_agg(items):
    """Corpus-level Targeted F1."""
    return _corpus_f1(items)
