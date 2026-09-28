# src/error_analysis/output_errors.py
"""E1 — Output / Generation Errors. Taxonomy v2 §4."""
import json
from typing import Any, Dict, Optional

from . import record as rec

E1_TRUNCATED = "truncated"
E1_MALFORMED_UNRECOVERABLE = "malformed_unrecoverable"
E1_EMPTY = "empty"

TRUNCATION_LEN_RATIO = 0.95  # §4.2 condition 2, "≥ 95%"


def _is_schema_valid_json(text: str) -> bool:
    """Loose structural check for the expected `{"opinions": [...]}` shape.

    Deliberately lenient about per-opinion field shapes (postprocess_response
    already tolerates several variants) — this only asks "did the model
    produce parseable, schema-shaped JSON", not "is every span correct".
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict) or "opinions" not in parsed:
        return False
    opinions = parsed["opinions"]
    if not isinstance(opinions, list):
        return False
    return all(isinstance(op, dict) for op in opinions)


def _is_balanced_json_fragment(text: str) -> bool:
    depth = 0
    seen_open = False
    for ch in text:
        if ch == "{":
            depth += 1
            seen_open = True
        elif ch == "}":
            depth -= 1
    return seen_open and depth == 0


def is_raw_malformed(sample: Dict[str, Any]) -> bool:
    """§4.1 Raw Malformed Rate numerator — measures the LLM itself, pre-pipeline-fix."""
    return not _is_schema_valid_json(rec.raw_response(sample))


def is_unrecoverable(sample: Dict[str, Any]) -> bool:
    """§4.1 Unrecoverable Rate numerator — measures the post-processing pipeline's limits."""
    opinions = rec.predicted_opinions(rec.filtered_response(sample))
    return opinions is None


def is_truncated(sample: Dict[str, Any]) -> bool:
    """§4.2 — all three conditions must hold simultaneously."""
    raw = rec.raw_response(sample)
    args = rec.gen_args(sample)
    until = args.get("until") or []
    max_gen_toks = args.get("max_gen_toks")

    ends_with_stop = any(raw.endswith(u) for u in until) if until else False
    if ends_with_stop:
        return False

    if not max_gen_toks:
        return False
    approx_tokens = len(raw.split())
    if approx_tokens < TRUNCATION_LEN_RATIO * max_gen_toks:
        return False

    return not _is_balanced_json_fragment(raw)


def is_empty_unusable(sample: Dict[str, Any]) -> bool:
    """§4.3 — checked only after truncation/malformed have been ruled out."""
    opinions = rec.predicted_opinions(rec.filtered_response(sample))
    if opinions is None:
        return False
    if opinions:
        return False
    return not rec.raw_response(sample).strip()


def classify_output_status(sample: Dict[str, Any]) -> Optional[str]:
    """Fixed check order per §4: Truncated -> Malformed(unrecoverable) -> Empty.

    Returns the E1 label that gates E2-E5 for this sample (§4 "Quy tắc chung
    E1"), or None if the output is reliable enough for semantic analysis.
    Raw-malformed is tracked as a separate corpus-level metric (§4.1) and is
    NOT gating on its own — only unrecoverable output blocks E2-E5.
    """
    if is_truncated(sample):
        return E1_TRUNCATED
    if is_unrecoverable(sample):
        return E1_MALFORMED_UNRECOVERABLE
    if is_empty_unusable(sample):
        return E1_EMPTY
    return None
