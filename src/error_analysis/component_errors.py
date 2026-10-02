# src/error_analysis/component_errors.py
"""E3 — Component Extraction Errors (§6) and E4 — Span/Surface Realization
Errors (§7), applied only to opinion pairs already confirmed `matched` by
the Opinion Alignment Procedure (§2.4) — never to Missed/Spurious opinions.

Structural note (non-obvious, worth keeping in mind when reading results):
alignment (§2.4) requires token-overlap > 0 on Source, Target AND Expression
simultaneously (`mode="all"`). Consequently a "matched" pair can, by
construction, never have a component that is fully missing on one side and
present on the other (that pair would have failed alignment and surfaced as
Missed/Spurious at the opinion level, §5.1/§5.2, instead). The
`missing`/`implicit_filled` subtypes below are still implemented for
robustness/completeness, but in practice will almost never fire — component
errors on matched pairs are overwhelmingly span-quality errors (§7).
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from semeval22_structured_sentiment.evaluation.evaluate import weighted_score

from .alignment import DEFAULT_TAU, OpinionTuple

COMPONENT_MODES = {"Source": "holder", "Target": "target", "Polar_expression": "expression"}
COMPONENT_TUPLE_IDX = {"Source": 0, "Target": 1, "Polar_expression": 2}

TIER_EXACT = "exact"
TIER_NEAR_MISS = "near_miss"
TIER_WRONG = "wrong"

SPAN_MISSING = "missing"                # gold has it, predicted doesn't (see module note)
SPAN_IMPLICIT_FILLED = "implicit_filled"  # gold empty, predicted invented one
SPAN_UNDER_EXTRACTION = "under_extraction"      # §7.1
SPAN_OVER_EXTRACTION = "over_extraction"        # §7.2
SPAN_WRONG_NO_OVERLAP = "wrong_span_no_overlap"     # §7.3
SPAN_WRONG_PARTIAL = "wrong_span_partial_overlap"   # §7.3 (boundary mismatch, neither side contains the other)
SPAN_PARAPHRASED = "paraphrased_non_source_span"    # §7.4
SPAN_INVALID_OFFSET = "invalid_unsupported_span"    # §7.5


@dataclass
class ComponentErrorResult:
    component: str
    has_error: bool
    tier: str
    span_subtype: Optional[str]
    overlap: float  # min(precision-direction, recall-direction) weighted_score, see below


@dataclass
class PolarityErrorResult:
    has_error: bool
    gold_polarity: str
    pred_polarity: str
    transition: Optional[str]


def _symmetric_overlap(pred_tuple: OpinionTuple, gold_tuple: OpinionTuple, mode: str) -> Tuple[float, float]:
    """Returns (precision_dir, recall_dir).

    §7.0 as literally written scores only `weighted_score(pred, [gold])`
    (precision-direction: |gold∩pred|/|pred|), which is 1.0 for ANY
    under-extraction (pred fully contained in gold) — that would silently
    contradict §7.1's own worked example. We therefore also take the
    recall-direction score and use the minimum of the two as the tier score,
    so a real under- or over-extraction is never mistaken for "no error".
    Both calls reuse the same scorer primitive per §2.4's design goal.
    """
    precision_dir = weighted_score(pred_tuple, [gold_tuple], mode=mode)
    recall_dir = weighted_score(gold_tuple, [pred_tuple], mode=mode)
    return precision_dir, recall_dir


def analyze_component(
    pred_tuple: OpinionTuple,
    gold_tuple: OpinionTuple,
    component: str,
    pred_component_texts: Optional[Dict[str, List[str]]] = None,
    doc_text: str = "",
    pred_component_positions: Optional[Dict[str, List[str]]] = None,
    tau: float = DEFAULT_TAU,
) -> ComponentErrorResult:
    idx = COMPONENT_TUPLE_IDX[component]
    mode = COMPONENT_MODES[component]
    gold_set, pred_set = gold_tuple[idx], pred_tuple[idx]

    if len(gold_set) == 0 and len(pred_set) == 0:
        return ComponentErrorResult(component, False, TIER_EXACT, None, 1.0)

    if len(gold_set) > 0 and len(pred_set) == 0:
        return ComponentErrorResult(component, True, TIER_WRONG, SPAN_MISSING, 0.0)
    if len(gold_set) == 0 and len(pred_set) > 0:
        return ComponentErrorResult(component, True, TIER_WRONG, SPAN_IMPLICIT_FILLED, 0.0)

    if gold_set == pred_set:
        return ComponentErrorResult(component, False, TIER_EXACT, None, 1.0)

    # §7.4 Paraphrased/Non-source span: predicted text not found verbatim in doc_text.
    if pred_component_texts is not None and doc_text:
        texts = pred_component_texts.get(component, [])
        if texts and any(t not in doc_text for t in texts if t):
            return ComponentErrorResult(component, True, TIER_WRONG, SPAN_PARAPHRASED, 0.0)

    # §7.5 Invalid/unsupported span: offset out of bounds of the source text.
    if pred_component_positions is not None and doc_text:
        for pos in pred_component_positions.get(component, []):
            try:
                b, e = map(int, pos.split(":"))
            except ValueError:
                continue
            if b < 0 or e > len(doc_text) or b > e:
                return ComponentErrorResult(component, True, TIER_WRONG, SPAN_INVALID_OFFSET, 0.0)

    precision_dir, recall_dir = _symmetric_overlap(pred_tuple, gold_tuple, mode)
    tier_score = min(precision_dir, recall_dir)

    if tier_score >= 1.0:
        return ComponentErrorResult(component, False, TIER_EXACT, None, tier_score)
    if tier_score >= tau:
        return ComponentErrorResult(component, True, TIER_NEAR_MISS, None, tier_score)

    if precision_dir >= 1.0 and recall_dir < 1.0:
        subtype = SPAN_UNDER_EXTRACTION
    elif recall_dir >= 1.0 and precision_dir < 1.0:
        subtype = SPAN_OVER_EXTRACTION
    elif precision_dir == 0.0 and recall_dir == 0.0:
        subtype = SPAN_WRONG_NO_OVERLAP
    else:
        subtype = SPAN_WRONG_PARTIAL

    return ComponentErrorResult(component, True, TIER_WRONG, subtype, tier_score)


_VALID_POLARITIES = {"Positive", "Negative", "Neutral"}


def analyze_polarity(pred_tuple: OpinionTuple, gold_tuple: OpinionTuple) -> PolarityErrorResult:
    gold_pol, pred_pol = gold_tuple[3], pred_tuple[3]
    if gold_pol == pred_pol:
        return PolarityErrorResult(False, gold_pol, pred_pol, None)
    return PolarityErrorResult(True, gold_pol, pred_pol, f"{gold_pol or '?'}->{pred_pol or '?'}")
