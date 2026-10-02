# src/error_analysis/classify.py
"""Per-sample orchestration: runs §2.4 alignment, then E1-E5 in the fixed
priority order from §13.1 so each predicted/gold opinion gets exactly one
primary_error (avoids the double-counting §2.3 warns against)."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from semeval22_structured_sentiment.evaluation.evaluate import convert_opinion_to_tuple, set_tokenizer

from . import record as rec
from .alignment import DEFAULT_TAU, OpinionTuple, align_opinions
from .component_errors import COMPONENT_TUPLE_IDX, analyze_component, analyze_polarity
from .opinion_errors import detect_duplicates, opinion_count_error
from .output_errors import classify_output_status
from .structural_errors import detect_association_error, detect_merges, detect_splits

PRIMARY_E1 = "E1_output"
PRIMARY_E2_MISSED = "E2_missed_opinion"
PRIMARY_E2_SPURIOUS = "E2_spurious_opinion"
PRIMARY_E2_DUPLICATE = "E2_duplicate_opinion"
PRIMARY_E2_MERGE = "E2_opinion_merge"
PRIMARY_E2_SPLIT = "E2_opinion_split"
PRIMARY_E5 = "E5_association"
PRIMARY_E3_E4 = "E3_E4_component_span"
PRIMARY_E3_4 = "E3.4_polarity"
NONE_NO_ERROR = "no_error"
FLAG_AMBIGUOUS = "unresolved_alignment_ambiguous"  # not an E-category; see alignment.contested_gold


@dataclass
class MatchedPairFinding:
    pred_idx: int
    gold_idx: int
    primary_error: str
    component_results: Dict[str, Any] = field(default_factory=dict)
    polarity_result: Optional[Any] = None
    association_detail: Optional[Any] = None


@dataclass
class SampleAnalysis:
    sent_id: str
    dataset: str
    language: str
    e1_status: Optional[str]
    n_gold: int = 0
    n_pred: int = 0
    delta_n: int = 0
    missed_gold_idxs: List[int] = field(default_factory=list)
    spurious_pred_idxs: List[int] = field(default_factory=list)
    duplicate_gold_idxs: Dict[int, str] = field(default_factory=dict)
    merges: List[Any] = field(default_factory=list)
    splits: List[Any] = field(default_factory=list)
    ambiguous_gold_idxs: List[int] = field(default_factory=list)
    matched_pair_findings: List[MatchedPairFinding] = field(default_factory=list)


def _gold_tuples_from_doc(sample: Dict[str, Any]) -> List[OpinionTuple]:
    sent = {"text": rec.doc_text(sample), "opinions": rec.gold_opinions(sample)}
    return convert_opinion_to_tuple(sent)


def _pred_tuples_from_filtered(sample: Dict[str, Any], opinions: List[Dict[str, Any]]) -> List[OpinionTuple]:
    sent = {"text": rec.doc_text(sample), "opinions": opinions}
    return convert_opinion_to_tuple(sent)


def classify_matched_pair(
    pred_idx: int,
    gold_idx: int,
    pred_tuples: List[OpinionTuple],
    gold_tuples: List[OpinionTuple],
    doc_text: str,
    pred_opinion_raw: Optional[Dict[str, Any]] = None,
    tau: float = DEFAULT_TAU,
) -> MatchedPairFinding:
    """Always computes component + polarity results (cheap, and needed for
    corpus-level metrics like the SF1-vs-NSF1 polarity signal, §6.4), but the
    `primary_error` label follows the §13.1 priority order so no instance is
    double-counted in the primary-error distribution (§14.1)."""
    pred_texts = pred_positions = None
    if pred_opinion_raw:
        pred_texts, pred_positions = {}, {}
        for component in COMPONENT_TUPLE_IDX:
            comp_data = pred_opinion_raw.get(component, [[], []])
            if isinstance(comp_data, list) and len(comp_data) == 2:
                pred_texts[component] = comp_data[0]
                pred_positions[component] = comp_data[1]

    component_results = {
        component: analyze_component(
            pred_tuples[pred_idx], gold_tuples[gold_idx], component,
            pred_component_texts=pred_texts, doc_text=doc_text,
            pred_component_positions=pred_positions, tau=tau,
        )
        for component in COMPONENT_TUPLE_IDX
    }
    polarity = analyze_polarity(pred_tuples[pred_idx], gold_tuples[gold_idx])
    assoc = detect_association_error(pred_idx, gold_idx, pred_tuples, gold_tuples, tau=tau)

    if assoc is not None:
        primary = PRIMARY_E5
    elif any(r.has_error for r in component_results.values()):
        primary = PRIMARY_E3_E4
    elif polarity.has_error:
        primary = PRIMARY_E3_4
    else:
        primary = NONE_NO_ERROR

    return MatchedPairFinding(pred_idx, gold_idx, primary, component_results, polarity, assoc)


def classify_sample(sample: Dict[str, Any], tau: float = DEFAULT_TAU) -> SampleAnalysis:
    sent_id = rec.doc_sent_id(sample)
    dataset = rec.doc_dataset(sample)
    language = rec.doc_language(sample)

    e1_status = classify_output_status(sample)
    if e1_status is not None:
        return SampleAnalysis(sent_id, dataset, language, e1_status)

    set_tokenizer(language)
    gold_tuples = _gold_tuples_from_doc(sample)
    pred_opinions = rec.predicted_opinions(rec.filtered_response(sample)) or []
    pred_tuples = _pred_tuples_from_filtered(sample, pred_opinions)
    doc_text = rec.doc_text(sample)

    alignment = align_opinions(gold_tuples, pred_tuples)

    merges, merge_gold, merge_pred = detect_merges(
        pred_tuples, gold_tuples, alignment.missed_gold_idxs, alignment.spurious_pred_idxs, tau=tau
    )
    remaining_missed = [g for g in alignment.missed_gold_idxs if g not in merge_gold]
    remaining_spurious = [p for p in alignment.spurious_pred_idxs if p not in merge_pred]

    splits, split_gold, split_pred = detect_splits(
        pred_tuples, gold_tuples, remaining_missed, remaining_spurious, tau=tau
    )
    remaining_missed = [g for g in remaining_missed if g not in split_gold]
    remaining_spurious = [p for p in remaining_spurious if p not in split_pred]

    duplicates = detect_duplicates(pred_tuples, alignment, tau=tau)

    matched_findings = [
        classify_matched_pair(
            p_idx, g_idx, pred_tuples, gold_tuples, doc_text,
            pred_opinion_raw=pred_opinions[p_idx] if p_idx < len(pred_opinions) else None,
            tau=tau,
        )
        for p_idx, g_idx in alignment.matched_pairs
    ]

    return SampleAnalysis(
        sent_id=sent_id,
        dataset=dataset,
        language=language,
        e1_status=None,
        n_gold=len(gold_tuples),
        n_pred=len(pred_tuples),
        delta_n=opinion_count_error(len(pred_tuples), len(gold_tuples)),
        missed_gold_idxs=remaining_missed,
        spurious_pred_idxs=remaining_spurious,
        duplicate_gold_idxs=duplicates,
        merges=merges,
        splits=splits,
        ambiguous_gold_idxs=list(alignment.contested_gold.keys()),
        matched_pair_findings=matched_findings,
    )
