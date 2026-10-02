# src/error_analysis/structural_errors.py
"""Opinion Merge/Split (§5.4/§5.5) and Association errors (E5, §8), resolved
in the fixed order required by §8.3: Merge -> Split -> Wrong Association.

Merge/Split operate on the *unresolved* pool (truly missed golds / truly
spurious preds — i.e. opinions that failed the alignment "all" check
entirely, §2.4). Association operates only on already-*matched* pairs. The
two pools never overlap, so the §8.3 ordering is naturally respected: a
predicted opinion cannot simultaneously be "spurious" (candidate for
Merge/Split) and "matched" (candidate for Association).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from semeval22_structured_sentiment.evaluation.evaluate import weighted_score

from .alignment import DEFAULT_TAU, OpinionTuple
from .component_errors import COMPONENT_MODES, COMPONENT_TUPLE_IDX

MERGE_COMPONENTS = ("Target", "Polar_expression")  # §5.4 examples use these two
SPLIT_COMPONENTS = ("Source", "Target", "Polar_expression")

WRONG_ASSOCIATION = "wrong_association"
CROSS_OPINION_CONTAMINATION = "cross_opinion_contamination"


@dataclass
class MergeEvent:
    pred_idx: int
    component: str
    gold_idxs: List[int]


@dataclass
class SplitEvent:
    gold_idx: int
    component_to_pred: Dict[str, int]


@dataclass
class AssociationEvent:
    pred_idx: int
    assigned_gold_idx: int
    type: str
    misattributed: Dict[str, Tuple[int, float]]  # component -> (source_gold_idx, score)


def detect_merges(
    pred_tuples: List[OpinionTuple],
    gold_tuples: List[OpinionTuple],
    missed_gold_idxs: List[int],
    spurious_pred_idxs: List[int],
    tau: float = DEFAULT_TAU,
) -> Tuple[List[MergeEvent], Set[int], Set[int]]:
    events: List[MergeEvent] = []
    consumed_gold: Set[int] = set()
    consumed_pred: Set[int] = set()

    for p_idx in spurious_pred_idxs:
        p = pred_tuples[p_idx]
        for component in MERGE_COMPONENTS:
            idx = COMPONENT_TUPLE_IDX[component]
            mode = COMPONENT_MODES[component]
            if len(p[idx]) == 0:
                continue
            covered = [
                g_idx for g_idx in missed_gold_idxs
                if g_idx not in consumed_gold
                and len(gold_tuples[g_idx][idx]) > 0
                and weighted_score(gold_tuples[g_idx], [p], mode=mode) >= tau
            ]
            if len(covered) >= 2:
                events.append(MergeEvent(p_idx, component, covered))
                consumed_gold.update(covered)
                consumed_pred.add(p_idx)

    return events, consumed_gold, consumed_pred


def detect_splits(
    pred_tuples: List[OpinionTuple],
    gold_tuples: List[OpinionTuple],
    missed_gold_idxs: List[int],
    spurious_pred_idxs: List[int],
    tau: float = DEFAULT_TAU,
) -> Tuple[List[SplitEvent], Set[int], Set[int]]:
    events: List[SplitEvent] = []
    consumed_gold: Set[int] = set()
    consumed_pred: Set[int] = set()

    for g_idx in missed_gold_idxs:
        if g_idx in consumed_gold:
            continue
        g = gold_tuples[g_idx]
        needed = [c for c in SPLIT_COMPONENTS if len(g[COMPONENT_TUPLE_IDX[c]]) > 0]
        if not needed:
            continue

        contributors: Dict[str, int] = {}
        for component in needed:
            idx = COMPONENT_TUPLE_IDX[component]
            mode = COMPONENT_MODES[component]
            best_p, best_cov = None, 0.0
            for p_idx in spurious_pred_idxs:
                if p_idx in consumed_pred:
                    continue
                p = pred_tuples[p_idx]
                if len(p[idx]) == 0:
                    continue
                cov = weighted_score(g, [p], mode=mode)
                if cov > best_cov:
                    best_p, best_cov = p_idx, cov
            if best_p is not None and best_cov >= tau:
                contributors[component] = best_p

        contributing_preds = set(contributors.values())
        if len(contributing_preds) >= 2 and set(contributors.keys()) == set(needed):
            events.append(SplitEvent(g_idx, contributors))
            consumed_gold.add(g_idx)
            consumed_pred.update(contributing_preds)

    return events, consumed_gold, consumed_pred


def detect_association_error(
    pred_idx: int,
    assigned_gold_idx: int,
    pred_tuples: List[OpinionTuple],
    gold_tuples: List[OpinionTuple],
    tau: float = DEFAULT_TAU,
) -> Optional[AssociationEvent]:
    p = pred_tuples[pred_idx]
    assigned = gold_tuples[assigned_gold_idx]
    misattributed: Dict[str, Tuple[int, float]] = {}

    for component, mode in COMPONENT_MODES.items():
        idx = COMPONENT_TUPLE_IDX[component]
        if len(p[idx]) == 0:
            continue

        assigned_score = min(
            weighted_score(p, [assigned], mode=mode),
            weighted_score(assigned, [p], mode=mode),
        )
        best_g_idx, best_score = assigned_gold_idx, assigned_score
        for g_idx, g in enumerate(gold_tuples):
            if g_idx == assigned_gold_idx or len(g[idx]) == 0:
                continue
            score = min(weighted_score(p, [g], mode=mode), weighted_score(g, [p], mode=mode))
            if score > best_score:
                best_g_idx, best_score = g_idx, score

        if best_g_idx != assigned_gold_idx and best_score >= tau and best_score > assigned_score:
            misattributed[component] = (best_g_idx, best_score)

    if not misattributed:
        return None

    distinct_sources = {g_idx for g_idx, _ in misattributed.values()} | {assigned_gold_idx}
    error_type = WRONG_ASSOCIATION if (len(misattributed) >= 2 and len(distinct_sources) >= 2) else CROSS_OPINION_CONTAMINATION
    return AssociationEvent(pred_idx, assigned_gold_idx, error_type, misattributed)
