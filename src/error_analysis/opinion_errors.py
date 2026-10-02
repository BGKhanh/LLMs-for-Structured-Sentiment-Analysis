# src/error_analysis/opinion_errors.py
"""E2 — Opinion Structure Errors. Taxonomy v2 §5. Consumes an AlignmentResult
(§2.4) — never redefines what "matched" means."""
import sys
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from semeval22_structured_sentiment.evaluation.evaluate import weighted_score

from .alignment import DEFAULT_TAU, AlignmentResult, OpinionTuple
from .component_errors import COMPONENT_MODES

DUPLICATE = "duplicate_opinion"
MULTI_PRED_AMBIGUOUS = "gold_multi_pred_ambiguous"  # not a defined E2 subtype; flagged for manual review


def opinion_count_error(n_pred: int, n_gold: int) -> int:
    return n_pred - n_gold


def _components_near_identical(a: OpinionTuple, b: OpinionTuple, tau: float) -> bool:
    for component, mode in COMPONENT_MODES.items():
        idx = {"Source": 0, "Target": 1, "Polar_expression": 2}[component]
        a_set, b_set = a[idx], b[idx]
        if len(a_set) == 0 and len(b_set) == 0:
            continue
        if len(a_set) == 0 or len(b_set) == 0:
            return False
        score = min(weighted_score(a, [b], mode=mode), weighted_score(b, [a], mode=mode))
        if score < tau:
            return False
    return True


def detect_duplicates(
    pred_tuples: List[OpinionTuple],
    alignment: AlignmentResult,
    tau: float = DEFAULT_TAU,
) -> Dict[int, str]:
    """For each gold claimed by >1 predicted opinion (§5.3), decide Duplicate
    vs an ambiguous multi-match the taxonomy doesn't literally define."""
    labels = {}
    for g_idx, claimants in alignment.gold_multi_pred.items():
        pairwise_identical = all(
            _components_near_identical(pred_tuples[claimants[i]], pred_tuples[claimants[j]], tau)
            for i in range(len(claimants)) for j in range(i + 1, len(claimants))
        )
        labels[g_idx] = DUPLICATE if pairwise_identical else MULTI_PRED_AMBIGUOUS
    return labels
