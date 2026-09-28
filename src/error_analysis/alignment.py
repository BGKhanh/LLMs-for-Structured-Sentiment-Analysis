# src/error_analysis/alignment.py
"""Opinion Alignment Procedure — taxonomy v2 §2.4.

Reuses the exact primitives that already produce the reported SF1/NSF1
(`sent_tuples_in_list`, `weighted_score`) so alignment never diverges from
the metrics published for a run.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from semeval22_structured_sentiment.evaluation.evaluate import (
    sent_tuples_in_list,
    weighted_score,
)

OpinionTuple = Tuple[frozenset, frozenset, frozenset, str]

DEFAULT_TAU = 0.5  # §7.0 near-miss threshold, also reused as the Merge/Split coverage bar (§5.4/§5.5)


@dataclass
class AlignmentResult:
    pred_to_gold: Dict[int, Optional[int]]
    gold_to_pred: Dict[int, Optional[int]]
    # gold claimed as "best match" by >1 predicted opinion -> §5.3 Duplicate candidates
    gold_multi_pred: Dict[int, List[int]] = field(default_factory=dict)
    # gold that fully satisfies mode="all" with some pred, but lost the greedy
    # best-score tie-break to another gold -> ambiguous, not literally covered
    # by any single E2 subtype; flagged for manual review (see §16 qualitative).
    contested_gold: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def matched_pairs(self) -> List[Tuple[int, int]]:
        return [(p, g) for p, g in self.pred_to_gold.items() if g is not None]

    @property
    def missed_gold_idxs(self) -> List[int]:
        return [g for g, p in self.gold_to_pred.items() if p is None and g not in self.contested_gold]

    @property
    def spurious_pred_idxs(self) -> List[int]:
        return [p for p, g in self.pred_to_gold.items() if g is None]


def align_opinions(gold_tuples: List[OpinionTuple], pred_tuples: List[OpinionTuple]) -> AlignmentResult:
    # Candidates: which golds each predicted opinion structurally matches (mode="all").
    pred_candidates: Dict[int, List[int]] = {
        p_idx: [g_idx for g_idx, g in enumerate(gold_tuples)
                if sent_tuples_in_list(p, [g], keep_polarity=False, mode="all")]
        for p_idx, p in enumerate(pred_tuples)
    }

    pred_to_gold: Dict[int, Optional[int]] = {}
    gold_claimed_by: Dict[int, List[int]] = {g: [] for g in range(len(gold_tuples))}
    for p_idx, cands in pred_candidates.items():
        if not cands:
            pred_to_gold[p_idx] = None
            continue
        best_g = max(cands, key=lambda g_idx: weighted_score(pred_tuples[p_idx], [gold_tuples[g_idx]], mode="all"))
        pred_to_gold[p_idx] = best_g
        gold_claimed_by[best_g].append(p_idx)

    gold_to_pred: Dict[int, Optional[int]] = {}
    gold_multi_pred: Dict[int, List[int]] = {}
    contested_gold: Dict[int, List[int]] = {}
    for g_idx in range(len(gold_tuples)):
        claimants = gold_claimed_by[g_idx]
        if claimants:
            gold_to_pred[g_idx] = claimants[0]
            if len(claimants) > 1:
                gold_multi_pred[g_idx] = claimants
        else:
            other_matchers = [p_idx for p_idx, cands in pred_candidates.items() if g_idx in cands]
            gold_to_pred[g_idx] = None
            if other_matchers:
                contested_gold[g_idx] = other_matchers

    return AlignmentResult(
        pred_to_gold=pred_to_gold,
        gold_to_pred=gold_to_pred,
        gold_multi_pred=gold_multi_pred,
        contested_gold=contested_gold,
    )
