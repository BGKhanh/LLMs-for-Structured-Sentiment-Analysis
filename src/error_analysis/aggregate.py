# src/error_analysis/aggregate.py
"""Corpus-level quantitative report — taxonomy v2 §14. Every rate states its
own denominator explicitly (the ambiguity §14 calls out in v1)."""
import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .classify import (
    NONE_NO_ERROR,
    PRIMARY_E3_4,
    PRIMARY_E3_E4,
    PRIMARY_E5,
    SampleAnalysis,
)
from .component_errors import TIER_NEAR_MISS, TIER_WRONG
from .output_errors import (
    E1_EMPTY,
    E1_MALFORMED_UNRECOVERABLE,
    E1_TRUNCATED,
    is_raw_malformed,
)

_EPS = 1e-16


@dataclass
class AggregateReport:
    total_samples: int = 0

    # §4.1 / §4.2 / §4.3 — denominator: total_samples
    raw_malformed_rate: float = 0.0
    unrecoverable_rate: float = 0.0
    truncated_rate: float = 0.0
    empty_rate: float = 0.0
    valid_samples: int = 0  # total_samples minus E1-gated (§4 "Quy tắc chung E1")

    # §5 — denominators: predicted / gold opinions (see §14.3)
    n_gold_total: int = 0
    n_pred_total: int = 0
    mean_delta_n: float = 0.0
    missed_opinion_rate: float = 0.0
    spurious_opinion_rate: float = 0.0
    duplicate_count: int = 0
    merge_count: int = 0
    split_count: int = 0
    ambiguous_alignment_count: int = 0  # §2.4 tie-break edge case, not a defined E-category

    # §6/§7/§8 — denominator: matched (aligned) opinion pairs (§14.4)
    n_matched_pairs: int = 0
    component_error_rate: Dict[str, float] = field(default_factory=dict)
    near_miss_rate: float = 0.0  # among span errors only, §7.0/§14.4
    association_error_rate: float = 0.0
    polarity_error_rate: float = 0.0

    # §14.1 — denominator: total_samples, reported separately, never merged in
    pct_flagged_e7: float = 0.0

    primary_error_distribution: Dict[str, int] = field(default_factory=dict)


def build_report(samples: List[Dict[str, Any]], analyses: List[SampleAnalysis]) -> AggregateReport:
    report = AggregateReport(total_samples=len(samples))
    if not samples:
        return report

    report.raw_malformed_rate = sum(is_raw_malformed(s) for s in samples) / len(samples)
    report.unrecoverable_rate = sum(a.e1_status == E1_MALFORMED_UNRECOVERABLE for a in analyses) / len(analyses)
    report.truncated_rate = sum(a.e1_status == E1_TRUNCATED for a in analyses) / len(analyses)
    report.empty_rate = sum(a.e1_status == E1_EMPTY for a in analyses) / len(analyses)

    valid = [a for a in analyses if a.e1_status is None]
    report.valid_samples = len(valid)
    if not valid:
        return report

    report.n_gold_total = sum(a.n_gold for a in valid)
    report.n_pred_total = sum(a.n_pred for a in valid)
    report.mean_delta_n = sum(a.delta_n for a in valid) / len(valid)

    n_missed = sum(len(a.missed_gold_idxs) for a in valid)
    n_spurious = sum(len(a.spurious_pred_idxs) for a in valid)
    report.missed_opinion_rate = n_missed / (report.n_gold_total + _EPS)
    report.spurious_opinion_rate = n_spurious / (report.n_pred_total + _EPS)

    report.duplicate_count = sum(sum(v == "duplicate_opinion" for v in a.duplicate_gold_idxs.values()) for a in valid)
    report.merge_count = sum(len(a.merges) for a in valid)
    report.split_count = sum(len(a.splits) for a in valid)
    report.ambiguous_alignment_count = sum(len(a.ambiguous_gold_idxs) for a in valid)

    all_findings = [f for a in valid for f in a.matched_pair_findings]
    report.n_matched_pairs = len(all_findings)
    if all_findings:
        for component in ("Source", "Target", "Polar_expression"):
            errors = sum(f.component_results[component].has_error for f in all_findings)
            report.component_error_rate[component] = errors / len(all_findings)

        span_error_tiers = [
            r.tier for f in all_findings for r in f.component_results.values() if r.has_error
        ]
        if span_error_tiers:
            report.near_miss_rate = sum(t == TIER_NEAR_MISS for t in span_error_tiers) / len(span_error_tiers)

        report.association_error_rate = sum(f.primary_error == PRIMARY_E5 for f in all_findings) / len(all_findings)
        report.polarity_error_rate = sum(f.polarity_result is not None and f.polarity_result.has_error for f in all_findings) / len(all_findings)

        dist: Dict[str, int] = {}
        for f in all_findings:
            dist[f.primary_error] = dist.get(f.primary_error, 0) + 1
        report.primary_error_distribution = dist

    return report


def write_report_csv(report: AggregateReport, path: str) -> None:
    rows = [
        ("total_samples", report.total_samples),
        ("valid_samples (E1-gated excluded)", report.valid_samples),
        ("raw_malformed_rate (den=total_samples)", f"{report.raw_malformed_rate:.4f}"),
        ("unrecoverable_rate (den=total_samples)", f"{report.unrecoverable_rate:.4f}"),
        ("truncated_rate (den=total_samples)", f"{report.truncated_rate:.4f}"),
        ("empty_rate (den=total_samples)", f"{report.empty_rate:.4f}"),
        ("n_gold_total", report.n_gold_total),
        ("n_pred_total", report.n_pred_total),
        ("mean_delta_n (N_pred - N_gold)", f"{report.mean_delta_n:.4f}"),
        ("missed_opinion_rate (den=n_gold_total)", f"{report.missed_opinion_rate:.4f}"),
        ("spurious_opinion_rate (den=n_pred_total)", f"{report.spurious_opinion_rate:.4f}"),
        ("duplicate_count", report.duplicate_count),
        ("merge_count", report.merge_count),
        ("split_count", report.split_count),
        ("ambiguous_alignment_count (manual review)", report.ambiguous_alignment_count),
        ("n_matched_pairs (den for component/span/polarity/assoc)", report.n_matched_pairs),
        ("source_error_rate", f"{report.component_error_rate.get('Source', 0.0):.4f}"),
        ("target_error_rate", f"{report.component_error_rate.get('Target', 0.0):.4f}"),
        ("expression_error_rate", f"{report.component_error_rate.get('Polar_expression', 0.0):.4f}"),
        ("near_miss_rate (den=span errors only)", f"{report.near_miss_rate:.4f}"),
        ("association_error_rate", f"{report.association_error_rate:.4f}"),
        ("polarity_error_rate", f"{report.polarity_error_rate:.4f}"),
        ("pct_flagged_e7 (den=total_samples, always 0.0 until manually double-annotated)", f"{report.pct_flagged_e7:.4f}"),
    ]
    for label, count in sorted(report.primary_error_distribution.items()):
        rows.append((f"primary_error_distribution[{label}]", count))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
