"""Layer 5: construct convergence, divergence, and contrast analytics.

Operates on the paper-level specification codes (Stage 2A.5 columns). Every
statistic is traceable: functions return the contributing ``paper_id`` lists
alongside the scores so the platform can show the evidence behind any number.

Convergence for a group = how concentrated its papers are on each of the seven
specification dimensions (1.0 = all papers agree, 0.0 = maximally fragmented),
reusing the entropy-based :func:`summarize_distribution`.

Construct contrast = papers that agree on one dimension but differ on another
(e.g. same AI type, different role) — the core theoretical signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from aecsp.analytics.metrics import summarize_distribution
from aecsp.specification.schema import SPECIFICATION_DIMENSIONS

DIMENSION_COLUMNS: tuple[str, ...] = tuple(d.column for d in SPECIFICATION_DIMENSIONS)


@dataclass
class GroupConvergence:
    """Convergence profile for one group of papers (journal, topic, query...)."""

    group: str
    paper_count: int
    dimension_scores: dict[str, float]
    overall_specification_clarity_score: float
    fragmentation_score: float
    dominant_values: dict[str, str | None]
    paper_ids: list[str] = field(default_factory=list)


def group_convergence(frame: pd.DataFrame, group_label: str = "all") -> GroupConvergence:
    """Convergence across all seven dimensions for one set of papers."""

    scores: dict[str, float] = {}
    dominant: dict[str, str | None] = {}
    for column in DIMENSION_COLUMNS:
        values = frame[column].tolist() if column in frame.columns else []
        metric = summarize_distribution(values)
        scores[column] = round(metric.convergence_score, 4)
        dominant[column] = metric.dominant_value

    present = [scores[c] for c in DIMENSION_COLUMNS if c in frame.columns]
    overall = round(sum(present) / len(present), 4) if present else 0.0
    return GroupConvergence(
        group=group_label,
        paper_count=len(frame),
        dimension_scores=scores,
        overall_specification_clarity_score=overall,
        fragmentation_score=round(1.0 - overall, 4),
        dominant_values=dominant,
        paper_ids=frame["paper_id"].tolist() if "paper_id" in frame.columns else [],
    )


def convergence_by(
    frame: pd.DataFrame, group_column: str, min_papers: int = 2
) -> pd.DataFrame:
    """One convergence row per group value (e.g. per journal, topic, or query).

    Groups with fewer than ``min_papers`` papers are dropped (a single paper
    trivially "converges" with itself).
    """

    if group_column not in frame.columns:
        raise KeyError(f"group column {group_column!r} not in frame")

    rows: list[dict] = []
    for value, sub in frame.groupby(group_column):
        if not str(value).strip() or len(sub) < min_papers:
            continue
        conv = group_convergence(sub, group_label=str(value))
        row = {
            group_column: value,
            "paper_count": conv.paper_count,
            "overall_specification_clarity_score": conv.overall_specification_clarity_score,
            "fragmentation_score": conv.fragmentation_score,
        }
        for column, score in conv.dimension_scores.items():
            row[f"{column}_convergence_score"] = score
            row[f"{column}_dominant"] = conv.dominant_values[column]
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("fragmentation_score", ascending=False).reset_index(drop=True)
    return result


def construct_contrast(
    frame: pd.DataFrame, shared_column: str, contrast_column: str
) -> pd.DataFrame:
    """Find groups that share one dimension value but differ on another.

    Returns one row per shared value that shows >1 distinct contrast value,
    with counts and example paper_ids per contrast value for evidence panels.
    A high contrast_value_count means the same, e.g., AI type is being given
    many different roles/mechanisms — a construct contrast hotspot.
    """

    for column in (shared_column, contrast_column):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} not in frame")

    rows: list[dict] = []
    for shared_value, sub in frame.groupby(shared_column):
        if not str(shared_value).strip():
            continue
        contrast_values = sub[contrast_column][sub[contrast_column].astype(str).str.strip() != ""]
        distinct = contrast_values.nunique()
        if distinct < 2:
            continue
        by_value = {
            str(cv): grp["paper_id"].tolist() if "paper_id" in grp.columns else []
            for cv, grp in sub.groupby(contrast_column)
            if str(cv).strip()
        }
        rows.append(
            {
                "shared_dimension": shared_column,
                "shared_value": shared_value,
                "contrast_dimension": contrast_column,
                "contrast_value_count": distinct,
                "paper_count": len(sub),
                "contrast_breakdown": {k: len(v) for k, v in by_value.items()},
                "example_paper_ids": {k: v[:5] for k, v in by_value.items()},
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("contrast_value_count", ascending=False).reset_index(drop=True)
    return result
