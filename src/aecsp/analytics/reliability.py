"""Pairwise model/human reliability and comparison-set diagnostics."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from aecsp.analytics.agreement import krippendorff_alpha_nominal, pairwise_percent_agreement


def pairwise_dimension_agreement(
    left: pd.DataFrame, right: pd.DataFrame, dimension: str
) -> dict:
    joined = left[["paper_id", dimension]].merge(
        right[["paper_id", dimension]], on="paper_id", suffixes=("_left", "_right")
    )
    a = joined[f"{dimension}_left"].tolist()
    b = joined[f"{dimension}_right"].tolist()
    result = asdict(pairwise_percent_agreement(a, b))
    result["krippendorff_alpha_nominal"] = krippendorff_alpha_nominal(list(map(list, zip(a, b))))
    result["paper_ids"] = joined["paper_id"].tolist()
    return result


def intersection_ids(*frames: pd.DataFrame) -> set[str]:
    if not frames:
        return set()
    sets = [set(frame["paper_id"].astype(str)) for frame in frames]
    return set.intersection(*sets)


def selection_diagnostics(corpus: pd.DataFrame, selected_ids: set[str]) -> dict:
    selected = corpus[corpus["paper_id"].astype(str).isin(selected_ids)]
    lengths = selected.get("Abstract", pd.Series("", index=selected.index)).fillna("").str.split().str.len()
    result = {
        "full_corpus_n": len(corpus),
        "selected_n": len(selected),
        "selected_share": len(selected) / len(corpus) if len(corpus) else 0.0,
        "mean_abstract_words": float(lengths.mean()) if len(lengths) else None,
    }
    for index in range(1, 5):
        column = f"in_query_{index}"
        if column in selected:
            result[f"query_{index}_share"] = float(
                pd.to_numeric(selected[column], errors="coerce").fillna(0).mean()
            )
    years = pd.to_numeric(selected.get("Year"), errors="coerce")
    result["year_median"] = float(years.median()) if years.notna().any() else None
    return result
