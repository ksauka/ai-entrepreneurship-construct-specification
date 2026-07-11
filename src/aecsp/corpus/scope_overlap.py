"""Report overlap among query-derived analytical scopes."""

from __future__ import annotations

import pandas as pd

from aecsp.corpus.scopes import scope_frame


def pairwise_overlap(master: pd.DataFrame, scope_a: str, scope_b: str) -> dict:
    a = set(scope_frame(master, scope_a)["paper_id"].astype(str))
    b = set(scope_frame(master, scope_b)["paper_id"].astype(str))
    intersection = a & b
    union = a | b
    return {
        "scope_a": scope_a,
        "scope_b": scope_b,
        "n_a": len(a),
        "n_b": len(b),
        "n_intersection": len(intersection),
        "n_union": len(union),
        "jaccard": len(intersection) / len(union) if union else 0.0,
    }


def overlap_table(master: pd.DataFrame, scopes: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        pairwise_overlap(master, left, right)
        for index, left in enumerate(scopes)
        for right in scopes[index + 1 :]
    )
