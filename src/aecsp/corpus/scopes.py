"""Define and select the canonical overlapping dataset scopes.

Inputs: the master corpus and its query-membership columns.
Outputs: consistent full-corpus and Query 1-4 DataFrame views.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aecsp.corpus.query_provenance import SEARCH_QUERIES


@dataclass(frozen=True)
class DatasetScope:
    """One named analytical view over the master corpus."""

    id: str
    label: str
    filter_column: str | None  # None means no filter (full corpus)


DATASET_SCOPES: tuple[DatasetScope, ...] = (
    DatasetScope("full_corpus", "Full corpus (all papers)", None),
    *(
        DatasetScope(query.id, query.label, query.one_hot_column)
        for query in SEARCH_QUERIES
    ),
)

# Bonus scope: the core AI and entrepreneurship subset flagged in Stage 1.5.
STRICT_AI_ENT_SCOPE = DatasetScope(
    "strict_ai_ent", "Core AI and entrepreneurship", "ai_ent_relevant"
)

SCOPE_BY_ID: dict[str, DatasetScope] = {
    scope.id: scope for scope in (*DATASET_SCOPES, STRICT_AI_ENT_SCOPE)
}


class ScopeError(ValueError):
    """Raised when a scope cannot be resolved against the corpus."""


def scope_frame(master: pd.DataFrame, scope_id: str) -> pd.DataFrame:
    """Return the master-corpus rows belonging to one named scope."""

    scope = SCOPE_BY_ID.get(scope_id)
    if scope is None:
        raise ScopeError(
            f"Unknown scope {scope_id!r}; expected one of {sorted(SCOPE_BY_ID)}"
        )
    if scope.filter_column is None:
        return master.reset_index(drop=True)
    if scope.filter_column not in master.columns:
        raise ScopeError(
            f"Scope {scope_id!r} needs column {scope.filter_column!r}, "
            "which is missing from the corpus"
        )
    flags = pd.to_numeric(master[scope.filter_column], errors="coerce").fillna(0).astype(int)
    return master[flags == 1].reset_index(drop=True)


def iter_scopes(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The five required views, keyed by scope id, for stage output loops."""

    return {scope.id: scope_frame(master, scope.id) for scope in DATASET_SCOPES}
