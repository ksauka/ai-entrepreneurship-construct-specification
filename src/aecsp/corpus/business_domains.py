"""Registered query-domain assignments used in domain contrasting."""

from __future__ import annotations

import pandas as pd


REGISTERED_QUERY_DOMAIN_RULES: tuple[dict[str, str], ...] = (
    {
        "domain_id": "ft50",
        "domain_label": "FT50",
        "query_id": "query_2",
        "flag_column": "in_query_2",
    },
    {
        "domain_id": "core_entrepreneurship",
        "domain_label": "Core entrepreneurship",
        "query_id": "query_3",
        "flag_column": "in_query_3",
    },
    {
        "domain_id": "other_entrepreneurship",
        "domain_label": "Additional entrepreneurship",
        "query_id": "query_4",
        "flag_column": "in_query_4",
    },
)

ENTREPRENEURSHIP_DOMAIN_RULES: tuple[dict[str, str], ...] = tuple(
    rule
    for rule in REGISTERED_QUERY_DOMAIN_RULES
    if rule["domain_id"] in {"core_entrepreneurship", "other_entrepreneurship"}
)


def _flag(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    invalid = ~values.isin([0, 1])
    if invalid.any():
        raise ValueError("Domain membership flags must contain only 0 or 1")
    return values.eq(1)


def _build_assignments(
    corpus: pd.DataFrame,
    rules: tuple[dict[str, str], ...],
) -> pd.DataFrame:
    """Return one row per paper and registered query-defined domain."""

    required = {
        "paper_id",
        "Source title",
        *(rule["flag_column"] for rule in rules),
    }
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Corpus is missing domain fields: {sorted(missing)}")
    if corpus["paper_id"].duplicated().any():
        raise ValueError("paper_id must be unique before domain assignment")

    masks = {
        rule["domain_id"]: _flag(corpus[rule["flag_column"]])
        for rule in rules
    }
    overlap = masks["core_entrepreneurship"] & masks["other_entrepreneurship"]
    if overlap.any():
        raise ValueError(
            f"Core and other entrepreneurship overlap for {int(overlap.sum())} papers"
        )

    assignments: list[pd.DataFrame] = []
    for rule in rules:
        selected = corpus.loc[
            masks[rule["domain_id"]], ["paper_id", "Source title"]
        ].copy()
        selected = selected.rename(columns={"Source title": "source_title"})
        selected["domain_id"] = rule["domain_id"]
        selected["domain_label"] = rule["domain_label"]
        selected["assignment_basis"] = rule["query_id"]
        assignments.append(selected)

    result = pd.concat(assignments, ignore_index=True)
    if result.duplicated(["paper_id", "domain_id"]).any():
        raise ValueError("Duplicate paper-domain assignments were generated")
    return result.sort_values(
        ["domain_id", "source_title", "paper_id"], kind="stable"
    ).reset_index(drop=True)


def build_registered_query_domain_assignments(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Build FT50 and the two registered entrepreneurship domains.

    FT50 can overlap either entrepreneurship domain. Core and other
    entrepreneurship must remain disjoint because Queries 3 and 4 define two
    separate registered journal populations.
    """

    return _build_assignments(corpus, REGISTERED_QUERY_DOMAIN_RULES)


def build_entrepreneurship_domain_assignments(
    corpus: pd.DataFrame,
) -> pd.DataFrame:
    """Compatibility wrapper for the two entrepreneurship domains."""

    return _build_assignments(corpus, ENTREPRENEURSHIP_DOMAIN_RULES)


def summarize_entrepreneurship_domain_journals(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paper counts by entrepreneurship domain and source title."""

    return (
        assignments.groupby(
            ["domain_id", "domain_label", "assignment_basis", "source_title"],
            as_index=False,
        )
        .agg(papers=("paper_id", "nunique"))
        .sort_values(
            ["domain_id", "papers", "source_title"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_registered_query_domain_sources(
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize paper counts by registered domain and Scopus source title."""

    return summarize_entrepreneurship_domain_journals(assignments)
