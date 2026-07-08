"""Query membership conventions for the four-query corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchQuery:
    """Canonical identity for one Scopus search query."""

    id: str
    label: str
    one_hot_column: str
    description: str
    july_2026_count: int


SEARCH_QUERIES: tuple[SearchQuery, ...] = (
    SearchQuery(
        id="query_1",
        label="Query 1",
        one_hot_column="in_query_1",
        description="Broad curated 695-source-title AI x entrepreneurship query.",
        july_2026_count=29294,
    ),
    SearchQuery(
        id="query_2",
        label="Query 2",
        one_hot_column="in_query_2",
        description="2026 FT50 query subset.",
        july_2026_count=818,
    ),
    SearchQuery(
        id="query_3",
        label="Query 3",
        one_hot_column="in_query_3",
        description="Leading entrepreneurship journals query based on Burnell et al. (2026).",
        july_2026_count=1097,
    ),
    SearchQuery(
        id="query_4",
        label="Query 4",
        one_hot_column="in_query_4",
        description="Other entrepreneurship journals query subset.",
        july_2026_count=1509,
    ),
)

QUERY_BY_ID: dict[str, SearchQuery] = {query.id: query for query in SEARCH_QUERIES}
QUERY_BY_LABEL: dict[str, SearchQuery] = {query.label.lower(): query for query in SEARCH_QUERIES}

QUERY_ONE_HOT_COLUMNS: tuple[str, ...] = tuple(
    query.one_hot_column for query in SEARCH_QUERIES
)

QUERY_SOURCE_COLUMN = "query_sources"
QUERY_COUNT_COLUMN = "query_count"
