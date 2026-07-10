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
        label="Broad business and management journals",
        one_hot_column="in_query_1",
        description="Broad curated list of 695 business, management, and economics source titles.",
        july_2026_count=29294,
    ),
    SearchQuery(
        id="query_2",
        label="FT50 journals",
        one_hot_column="in_query_2",
        description="The Financial Times 50 journal list, 2026 edition.",
        july_2026_count=818,
    ),
    SearchQuery(
        id="query_3",
        label="Leading entrepreneurship journals",
        one_hot_column="in_query_3",
        description="Leading entrepreneurship journals following Burnell et al. (2026).",
        july_2026_count=1097,
    ),
    SearchQuery(
        id="query_4",
        label="Additional entrepreneurship journals",
        one_hot_column="in_query_4",
        description="A wider set of entrepreneurship journals beyond the leading list.",
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
