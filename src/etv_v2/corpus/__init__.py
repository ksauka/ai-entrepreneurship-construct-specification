"""Corpus import, deduplication, validation, and query provenance."""

from etv_v2.corpus.merge import (
    CorpusMergeError,
    merge_query_records,
    normalize_scopus_record,
    publication_dedup_key,
)
from etv_v2.corpus.query_provenance import (
    QUERY_COUNT_COLUMN,
    QUERY_ONE_HOT_COLUMNS,
    QUERY_SOURCE_COLUMN,
    SEARCH_QUERIES,
    SearchQuery,
)

__all__ = [
    "CorpusMergeError",
    "QUERY_COUNT_COLUMN",
    "QUERY_ONE_HOT_COLUMNS",
    "QUERY_SOURCE_COLUMN",
    "SEARCH_QUERIES",
    "SearchQuery",
    "merge_query_records",
    "normalize_scopus_record",
    "publication_dedup_key",
]
