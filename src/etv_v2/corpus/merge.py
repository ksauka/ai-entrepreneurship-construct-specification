"""Merge and deduplicate Scopus records while preserving query provenance."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from etv_v2.corpus.query_provenance import (
    QUERY_BY_ID,
    QUERY_COUNT_COLUMN,
    QUERY_ONE_HOT_COLUMNS,
    QUERY_SOURCE_COLUMN,
    SearchQuery,
)

EID_COLUMNS = ("eid", "EID", "scopus_eid", "Scopus EID")
DOI_COLUMNS = ("doi", "DOI")
TITLE_COLUMNS = ("title", "Title", "document_title")
YEAR_COLUMNS = ("year", "Year", "publication_year", "Publication Year")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "eid": EID_COLUMNS,
    "doi": DOI_COLUMNS,
    "title": TITLE_COLUMNS,
    "year": YEAR_COLUMNS,
    "authors": ("authors", "Authors", "author_names"),
    "source_title": ("source_title", "Source title", "journal", "Journal", "source"),
    "abstract": ("abstract", "Abstract", "description"),
    "document_type": ("document_type", "Document Type", "doctype"),
}


class CorpusMergeError(ValueError):
    """Raised when query records cannot be merged safely."""


def merge_query_records(
    query_records: Mapping[str | SearchQuery, Iterable[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Merge Query 1-4 records into one deduplicated analytical corpus.

    Deduplication priority follows the project brief: Scopus EID first, DOI
    second, then normalized title-year. Query provenance is never collapsed away;
    duplicate papers keep all query one-hot columns and source ids.
    """

    merged: dict[str, dict[str, Any]] = {}

    for query_key, rows in query_records.items():
        query = _resolve_query(query_key)
        for row in rows:
            normalized = normalize_scopus_record(row)
            dedup_key = publication_dedup_key(normalized)
            if dedup_key is None:
                raise CorpusMergeError(
                    f"Record in {query.id} has no EID, DOI, or title-year fallback"
                )

            existing = merged.get(dedup_key)
            if existing is None:
                existing = _empty_query_flags({**row, **normalized})
                merged[dedup_key] = existing
            else:
                _merge_missing_values(existing, row)
                _merge_missing_values(existing, normalized)

            existing[query.one_hot_column] = 1
            _refresh_query_summary(existing)

    return list(merged.values())


def normalize_scopus_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical bibliographic fields without discarding original columns."""

    normalized: dict[str, Any] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        value = _first_value(row, aliases)
        if value is not None:
            normalized[canonical] = value
    return normalized


def publication_dedup_key(row: Mapping[str, Any]) -> str | None:
    """Create the stable deduplication key for one publication record."""

    eid = _first_value(row, EID_COLUMNS)
    if eid is not None:
        return f"eid:{_normalize_identifier(eid)}"

    doi = _first_value(row, DOI_COLUMNS)
    if doi is not None:
        return f"doi:{_normalize_doi(doi)}"

    title = _first_value(row, TITLE_COLUMNS)
    year = _first_value(row, YEAR_COLUMNS)
    if title is not None and year is not None:
        return f"title_year:{_normalize_title(title)}:{_normalize_identifier(year)}"

    return None


def _resolve_query(query: str | SearchQuery) -> SearchQuery:
    if isinstance(query, SearchQuery):
        return query

    normalized = str(query).strip().lower().replace(" ", "_")
    if normalized in QUERY_BY_ID:
        return QUERY_BY_ID[normalized]

    label = str(query).strip().lower()
    if label in {known.label.lower() for known in QUERY_BY_ID.values()}:
        for known in QUERY_BY_ID.values():
            if known.label.lower() == label:
                return known

    raise CorpusMergeError(f"Unknown query key: {query!r}")


def _empty_query_flags(row: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(row)
    for column in QUERY_ONE_HOT_COLUMNS:
        record[column] = 1 if _is_truthy(record.get(column)) else 0
    _refresh_query_summary(record)
    return record


def _refresh_query_summary(record: dict[str, Any]) -> None:
    sources = [
        query_id
        for query_id, query in QUERY_BY_ID.items()
        if _is_truthy(record.get(query.one_hot_column))
    ]
    record[QUERY_SOURCE_COLUMN] = ";".join(sources)
    record[QUERY_COUNT_COLUMN] = len(sources)


def _merge_missing_values(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if _clean(value) is None:
            continue
        if _clean(target.get(key)) is None:
            target[key] = value


def _first_value(row: Mapping[str, Any], columns: Iterable[str]) -> str | None:
    for column in columns:
        value = _clean(row.get(column))
        if value is not None:
            return value
    return None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip().lower())


def _normalize_doi(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    return _normalize_identifier(text)


def _normalize_title(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _clean(value)
    if text is None:
        return False
    return text.lower() in {"1", "true", "yes", "y", "x"}
