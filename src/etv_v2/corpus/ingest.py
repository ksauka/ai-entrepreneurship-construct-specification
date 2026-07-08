"""Stage 0 / 0.5: load Query 1-4 Scopus exports and merge with query provenance.

This is the pandas implementation used for the real July 2026 corpus. It keeps
the same deduplication priority as :mod:`etv_v2.corpus.merge` (EID, then DOI,
then normalized title-year) but works on whole DataFrames so the 30k-record
Query 1 export stays fast, and all original Scopus columns survive the merge.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from etv_v2.corpus.query_provenance import (
    QUERY_BY_ID,
    QUERY_COUNT_COLUMN,
    QUERY_ONE_HOT_COLUMNS,
    QUERY_SOURCE_COLUMN,
    SEARCH_QUERIES,
)

DEDUP_KEY_COLUMN = "dedup_key"
PAPER_ID_COLUMN = "paper_id"

# Canonical lowercase columns expected by downstream ETV_V2 modules, mapped
# from Scopus export headers.
CANONICAL_COLUMNS: dict[str, str] = {
    "Title": "title",
    "Year": "year",
    "Source title": "source_title",
    "DOI": "doi",
    "EID": "eid",
    "Abstract": "abstract",
    "Authors": "authors",
    "Document Type": "document_type",
}


class IngestError(ValueError):
    """Raised when raw query exports cannot be loaded safely."""


def load_query_frame(paths: Sequence[Path]) -> pd.DataFrame:
    """Load one query's Scopus export (possibly split across CSV parts)."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise IngestError(f"Raw export not found: {path}")
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig"))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.map(lambda v: v.strip() if isinstance(v, str) else v)
    return frame


def merge_query_frames(query_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge per-query DataFrames into one deduplicated, provenance-aware corpus.

    Records are keyed by EID first, DOI second, normalized title-year third.
    A paper captured by several queries keeps one row with all matching
    one-hot columns set, plus ``query_count`` and ``query_sources``.
    """

    tagged: list[pd.DataFrame] = []
    for query_id in query_frames:
        if query_id not in QUERY_BY_ID:
            raise IngestError(f"Unknown query id: {query_id!r}")

    # Iterate in canonical Query 1-4 order so the surviving row for a
    # duplicate is deterministic.
    for query in SEARCH_QUERIES:
        frame = query_frames.get(query.id)
        if frame is None:
            continue
        frame = frame.copy()
        frame[DEDUP_KEY_COLUMN] = _dedup_keys(frame)
        missing = frame[DEDUP_KEY_COLUMN].isna()
        if missing.any():
            raise IngestError(
                f"{query.label}: {int(missing.sum())} records have no EID, DOI, "
                "or title-year fallback"
            )
        frame["_query_id"] = query.id
        # Within one query, the same paper occasionally appears twice across
        # export parts; keep the first occurrence.
        frame = frame.drop_duplicates(subset=DEDUP_KEY_COLUMN, keep="first")
        tagged.append(frame)

    if not tagged:
        raise IngestError("No query frames supplied")

    combined = pd.concat(tagged, ignore_index=True)

    membership = (
        combined.groupby(DEDUP_KEY_COLUMN)["_query_id"]
        .agg(lambda ids: sorted(set(ids), key=lambda q: QUERY_BY_ID[q].label))
        .rename("_query_ids")
    )

    merged = combined.drop_duplicates(subset=DEDUP_KEY_COLUMN, keep="first").copy()
    merged = merged.merge(membership, left_on=DEDUP_KEY_COLUMN, right_index=True)
    merged = merged.drop(columns=["_query_id"])

    for query in SEARCH_QUERIES:
        merged[query.one_hot_column] = merged["_query_ids"].map(
            lambda ids, qid=query.id: int(qid in ids)
        )
    merged[QUERY_SOURCE_COLUMN] = merged["_query_ids"].str.join(";")
    merged[QUERY_COUNT_COLUMN] = merged["_query_ids"].str.len()
    merged = merged.drop(columns=["_query_ids"])

    merged = _add_canonical_columns(merged)
    merged[PAPER_ID_COLUMN] = merged[DEDUP_KEY_COLUMN]

    return merged.reset_index(drop=True)


def query_view(master: pd.DataFrame, query_id: str) -> pd.DataFrame:
    """Return the overlapping view of the master corpus for one query."""

    query = QUERY_BY_ID.get(query_id)
    if query is None:
        raise IngestError(f"Unknown query id: {query_id!r}")
    return master[master[query.one_hot_column] == 1].reset_index(drop=True)


def _dedup_keys(frame: pd.DataFrame) -> pd.Series:
    eid = _column_or_empty(frame, ("EID", "eid"))
    doi = _column_or_empty(frame, ("DOI", "doi"))
    title = _column_or_empty(frame, ("Title", "title"))
    year = _column_or_empty(frame, ("Year", "year"))

    eid_norm = eid.str.lower().str.replace(r"\s+", "", regex=True)
    doi_norm = (
        doi.str.lower()
        .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    title_norm = (
        title.str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    year_norm = year.str.lower().str.replace(r"\s+", "", regex=True)

    keys = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    has_title_year = (title_norm != "") & (year_norm != "")
    keys[has_title_year] = "title_year:" + title_norm + ":" + year_norm
    keys[doi_norm != ""] = "doi:" + doi_norm
    keys[eid_norm != ""] = "eid:" + eid_norm
    return keys


def _column_or_empty(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].fillna("").astype(str)
    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _add_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for scopus_name, canonical in CANONICAL_COLUMNS.items():
        if canonical in frame.columns:
            continue
        if scopus_name in frame.columns:
            frame[canonical] = frame[scopus_name]
    return frame
