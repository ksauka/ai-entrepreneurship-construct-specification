"""Validate corpus source titles against configured query titles.

Inputs: corpus records and the saved Scopus query configuration.
Outputs: normalized title sets and a source-title validity flag per record.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

SOURCE_TITLE_VALID_COLUMN = "source_title_valid"
SOURCE_TITLE_MATCHED_COLUMN = "source_title_matched"

_EXACTSRCTITLE_PATTERN = re.compile(
    r'LIMIT-TO\s*\(\s*EXACTSRCTITLE\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)', re.IGNORECASE
)


def load_source_title_universe(config_path: Path) -> dict[str, set[str]]:
    """Return per-query and combined normalized source-title sets."""

    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    universe: dict[str, set[str]] = {}
    combined: set[str] = set()
    for query_label, query_config in config.get("queries", {}).items():
        scopus_query = query_config.get("scopus_query", "") or ""
        titles = {
            normalize_source_title(match)
            for match in _EXACTSRCTITLE_PATTERN.findall(scopus_query)
        }
        titles.discard("")
        universe[query_label] = titles
        combined |= titles

    universe["combined"] = combined
    return universe


def normalize_source_title(title: str) -> str:
    """Normalize a source title for matching (case, accents, punctuation)."""

    text = unicodedata.normalize("NFKD", str(title))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Scopus / fallback names for the source-title field, in preference order.
SOURCE_TITLE_COLUMNS = ("Source title", "source_title", "source", "Journal", "journal")


def validate_source_titles(
    frame: pd.DataFrame,
    universe: set[str],
    source_title_column: str | None = None,
) -> pd.DataFrame:
    """Flag each record's source title against the combined universe.

    Adds ``source_title_valid`` (0/1) and ``source_title_matched`` (the
    normalized title that matched, empty when invalid). Does not drop rows;
    filtering is the caller's decision so the audit trail stays intact.

    When ``source_title_column`` is None the first present column from
    ``SOURCE_TITLE_COLUMNS`` is used (Scopus ``Source title`` by default).
    """

    frame = frame.copy()
    column = source_title_column or next(
        (c for c in SOURCE_TITLE_COLUMNS if c in frame.columns), None
    )
    series = (
        frame[column]
        if column is not None
        else pd.Series([""] * len(frame), index=frame.index)
    )
    normalized = series.fillna("").astype(str).map(normalize_source_title)
    valid = normalized.isin(universe)
    frame[SOURCE_TITLE_VALID_COLUMN] = valid.astype(int)
    frame[SOURCE_TITLE_MATCHED_COLUMN] = normalized.where(valid, "")
    return frame
