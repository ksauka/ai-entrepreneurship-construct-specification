"""Stage 1: validate source titles against the Query 1-4 source-title universe.

The universe is derived from the saved Scopus queries themselves: every
``LIMIT-TO ( EXACTSRCTITLE , "..." )`` clause in
``configs/search_queries_july2026_q1_q4.yaml``. A record whose source title is
outside that universe was captured by a query facet drift (Scopus sometimes
returns variant source titles) and gets flagged rather than silently kept.
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


def validate_source_titles(
    frame: pd.DataFrame,
    universe: set[str],
    source_title_column: str = "source_title",
) -> pd.DataFrame:
    """Flag each record's source title against the combined universe.

    Adds ``source_title_valid`` (0/1) and ``source_title_matched`` (the
    normalized title that matched, empty when invalid). Does not drop rows;
    filtering is the caller's decision so the audit trail stays intact.
    """

    frame = frame.copy()
    normalized = (
        frame.get(source_title_column, pd.Series([""] * len(frame), index=frame.index))
        .fillna("")
        .astype(str)
        .map(normalize_source_title)
    )
    valid = normalized.isin(universe)
    frame[SOURCE_TITLE_VALID_COLUMN] = valid.astype(int)
    frame[SOURCE_TITLE_MATCHED_COLUMN] = normalized.where(valid, "")
    return frame
