"""Stage 1B: write VOSviewer-ready Scopus CSVs for the five dataset scopes.

VOSviewer reads Scopus CSV exports directly, so each scope export keeps the
original Scopus columns (VOSviewer needs Authors, Title, Year, Source title,
Cited by, DOI, Abstract, Author Keywords, Index Keywords, References, EID).
ETV_V2 provenance columns are dropped from the export because VOSviewer would
treat them as unknown metadata.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aecsp.corpus.query_provenance import SEARCH_QUERIES

# Scopus columns VOSviewer uses; the export keeps any of these that exist.
VOS_SCOPUS_COLUMNS = (
    "Authors",
    "Author full names",
    "Author(s) ID",
    "Title",
    "Year",
    "Source title",
    "Volume",
    "Issue",
    "Page start",
    "Page end",
    "Cited by",
    "DOI",
    "Link",
    "Affiliations",
    "Authors with affiliations",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "References",
    "Correspondence Address",
    "Publisher",
    "ISSN",
    "ISBN",
    "Language of Original Document",
    "Abbreviated Source Title",
    "Document Type",
    "Source",
    "EID",
)


def export_vos_scopes(master: pd.DataFrame, output_dir: Path) -> dict[str, dict]:
    """Write full-corpus and Query 1-4 VOSviewer CSVs; return per-scope stats."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scopes: dict[str, pd.DataFrame] = {"full_corpus": master}
    for query in SEARCH_QUERIES:
        scopes[query.id] = master[master[query.one_hot_column] == 1]

    stats: dict[str, dict] = {}
    for scope_name, frame in scopes.items():
        export = _to_scopus_export(frame)
        path = output_dir / f"vos_{scope_name}.csv"
        export.to_csv(path, index=False, encoding="utf-8-sig")
        stats[scope_name] = {"records": len(export), "path": str(path)}
    return stats


def _to_scopus_export(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in VOS_SCOPUS_COLUMNS if col in frame.columns]
    return frame[columns].copy()
