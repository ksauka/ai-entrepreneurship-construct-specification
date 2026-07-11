"""Split corpus scopes by VOSviewer citation connectivity.

Inputs: the master corpus, per-scope VOSviewer maps, and file timestamps.
Outputs: retained and dropped scope datasets with processing statistics.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from aecsp.corpus.scopes import scope_frame
from aecsp.progress import ProgressReporter

SCOPE_VOS_FILES: dict[str, str] = {
    "full_corpus": "master_corpus_vos.csv",
    "query_1": "query_1_vos.csv",
    "query_2": "query_2_vos.csv",
    "query_3": "query_3_vos.csv",
    "query_4": "query_4_vos.csv",
}


def normalize_doi(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    return re.sub(r"\s+", "", text).rstrip(".,;)")


def vos_status(vos_path: Path, reference_path: Path) -> str:
    """'current', 'missing', or 'stale' for one scope's VOS map."""

    if not vos_path.exists():
        return "missing"
    if reference_path.exists() and reference_path.stat().st_mtime > vos_path.stat().st_mtime:
        return "stale"
    return "current"


def load_vos_dois(path: Path) -> set[str]:
    """Normalized DOI set present in a VOS map (the citation-connected papers)."""

    df = pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)
    doi_col = _find_column(list(df.columns), "doi", "url", "link")
    if doi_col is None:
        return set()
    return {d for d in df[doi_col].map(normalize_doi) if d}


def split_scope(
    master: pd.DataFrame, scope_id: str, vos_dois: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (retained, dropped) for one scope, without adding VOS columns."""

    scope_papers = scope_frame(master, scope_id)
    doi_norm = scope_papers.get("DOI", pd.Series([""] * len(scope_papers))).map(normalize_doi)
    retained_mask = doi_norm.isin(vos_dois)
    retained = scope_papers[retained_mask].reset_index(drop=True)
    dropped = scope_papers[~retained_mask].reset_index(drop=True)
    return retained, dropped


def filter_all_scopes(
    master: pd.DataFrame,
    vos_dir: Path,
    reference_path: Path,
    output_dir: Path,
    *,
    show_progress: bool = False,
) -> dict:
    """Process every scope whose VOS map is current; write retained/dropped CSVs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    progress = (
        ProgressReporter("VOS scopes", len(SCOPE_VOS_FILES)) if show_progress else None
    )
    for scope_number, (scope_id, filename) in enumerate(
        SCOPE_VOS_FILES.items(), start=1
    ):
        vos_path = vos_dir / filename
        status = vos_status(vos_path, reference_path)
        if status != "current":
            stats[scope_id] = {"status": status}
            if progress is not None:
                progress.update(scope_number, detail=f"{scope_id}: {status}")
            continue

        vos_dois = load_vos_dois(vos_path)
        retained, dropped = split_scope(master, scope_id, vos_dois)
        retained.to_csv(output_dir / f"{scope_id}_retained.csv", index=False, encoding="utf-8-sig")
        dropped.to_csv(output_dir / f"{scope_id}_dropped.csv", index=False, encoding="utf-8-sig")

        total = len(retained) + len(dropped)
        stats[scope_id] = {
            "status": "filtered",
            "map_dois": len(vos_dois),
            "scope_papers": total,
            "retained": len(retained),
            "dropped": len(dropped),
            "retained_share": round(len(retained) / total, 4) if total else 0.0,
        }
        if progress is not None:
            progress.update(scope_number, detail=f"{scope_id}: filtered")
    return stats


def _find_column(columns: list[str], *needles: str) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for needle in needles:
        for lc, original in lowered.items():
            if needle in lc:
                return original
    return None
