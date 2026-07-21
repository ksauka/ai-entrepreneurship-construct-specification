"""Apply reviewed journal-domain registries to papers already in the corpus."""

from __future__ import annotations

import pandas as pd


def normalize_source_title(value: object) -> str:
    """Match the historical registry rule: casefold and collapse whitespace."""

    return " ".join(str(value or "").split()).casefold().strip()


def build_registry_domain_assignments(
    corpus: pd.DataFrame,
    domain_journals: dict[str, dict[str, object]],
    aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return paper assignments and the represented-journal audit.

    The function never expands the corpus. A registered journal contributes to
    a domain only when that source title is represented by an existing paper.
    """

    required = {"paper_id", "Source title"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Corpus is missing domain fields: {sorted(missing)}")
    if corpus["paper_id"].duplicated().any():
        raise ValueError("paper_id must be unique before domain assignment")
    alias_required = {"registered_title", "corpus_title", "review_status"}
    if not alias_required.issubset(aliases.columns):
        raise ValueError("Journal alias table is missing required columns")
    approved = aliases[aliases["review_status"].astype(str).eq("approved")].copy()
    if approved["registered_title"].map(normalize_source_title).duplicated().any():
        raise ValueError("Approved aliases must be unique by registered title")
    alias_map = {
        normalize_source_title(row["registered_title"]): str(row["corpus_title"])
        for _, row in approved.iterrows()
    }

    corpus_sources = {
        normalize_source_title(value): str(value)
        for value in corpus["Source title"].astype(str).unique()
    }
    assignments = []
    source_rows = []
    for domain_id, definition in domain_journals.items():
        domain_label = str(definition["label"])
        registry_field = str(definition["registry_field"])
        seen_sources: set[str] = set()
        for registered_title in definition["journals"]:
            registered_title = str(registered_title)
            normalized = normalize_source_title(registered_title)
            corpus_title = corpus_sources.get(normalized)
            alias_applied = False
            if corpus_title is None and normalized in alias_map:
                candidate = alias_map[normalized]
                corpus_title = corpus_sources.get(normalize_source_title(candidate))
                alias_applied = corpus_title is not None
            if corpus_title is None or corpus_title in seen_sources:
                continue
            seen_sources.add(corpus_title)
            selected = corpus.loc[
                corpus["Source title"].astype(str).eq(corpus_title),
                ["paper_id", "Source title"],
            ].copy()
            selected = selected.rename(columns={"Source title": "source_title"})
            selected["domain_id"] = domain_id
            selected["domain_label"] = domain_label
            selected["assignment_basis"] = f"journal_registry:{registry_field}"
            assignments.append(selected)
            source_rows.append(
                {
                    "domain_id": domain_id,
                    "domain_label": domain_label,
                    "registry_field": registry_field,
                    "registered_source_title": registered_title,
                    "source_title": corpus_title,
                    "alias_applied": alias_applied,
                    "papers": len(selected),
                }
            )

    columns = [
        "paper_id",
        "source_title",
        "domain_id",
        "domain_label",
        "assignment_basis",
    ]
    if assignments:
        assignment_frame = pd.concat(assignments, ignore_index=True)[columns]
        assignment_frame = assignment_frame.drop_duplicates(
            ["paper_id", "domain_id"]
        ).sort_values(["domain_id", "source_title", "paper_id"], kind="stable")
    else:
        assignment_frame = pd.DataFrame(columns=columns)
    source_frame = pd.DataFrame(source_rows).sort_values(
        ["domain_id", "papers", "source_title"],
        ascending=[True, False, True],
        kind="stable",
    ) if source_rows else pd.DataFrame()
    return assignment_frame.reset_index(drop=True), source_frame.reset_index(drop=True)
