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


def build_asjc_domain_assignments(
    corpus: pd.DataFrame,
    paper_asjc: pd.DataFrame,
    domains: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate official source-level ASJC codes into analytical domains.

    Only definitions with ``mapping_mode=official_asjc`` are processed here.
    A paper can inherit multiple domains and a domain can match several codes;
    output rows remain unique by paper and domain. The source-title summary
    records the exact codes that generated each domain membership.
    """

    corpus_required = {"paper_id", "Source title"}
    asjc_required = {
        "paper_id",
        "asjc_code",
        "asjc_description",
    }
    if not corpus_required.issubset(corpus.columns):
        raise ValueError(
            f"Corpus is missing ASJC-domain fields: {sorted(corpus_required - set(corpus.columns))}"
        )
    if not asjc_required.issubset(paper_asjc.columns):
        raise ValueError(
            "Paper ASJC table is missing required fields: "
            f"{sorted(asjc_required - set(paper_asjc.columns))}"
        )
    if corpus["paper_id"].astype(str).duplicated().any():
        raise ValueError("paper_id must be unique before ASJC-domain assignment")

    corpus_index = corpus[["paper_id", "Source title"]].copy()
    corpus_index["paper_id"] = corpus_index["paper_id"].astype(str)
    corpus_index = corpus_index.rename(columns={"Source title": "source_title"})
    corpus_ids = set(corpus_index["paper_id"])
    codes = paper_asjc.copy()
    codes["paper_id"] = codes["paper_id"].astype(str)
    codes["asjc_code"] = codes["asjc_code"].astype(str).str.strip()
    codes = codes[codes["paper_id"].isin(corpus_ids)]

    assignments: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    for domain_id, definition in domains.items():
        if str(definition.get("mapping_mode", "")) != "official_asjc":
            continue
        domain_codes = {
            str(code).strip(): str(description)
            for code, description in dict(definition.get("asjc_codes", {})).items()
        }
        if not domain_codes:
            raise ValueError(f"ASJC domain {domain_id} has no registered codes")
        selected_codes = codes[codes["asjc_code"].isin(domain_codes)].copy()
        if selected_codes.empty:
            raise ValueError(
                f"ASJC domain {domain_id} has no represented papers for codes "
                f"{sorted(domain_codes)}"
            )
        grouped = (
            selected_codes.groupby("paper_id", sort=False)["asjc_code"]
            .agg(lambda values: ";".join(sorted(set(values))))
            .rename("matched_asjc_codes")
            .reset_index()
        )
        selected = corpus_index.merge(grouped, on="paper_id", how="inner")
        selected["domain_id"] = str(domain_id)
        selected["domain_label"] = str(definition["label"])
        selected["assignment_basis"] = selected["matched_asjc_codes"].map(
            lambda value: f"official_scopus_asjc:{value}"
        )
        assignments.append(
            selected[
                [
                    "paper_id",
                    "source_title",
                    "domain_id",
                    "domain_label",
                    "assignment_basis",
                ]
            ]
        )

        source_codes = selected_codes[
            ["paper_id", "asjc_code", "asjc_description"]
        ].merge(
            corpus_index, on="paper_id", how="inner", validate="many_to_one"
        )
        for source_title, source_group in source_codes.groupby(
            "source_title", sort=True
        ):
            source_papers = source_group["paper_id"].nunique()
            matched_codes = sorted(set(source_group["asjc_code"]))
            descriptions = [domain_codes[code] for code in matched_codes]
            source_rows.append(
                {
                    "domain_id": str(domain_id),
                    "domain_label": str(definition["label"]),
                    "mapping_mode": "official_asjc",
                    "asjc_codes": ";".join(matched_codes),
                    "asjc_descriptions": "; ".join(descriptions),
                    "source_title": str(source_title),
                    "papers": int(source_papers),
                    "assignment_basis": (
                        "official_scopus_asjc:" + ";".join(matched_codes)
                    ),
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
    source_frame = pd.DataFrame(source_rows)
    if not source_frame.empty:
        source_frame = source_frame.sort_values(
            ["domain_id", "papers", "source_title"],
            ascending=[True, False, True],
            kind="stable",
        )
    return assignment_frame.reset_index(drop=True), source_frame.reset_index(drop=True)
