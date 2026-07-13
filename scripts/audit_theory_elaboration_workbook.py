"""Match previously read workbook papers to the current corpus and samples.

Inputs:
    docs/ETP Theory Elaboration  Writing Workbook.xlsx
    data/processed/master_corpus.csv
    data/interim/proprietary_validation/proprietary_probability_sample_2235.csv
    data/interim/proprietary_validation/proprietary_rater_target_2276_papers.csv

Outputs:
    A row-level match audit, corpus-matched theory-elaboration sample,
    probability-sample overlap, unmatched-paper table, JSON manifest, and
    Markdown audit report. The frozen probability sample is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_title(value: object) -> str:
    """Normalize a title for conservative exact matching."""

    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    with_boundaries = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in without_marks
    )
    ascii_value = with_boundaries.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value).strip()


def normalize_doi(value: object) -> str:
    """Normalize a DOI without changing its substantive identifier."""

    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.rstrip(" .")


def normalize_eid(value: object) -> str:
    """Normalize an EID or paper ID to its Scopus EID form."""

    eid = str(value or "").strip().lower()
    return eid.removeprefix("eid:")


def column_number(reference: str) -> int:
    """Convert an XLSX cell reference to a one-based column number."""

    letters = re.match(r"[A-Z]+", reference)
    if not letters:
        raise ValueError(f"Invalid XLSX cell reference: {reference}")
    number = 0
    for letter in letters.group():
        number = number * 26 + ord(letter) - 64
    return number


def read_xlsx_rows(path: Path) -> dict[str, list[tuple[int, dict[int, str]]]]:
    """Read worksheet values using only the Python standard library."""

    sheets: dict[str, list[tuple[int, dict[int, str]]]] = {}
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                for item in root.findall(f"{{{MAIN_NS}}}si")
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        for sheet in workbook.find(f"{{{MAIN_NS}}}sheets") or []:
            name = sheet.attrib["name"].strip()
            relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
            target = targets[relationship_id].lstrip("/")
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            root = ET.fromstring(archive.read(target))
            rows: list[tuple[int, dict[int, str]]] = []
            for row in root.findall(
                f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"
            ):
                values: dict[int, str] = {}
                for cell in row.findall(f"{{{MAIN_NS}}}c"):
                    cell_type = cell.attrib.get("t")
                    raw_value = cell.find(f"{{{MAIN_NS}}}v")
                    value = ""
                    if cell_type == "inlineStr":
                        value = "".join(
                            node.text or ""
                            for node in cell.iter(f"{{{MAIN_NS}}}t")
                        )
                    elif raw_value is not None:
                        value = raw_value.text or ""
                        if cell_type == "s":
                            value = shared_strings[int(value)]
                    values[column_number(cell.attrib.get("r", "A1"))] = value
                if any(str(value).strip() for value in values.values()):
                    rows.append((int(row.attrib["r"]), values))
            sheets[name] = rows
    return sheets


def extract_selection_rows(
    rows: list[tuple[int, dict[int, str]]],
) -> list[dict[str, str | int]]:
    """Extract the old 367-paper screening dataset embedded in the workbook."""

    records = []
    for row_number, values in rows:
        if row_number <= 1 or not str(values.get(3, "")).strip():
            continue
        records.append(
            {
                "selection_row": row_number,
                "title": values.get(3, ""),
                "year": values.get(4, ""),
                "doi": values.get(5, ""),
                "reading_status": values.get(6, ""),
                "abstract": values.get(7, ""),
                "eid": values.get(8, ""),
            }
        )
    return records


def extract_proposition_rows(
    rows: list[tuple[int, dict[int, str]]],
) -> list[dict[str, str | int]]:
    """Extract paper rows from the 154-row proposition map."""

    records = []
    for row_number, values in rows:
        year = str(values.get(3, "")).strip()
        title = str(values.get(5, "")).strip()
        if row_number <= 3 or not re.fullmatch(r"\d{4}", year) or not title:
            continue
        records.append(
            {
                "workbook_row": row_number,
                "workbook_topic": values.get(1, ""),
                "workbook_year": year,
                "workbook_citation": values.get(4, ""),
                "workbook_title": title,
                "workbook_authors": values.get(6, ""),
                "workbook_journal": values.get(7, ""),
                "workbook_type": values.get(9, ""),
                "workbook_p1": values.get(10, ""),
                "workbook_p2": values.get(11, ""),
                "workbook_p2_contrast": values.get(12, ""),
                "workbook_p3": values.get(13, ""),
                "workbook_contribution_note": values.get(14, ""),
            }
        )
    return records


def unique_index(values: pd.Series, normalizer) -> dict[str, list[int]]:
    """Index normalized values while retaining duplicate positions."""

    index: dict[str, list[int]] = defaultdict(list)
    for position, value in enumerate(values):
        normalized = normalizer(value)
        if normalized:
            index[normalized].append(position)
    return dict(index)


def join_unique(values: pd.Series) -> str:
    """Join non-empty unique values in their first-seen order."""

    return "; ".join(dict.fromkeys(value for value in values.astype(str) if value))


def build_report(
    summary: dict, unmatched: pd.DataFrame, probability_overlap: pd.DataFrame
) -> str:
    """Create the human-readable workbook audit report."""

    unmatched_rows = []
    for row in unmatched.itertuples(index=False):
        unmatched_rows.append(
            f"| {row.workbook_year} | {row.workbook_citation} | "
            f"{str(row.workbook_title).replace('|', '/')} | "
            f"{str(row.workbook_old_doi).replace('|', '/')} |"
        )
    unmatched_table = "\n".join(unmatched_rows) or "| — | — | None | — |"
    overlap_rows = []
    for row in probability_overlap.itertuples(index=False):
        overlap_rows.append(
            f"| {row.workbook_probability_overlap_order} | "
            f"{str(row.workbook_citations).replace('|', '/')} | {row.Year} | "
            f"{str(row.Title).replace('|', '/')} | `{row.paper_id}` |"
        )
    overlap_table = "\n".join(overlap_rows)
    return f"""# Theory-elaboration workbook paper audit

## Conclusion

The workbook proposition map contains **{summary['workbook_map_rows']:,} rows**
representing **{summary['workbook_unique_papers']:,} unique previously read
papers**. One paper is deliberately mapped to two topic locations. Exact
matching found **{summary['matched_unique_papers']:,} unique papers** in the
current 22,345-paper corpus; **{summary['unmatched_unique_papers']:,} unique
workbook papers are not present**.

Of the matched historical papers, **{summary['probability_overlap_n']:,}** are
already members of the frozen 2,235-paper probability sample. Those 23 papers
are the workbook-linked model-validation subset. The remaining
**{summary['outside_probability_n']:,}** matched papers are retained as the
previously read qualitative evidence base but are **not added** to the
probability sample or proprietary-rater target.

## Methodological treatment

The frozen probability sample remains unchanged. Its sampling weights and IRR
estimates continue to apply only to the original 2,235 probability-selected
papers. The workbook papers are a **purposive, previously read
theory-elaboration supplement**. They preserve the evidentiary continuity of
Chapters 4 and 5 and may be used for qualitative checking, proposition tracing,
and sensitivity analysis. They must not be pooled into weighted probability-
sample estimates or described as randomly selected.

Claude/Gemini comparisons linked to the workbook use only the 23 papers that
entered the probability sample through the original random draw. The 136-paper
matched set remains available for Chapters 4 and 5 as a purposive targeted-read
evidence set, not as an IRR sample.

## Matching rules

1. Workbook titles were normalized only for Unicode, punctuation, case, and
   whitespace.
2. Embedded old-dataset EIDs and DOIs were recovered from the workbook's
   367-paper selection sheet.
3. Matches were accepted only when an exact normalized title or an exact
   Scopus EID/DOI resolved uniquely in the current master corpus.
4. No fuzzy match was accepted. Identifier disagreements are retained as audit
   warnings rather than silently resolved.
5. All {summary['matched_unique_papers']:,} matched records passed an exact
   field-level cross-check against the original Scopus query exports for title,
   abstract, author keywords, source title, and year; no discrepancies were
   found.
6. One workbook anomaly is retained in the row-level audit: the Ahmed et al.
   title appears beside two EID/DOI pairs in the old selection sheet. The exact
   current title and its matching EID/DOI identify the accepted record; the
   conflicting old identifier remains flagged.

## Workbook papers already in the probability sample

These 23 papers require no sample expansion and no new proprietary-model run.

| # | Workbook citation | Year | Title | Current paper ID |
|---:|---|---:|---|---|
{overlap_table}

## Papers absent from the current corpus

| Year | Workbook citation | Title | Old-workbook DOI |
|---:|---|---|---|
{unmatched_table}

## Reproducible outputs

- Row-level audit: `data/interim/theory_elaboration/workbook_paper_match_audit.csv`
- All 136 current-corpus matches: `data/interim/theory_elaboration/theory_elaboration_matched_papers.csv`
- The 23 probability-sample overlaps: `data/interim/theory_elaboration/theory_elaboration_probability_overlap_23.csv`
- Unmatched papers: `data/interim/theory_elaboration/workbook_unmatched_papers.csv`
- Scopus provenance audit: `data/interim/theory_elaboration/provenance_audit/provenance_audit_summary.json`
- Machine-readable manifest: `data/interim/theory_elaboration/workbook_audit_manifest.json`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        type=Path,
        default=PROJECT_ROOT / "docs/ETP Theory Elaboration  Writing Workbook.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/theory_elaboration",
    )
    args = parser.parse_args()
    workbook_path = args.workbook.resolve()
    output_dir = args.output_dir.resolve()
    master_path = PROJECT_ROOT / "data/processed/master_corpus.csv"
    probability_path = (
        PROJECT_ROOT
        / "data/interim/proprietary_validation/proprietary_probability_sample_2235.csv"
    )
    provider_target_path = (
        PROJECT_ROOT
        / "data/interim/proprietary_validation/proprietary_rater_target_2276_papers.csv"
    )

    sheets = read_xlsx_rows(workbook_path)
    selection = extract_selection_rows(sheets["Full Reading List Selection"])
    proposition = extract_proposition_rows(sheets["Proposition Paper map"])
    if len(selection) != 367:
        raise RuntimeError(f"Expected 367 selection rows, found {len(selection)}")
    if len(proposition) != 154:
        raise RuntimeError(f"Expected 154 proposition rows, found {len(proposition)}")

    master = pd.read_csv(
        master_path, dtype=str, keep_default_na=False, low_memory=False
    )
    probability = pd.read_csv(
        probability_path, dtype=str, keep_default_na=False, low_memory=False
    )
    provider_target = pd.read_csv(
        provider_target_path, dtype=str, keep_default_na=False, low_memory=False
    )
    if master["paper_id"].duplicated().any():
        raise RuntimeError("master corpus contains duplicate paper_id values")

    master_title = unique_index(master["Title"], normalize_title)
    master_doi = unique_index(master["DOI"], normalize_doi)
    master_eid = unique_index(master["EID"], normalize_eid)
    selection_title: dict[str, list[dict]] = defaultdict(list)
    for row in selection:
        selection_title[normalize_title(row["title"])].append(row)

    probability_ids = set(probability["paper_id"])
    provider_ids = set(provider_target["paper_id"])
    audit_rows = []
    for record in proposition:
        title_key = normalize_title(record["workbook_title"])
        old_records = selection_title.get(title_key, [])
        old_eids = sorted(
            {normalize_eid(row["eid"]) for row in old_records if normalize_eid(row["eid"])}
        )
        old_dois = sorted(
            {normalize_doi(row["doi"]) for row in old_records if normalize_doi(row["doi"])}
        )
        old_abstracts = list(
            dict.fromkeys(
                str(row["abstract"]).strip()
                for row in old_records
                if str(row["abstract"]).strip()
            )
        )
        old_statuses = list(
            dict.fromkeys(
                str(row["reading_status"]).strip()
                for row in old_records
                if str(row["reading_status"]).strip()
            )
        )
        hits: dict[str, set[int]] = {
            "title": set(master_title.get(title_key, [])),
            "eid": {position for eid in old_eids for position in master_eid.get(eid, [])},
            "doi": {position for doi in old_dois for position in master_doi.get(doi, [])},
        }
        all_hits = set().union(*hits.values())
        title_hits = hits["title"]
        warning = ""
        position: int | None = None
        if len(title_hits) == 1:
            position = next(iter(title_hits))
            conflicting = all_hits - {position}
            if conflicting:
                warning = "workbook_identifier_conflict"
        elif len(all_hits) == 1:
            position = next(iter(all_hits))
        elif len(all_hits) > 1:
            warning = "ambiguous_identifiers"

        accepted_methods = [name for name, values in hits.items() if position in values]
        row = {
            **record,
            "workbook_selection_matches": len(old_records),
            "workbook_old_eid": ";".join(old_eids),
            "workbook_old_doi": ";".join(old_dois),
            "workbook_old_abstract": " || ".join(old_abstracts),
            "workbook_old_reading_status": ";".join(old_statuses),
            "match_status": "matched" if position is not None else "not_in_current_corpus",
            "match_method": "+".join(accepted_methods),
            "match_warning": warning,
            "paper_id": "",
            "current_eid": "",
            "current_doi": "",
            "current_title": "",
            "current_year": "",
            "current_source_title": "",
            "in_probability_sample": 0,
            "in_existing_provider_target": 0,
        }
        if position is not None:
            current = master.iloc[position]
            row.update(
                {
                    "paper_id": current["paper_id"],
                    "current_eid": current["EID"],
                    "current_doi": current["DOI"],
                    "current_title": current["Title"],
                    "current_year": current["Year"],
                    "current_source_title": current["Source title"],
                    "in_probability_sample": int(current["paper_id"] in probability_ids),
                    "in_existing_provider_target": int(current["paper_id"] in provider_ids),
                }
            )
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows)
    matched_audit = audit[audit["match_status"] == "matched"].copy()
    unmatched = audit[audit["match_status"] != "matched"].copy()
    workbook_unique = audit["workbook_title"].map(normalize_title).nunique()
    unmatched_unique = unmatched["workbook_title"].map(normalize_title).nunique()

    grouped = (
        matched_audit.groupby("paper_id", sort=False)
        .agg(
            workbook_topics=("workbook_topic", join_unique),
            workbook_citations=("workbook_citation", join_unique),
            workbook_types=("workbook_type", join_unique),
            workbook_p1=("workbook_p1", join_unique),
            workbook_p2=("workbook_p2", join_unique),
            workbook_p2_contrast=("workbook_p2_contrast", join_unique),
            workbook_p3=("workbook_p3", join_unique),
            workbook_contribution_notes=("workbook_contribution_note", join_unique),
            workbook_map_rows=("workbook_row", lambda values: ";".join(values.astype(str))),
            workbook_map_row_count=("workbook_row", "size"),
            match_warning=("match_warning", join_unique),
        )
        .reset_index()
    )
    matched = grouped.merge(master, on="paper_id", how="left", validate="one_to_one")
    matched.insert(1, "in_probability_sample", matched["paper_id"].isin(probability_ids).astype(int))
    matched.insert(2, "in_existing_provider_target", matched["paper_id"].isin(provider_ids).astype(int))
    matched.insert(3, "historical_sample_role", "purposive_theory_elaboration")

    probability_overlap = matched[matched["paper_id"].isin(probability_ids)].copy()
    probability_overlap.insert(
        1, "workbook_probability_overlap_order", range(1, len(probability_overlap) + 1)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "workbook_paper_match_audit.csv"
    matched_path = output_dir / "theory_elaboration_matched_papers.csv"
    overlap_path = output_dir / "theory_elaboration_probability_overlap_23.csv"
    unmatched_path = output_dir / "workbook_unmatched_papers.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    matched.to_csv(matched_path, index=False, encoding="utf-8-sig")
    probability_overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": pd.Timestamp.now(tz="Europe/Amsterdam").isoformat(),
        "workbook": str(workbook_path.relative_to(PROJECT_ROOT)),
        "workbook_sha256": sha256(workbook_path),
        "master_corpus": str(master_path.relative_to(PROJECT_ROOT)),
        "master_corpus_sha256": sha256(master_path),
        "population_n": len(master),
        "workbook_selection_rows": len(selection),
        "workbook_map_rows": len(audit),
        "workbook_unique_papers": int(workbook_unique),
        "matched_map_rows": len(matched_audit),
        "matched_unique_papers": len(matched),
        "unmatched_map_rows": len(unmatched),
        "unmatched_unique_papers": int(unmatched_unique),
        "probability_overlap_n": int(matched["in_probability_sample"].sum()),
        "existing_provider_target_overlap_n": int(matched["in_existing_provider_target"].sum()),
        "outside_probability_n": int((matched["in_probability_sample"] == 0).sum()),
        "provider_target_expansion_adopted": False,
        "probability_sample_changed": False,
        "historical_sample_design": "purposive_previously_read_theory_elaboration",
        "fuzzy_matches_accepted": 0,
        "match_warnings_n": int(audit["match_warning"].ne("").sum()),
        "outputs": {},
    }
    for path in (audit_path, matched_path, overlap_path, unmatched_path):
        summary["outputs"][str(path.relative_to(PROJECT_ROOT))] = sha256(path)
    provenance_summary_path = output_dir / "provenance_audit/provenance_audit_summary.json"
    if provenance_summary_path.exists():
        provenance = json.loads(provenance_summary_path.read_text(encoding="utf-8"))
        summary["raw_scopus_provenance_audit_passed"] = bool(provenance["audit_passed"])
        summary["raw_scopus_substantive_discrepancies_n"] = int(
            provenance["substantive_discrepancies_n"]
        )
    manifest_path = output_dir / "workbook_audit_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = PROJECT_ROOT / "reports/analysis/THEORY_ELABORATION_WORKBOOK_AUDIT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_report(summary, unmatched, probability_overlap), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
