"""Scopus source-list parsing and auditable ASJC assignment.

The Scopus source list is distributed as an XLSX workbook.  This module reads
the required worksheets directly from the OOXML archive so the core pipeline
does not need an Excel-engine dependency.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterator
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

SOURCE_SHEET_PREFIX = "Scopus Sources "
CLASSIFICATION_SHEET_NAME = "ASJC Classification Codes"
ASJC_COLUMN = "All Science Journal Classification Codes (ASJC)"

SOURCE_COLUMNS = (
    "Sourcerecord ID",
    "Source Title",
    "ISSN",
    "EISSN",
    "Active or Inactive",
    "Coverage",
    "Titles Discontinued by Scopus",
    "Source Type",
    "Publisher",
    "Publisher Imprints Grouped to Main Publisher",
    ASJC_COLUMN,
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_issn(value: object) -> str:
    """Normalize an ISSN to eight uppercase alphanumeric characters."""

    return re.sub(r"[^0-9X]", "", str(value or "").upper())


def normalize_source_title(value: object) -> str:
    """Normalize a source title for conservative exact-title comparison."""

    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def split_asjc_codes(value: object) -> list[str]:
    """Return distinct ASJC codes in their source-list order."""

    seen: set[str] = set()
    codes: list[str] = []
    for item in str(value or "").split(";"):
        code = item.strip()
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"Invalid XLSX cell reference: {reference!r}")
    number = 0
    for character in letters.group():
        number = number * 26 + ord(character) - 64
    return number - 1


def _sheet_paths(workbook: ZipFile) -> dict[str, str]:
    workbook_root = ElementTree.parse(workbook.open("xl/workbook.xml")).getroot()
    relationship_root = ElementTree.parse(
        workbook.open("xl/_rels/workbook.xml.rels")
    ).getroot()
    targets = {
        relationship.attrib["Id"]: relationship.attrib["Target"]
        for relationship in relationship_root.findall(
            f"{{{PACKAGE_REL_NS}}}Relationship"
        )
    }
    paths: dict[str, str] = {}
    sheets = workbook_root.find(f"{{{SPREADSHEET_NS}}}sheets")
    if sheets is None:
        raise ValueError("The XLSX workbook has no worksheets")
    for sheet in sheets:
        relationship_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        target = targets[relationship_id]
        paths[sheet.attrib["name"]] = (
            target if target.startswith("xl/") else f"xl/{target}"
        )
    return paths


def _shared_strings(workbook: ZipFile) -> list[str]:
    strings: list[str] = []
    tag = f"{{{SPREADSHEET_NS}}}si"
    with workbook.open("xl/sharedStrings.xml") as stream:
        for _, element in ElementTree.iterparse(stream, events=("end",)):
            if element.tag == tag:
                strings.append("".join(element.itertext()))
                element.clear()
    return strings


def _iter_rows(
    workbook: ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> Iterator[dict[int, str]]:
    row_tag = f"{{{SPREADSHEET_NS}}}row"
    cell_tag = f"{{{SPREADSHEET_NS}}}c"
    value_tag = f"{{{SPREADSHEET_NS}}}v"
    inline_string_tag = f"{{{SPREADSHEET_NS}}}is"
    with workbook.open(sheet_path) as stream:
        for _, element in ElementTree.iterparse(stream, events=("end",)):
            if element.tag != row_tag:
                continue
            row: dict[int, str] = {}
            for cell in element.findall(cell_tag):
                reference = cell.attrib.get("r", "")
                if not reference:
                    continue
                value = cell.find(value_tag)
                if value is not None and value.text is not None:
                    item = value.text
                    if cell.attrib.get("t") == "s":
                        item = shared_strings[int(item)]
                    row[_column_index(reference)] = item
                    continue
                inline = cell.find(inline_string_tag)
                if inline is not None:
                    row[_column_index(reference)] = "".join(inline.itertext())
            if row:
                yield row
            element.clear()


def _select_sheet(
    sheet_paths: dict[str, str],
    *,
    exact_name: str | None = None,
    prefix: str | None = None,
) -> str:
    if exact_name and exact_name in sheet_paths:
        return sheet_paths[exact_name]
    if prefix:
        matches = [path for name, path in sheet_paths.items() if name.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
    requested = exact_name or prefix
    raise ValueError(f"Could not uniquely locate worksheet {requested!r}")


def read_scopus_source_list(path: Path) -> pd.DataFrame:
    """Read official source records and their ASJC codes from a Scopus workbook."""

    with ZipFile(path) as workbook:
        sheet_paths = _sheet_paths(workbook)
        shared_strings = _shared_strings(workbook)
        source_sheet = _select_sheet(sheet_paths, prefix=SOURCE_SHEET_PREFIX)
        rows = _iter_rows(workbook, source_sheet, shared_strings)
        header_row = next(rows)
        headers = {
            index: value.strip()
            for index, value in header_row.items()
            if value.strip() in SOURCE_COLUMNS
        }
        missing = set(SOURCE_COLUMNS) - set(headers.values())
        if missing:
            raise ValueError(f"Scopus source list is missing columns: {sorted(missing)}")
        records = [
            {header: row.get(index, "") for index, header in headers.items()}
            for row in rows
        ]
    frame = pd.DataFrame.from_records(records, columns=SOURCE_COLUMNS)
    frame["normalized_source_title"] = frame["Source Title"].map(
        normalize_source_title
    )
    frame["normalized_issn"] = frame["ISSN"].map(normalize_issn)
    frame["normalized_eissn"] = frame["EISSN"].map(normalize_issn)
    return frame


def read_asjc_classifications(
    path: Path,
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Read ASJC code labels and broad-group metadata from the source workbook."""

    with ZipFile(path) as workbook:
        sheet_paths = _sheet_paths(workbook)
        shared_strings = _shared_strings(workbook)
        classification_sheet = _select_sheet(
            sheet_paths, exact_name=CLASSIFICATION_SHEET_NAME
        )
        rows = list(_iter_rows(workbook, classification_sheet, shared_strings))

    labels: dict[str, str] = {}
    groups: dict[str, tuple[str, str]] = {}
    for row in rows:
        code = str(row.get(0, "")).strip()
        label = str(row.get(1, "")).strip()
        if re.fullmatch(r"\d{4}", code) and label:
            labels[code] = label
        group_code = str(row.get(3, "")).strip()
        group_label = str(row.get(4, "")).strip()
        supergroup = str(row.get(5, "")).strip()
        if re.fullmatch(r"\d{2}\*{2}", group_code) and group_label:
            groups[group_code[:2]] = (group_label, supergroup)
    groups.setdefault("10", ("Multidisciplinary", "Multidisciplinary"))
    return labels, groups


def _source_indexes(
    sources: pd.DataFrame,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_issn: defaultdict[str, list[int]] = defaultdict(list)
    by_title: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in sources.iterrows():
        for column in ("normalized_issn", "normalized_eissn"):
            identifier = row[column]
            if len(identifier) == 8 and index not in by_issn[identifier]:
                by_issn[identifier].append(index)
        by_title[row["normalized_source_title"]].append(index)
    return dict(by_issn), dict(by_title)


def _reviewed_override_index(
    reviewed_overrides: pd.DataFrame | None,
) -> dict[tuple[str, str], dict[str, object]]:
    """Index reviewed non-exact matches by normalized corpus title and ISSN."""

    if reviewed_overrides is None or reviewed_overrides.empty:
        return {}
    required = {"source_title", "corpus_issn", "scopus_source_record_id"}
    missing = required - set(reviewed_overrides.columns)
    if missing:
        raise ValueError(
            f"Reviewed ASJC overrides are missing columns: {sorted(missing)}"
        )

    index: dict[tuple[str, str], dict[str, object]] = {}
    for record in reviewed_overrides.to_dict("records"):
        key = (
            normalize_source_title(record["source_title"]),
            normalize_issn(record["corpus_issn"]),
        )
        if key in index:
            raise ValueError(
                "Duplicate reviewed ASJC override for "
                f"{record['source_title']!r}, {record['corpus_issn']!r}"
            )
        index[key] = record
    return index


def build_source_crosswalk(
    corpus_sources: pd.DataFrame,
    official_sources: pd.DataFrame,
    asjc_labels: dict[str, str],
    asjc_groups: dict[str, tuple[str, str]],
    reviewed_overrides: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Match distinct corpus source-title/ISSN pairs to official Scopus sources.

    Match priority is deliberately conservative:

    1. exact normalized title and ISSN;
    2. a reviewed override for a non-exact title or ISSN match;
    3. review required;
    4. unresolved.

    A unique title-only or ISSN-only candidate is retained as an automated
    suggestion but is not accepted without a reviewed override.
    """

    required = {"Source title", "ISSN"}
    missing = required - set(corpus_sources.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {sorted(missing)}")

    pairs = (
        corpus_sources.groupby(["Source title", "ISSN"], dropna=False)
        .size()
        .rename("paper_count")
        .reset_index()
    )
    by_issn, by_title = _source_indexes(official_sources)
    overrides = _reviewed_override_index(reviewed_overrides)
    source_record_rows: defaultdict[str, list[int]] = defaultdict(list)
    for index, source_record_id in official_sources["Sourcerecord ID"].items():
        source_record_rows[str(source_record_id).strip()].append(index)

    output: list[dict[str, object]] = []
    for pair in pairs.to_dict("records"):
        corpus_title = str(pair["Source title"])
        corpus_issn = normalize_issn(pair["ISSN"])
        paper_count = int(pair["paper_count"])
        normalized_title = normalize_source_title(corpus_title)
        issn_candidates = by_issn.get(corpus_issn, [])
        title_candidates = by_title.get(normalized_title, [])
        exact_candidates = [
            index
            for index in issn_candidates
            if official_sources.at[index, "normalized_source_title"]
            == normalized_title
        ]

        selected: int | None = None
        automated_match_method = "unresolved"
        if len(exact_candidates) == 1:
            selected = exact_candidates[0]
            automated_match_method = "issn_and_title"
        elif len(title_candidates) == 1:
            selected = title_candidates[0]
            automated_match_method = "title_only"
        elif len(issn_candidates) == 1:
            selected = issn_candidates[0]
            automated_match_method = "issn_only"

        override = overrides.get((normalized_title, corpus_issn))
        review_status = "not_required"
        match_method = automated_match_method
        if automated_match_method != "issn_and_title":
            review_status = "required"
            match_method = automated_match_method
            if override is not None:
                source_record_id = str(
                    override["scopus_source_record_id"]
                ).strip()
                override_candidates = source_record_rows.get(source_record_id, [])
                if len(override_candidates) != 1:
                    raise ValueError(
                        "Reviewed override source record ID must identify exactly "
                        f"one official source: {source_record_id!r}"
                    )
                selected = override_candidates[0]
                review_status = "completed"
                match_method = "reviewed_override"

        base: dict[str, object] = {
            "source_title": corpus_title,
            "normalized_source_title": normalized_title,
            "corpus_issn": corpus_issn,
            "paper_count": paper_count,
            "match_method": match_method,
            "automated_match_method": automated_match_method,
            "match_status": (
                "matched"
                if selected is not None and review_status != "required"
                else (
                    "review_required"
                    if selected is not None
                    else "unresolved"
                )
            ),
            "review_status": review_status,
            "review_date": str((override or {}).get("review_date", "")).strip(),
            "review_rationale": str(
                (override or {}).get("review_rationale", "")
            ).strip(),
            "review_evidence": str(
                (override or {}).get("review_evidence", "")
            ).strip(),
            "identifier_conflict": bool(
                automated_match_method == "title_only"
                and issn_candidates
                and selected not in issn_candidates
            ),
            "issn_candidate_count": len(issn_candidates),
            "title_candidate_count": len(title_candidates),
        }
        if selected is None:
            output.append(base)
            continue

        source = official_sources.loc[selected]
        official_codes = split_asjc_codes(source[ASJC_COLUMN])
        override_codes = split_asjc_codes(
            (override or {}).get("asjc_codes_override", "")
        )
        codes = override_codes or official_codes
        descriptions = [asjc_labels.get(code, "") for code in codes]
        subject_areas = [
            asjc_groups.get(code[:2], ("", ""))[0]
            for code in codes
        ]
        supergroups = [
            asjc_groups.get(code[:2], ("", ""))[1]
            for code in codes
        ]
        base.update(
            {
                "scopus_source_record_id": source["Sourcerecord ID"],
                "scopus_source_title": source["Source Title"],
                "official_issn": source["normalized_issn"],
                "official_eissn": source["normalized_eissn"],
                "source_status": source["Active or Inactive"],
                "source_coverage": source["Coverage"],
                "source_type": source["Source Type"],
                "publisher": source["Publisher"],
                "publisher_group": source[
                    "Publisher Imprints Grouped to Main Publisher"
                ],
                "official_asjc_codes": "; ".join(official_codes),
                "asjc_codes": "; ".join(codes),
                "asjc_descriptions": "; ".join(descriptions),
                "asjc_subject_areas": "; ".join(dict.fromkeys(subject_areas)),
                "asjc_supergroups": "; ".join(dict.fromkeys(supergroups)),
                "asjc_code_count": len(codes),
                "title_exact": source["normalized_source_title"]
                == normalized_title,
                "issn_exact": corpus_issn
                in {source["normalized_issn"], source["normalized_eissn"]},
            }
        )
        output.append(base)
    return pd.DataFrame.from_records(output)


def expand_asjc_assignments(
    paper_assignments: pd.DataFrame,
    asjc_labels: dict[str, str],
    asjc_groups: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Expand semicolon-delimited paper assignments to one row per ASJC code."""

    records: list[dict[str, object]] = []
    for row in paper_assignments.itertuples(index=False):
        for code in split_asjc_codes(row.asjc_codes):
            subject_area, supergroup = asjc_groups.get(code[:2], ("", ""))
            records.append(
                {
                    "paper_id": row.paper_id,
                    "source_title": row.source_title,
                    "corpus_issn": row.corpus_issn,
                    "scopus_source_record_id": row.scopus_source_record_id,
                    "asjc_code": code,
                    "asjc_description": asjc_labels.get(code, ""),
                    "asjc_subject_area": subject_area,
                    "asjc_supergroup": supergroup,
                    "match_method": row.match_method,
                }
            )
    return pd.DataFrame.from_records(records)
