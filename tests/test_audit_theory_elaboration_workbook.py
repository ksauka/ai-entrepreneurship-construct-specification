"""Tests for the historical theory-elaboration workbook audit."""

from pathlib import Path

from scripts.audit_theory_elaboration_workbook import (
    extract_proposition_rows,
    extract_selection_rows,
    normalize_doi,
    normalize_eid,
    normalize_title,
    read_xlsx_rows,
)


def test_identifier_normalization_is_conservative() -> None:
    assert normalize_eid("eid:2-S2.0-123") == "2-s2.0-123"
    assert normalize_doi("https://doi.org/10.1000/Example. ") == "10.1000/example"
    assert normalize_title("AI–Enabled Enterprise") == "ai enabled enterprise"


def test_workbook_expected_paper_populations() -> None:
    root = Path(__file__).resolve().parents[1]
    sheets = read_xlsx_rows(
        root / "docs/ETP Theory Elaboration  Writing Workbook.xlsx"
    )

    selection = extract_selection_rows(sheets["Full Reading List Selection"])
    proposition = extract_proposition_rows(sheets["Proposition Paper map"])

    assert len(selection) == 367
    assert len(proposition) == 154
    assert len({normalize_title(row["workbook_title"]) for row in proposition}) == 153
