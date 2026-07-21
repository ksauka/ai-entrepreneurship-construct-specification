import pandas as pd

from aecsp.corpus.asjc import (
    build_source_crosswalk,
    normalize_issn,
    normalize_source_title,
    split_asjc_codes,
)


def official_sources() -> pd.DataFrame:
    rows = [
        {
            "Sourcerecord ID": "1",
            "Source Title": "Exact Journal",
            "ISSN": "12345678",
            "EISSN": "87654321",
            "Active or Inactive": "Active",
            "Coverage": "2000-2026",
            "Titles Discontinued by Scopus": "",
            "Source Type": "Journal",
            "Publisher": "Publisher",
            "Publisher Imprints Grouped to Main Publisher": "Group",
            "All Science Journal Classification Codes (ASJC)": "1408; 1403",
        },
        {
            "Sourcerecord ID": "2",
            "Source Title": "Title Repairs Identifier",
            "ISSN": "11111111",
            "EISSN": "",
            "Active or Inactive": "Active",
            "Coverage": "2000-2026",
            "Titles Discontinued by Scopus": "",
            "Source Type": "Journal",
            "Publisher": "Publisher",
            "Publisher Imprints Grouped to Main Publisher": "Group",
            "All Science Journal Classification Codes (ASJC)": "1404",
        },
        {
            "Sourcerecord ID": "3",
            "Source Title": "Official Renamed Journal",
            "ISSN": "22222222",
            "EISSN": "",
            "Active or Inactive": "Active",
            "Coverage": "2000-2026",
            "Titles Discontinued by Scopus": "",
            "Source Type": "Journal",
            "Publisher": "Publisher",
            "Publisher Imprints Grouped to Main Publisher": "Group",
            "All Science Journal Classification Codes (ASJC)": "1406",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["normalized_source_title"] = frame["Source Title"].map(
        normalize_source_title
    )
    frame["normalized_issn"] = frame["ISSN"].map(normalize_issn)
    frame["normalized_eissn"] = frame["EISSN"].map(normalize_issn)
    return frame


def test_normalization_and_code_splitting():
    assert normalize_issn("1234-567X") == "1234567X"
    assert normalize_source_title("The Journal: Of AI") == "the journal of ai"
    assert split_asjc_codes("1408; 1403; 1408") == ["1408", "1403"]


def test_crosswalk_uses_auditable_matching_priority():
    corpus = pd.DataFrame(
        [
            {"Source title": "Exact Journal", "ISSN": "12345678"},
            {
                "Source title": "Title Repairs Identifier",
                "ISSN": "99999999",
            },
            {"Source title": "Legacy Name", "ISSN": "22222222"},
        ]
    )
    overrides = pd.DataFrame(
        [
            {
                "source_title": "Title Repairs Identifier",
                "corpus_issn": "99999999",
                "scopus_source_record_id": "2",
                "asjc_codes_override": "",
                "review_date": "2026-07-20",
                "review_rationale": "Reviewed title match",
                "review_evidence": "Test fixture",
            },
            {
                "source_title": "Legacy Name",
                "corpus_issn": "22222222",
                "scopus_source_record_id": "3",
                "asjc_codes_override": "",
                "review_date": "2026-07-20",
                "review_rationale": "Reviewed ISSN match",
                "review_evidence": "Test fixture",
            },
        ]
    )
    crosswalk = build_source_crosswalk(
        corpus,
        official_sources(),
        {"1408": "Strategy", "1403": "Business", "1404": "MIS", "1406": "Marketing"},
        {"14": ("Business, Management and Accounting", "Social Sciences")},
        overrides,
    ).set_index("source_title")

    assert crosswalk.loc["Exact Journal", "match_method"] == "issn_and_title"
    assert (
        crosswalk.loc["Title Repairs Identifier", "automated_match_method"]
        == "title_only"
    )
    assert crosswalk.loc["Legacy Name", "automated_match_method"] == "issn_only"
    assert (
        crosswalk.loc["Title Repairs Identifier", "match_method"]
        == "reviewed_override"
    )
    assert crosswalk.loc["Legacy Name", "review_status"] == "completed"
    assert crosswalk.loc["Title Repairs Identifier", "asjc_codes"] == "1404"
    assert crosswalk.loc["Exact Journal", "asjc_code_count"] == 2


def test_non_exact_candidate_requires_review():
    corpus = pd.DataFrame(
        [{"Source title": "Title Repairs Identifier", "ISSN": "99999999"}]
    )
    crosswalk = build_source_crosswalk(
        corpus,
        official_sources(),
        {"1404": "MIS"},
        {"14": ("Business, Management and Accounting", "Social Sciences")},
    ).iloc[0]

    assert crosswalk["automated_match_method"] == "title_only"
    assert crosswalk["match_status"] == "review_required"
    assert crosswalk["review_status"] == "required"


def test_reviewed_override_can_correct_workbook_asjc():
    corpus = pd.DataFrame(
        [{"Source title": "Title Repairs Identifier", "ISSN": "99999999"}]
    )
    overrides = pd.DataFrame(
        [
            {
                "source_title": "Title Repairs Identifier",
                "corpus_issn": "99999999",
                "scopus_source_record_id": "2",
                "asjc_codes_override": "1408; 1710",
            }
        ]
    )
    crosswalk = build_source_crosswalk(
        corpus,
        official_sources(),
        {"1404": "MIS", "1408": "Strategy", "1710": "Information Systems"},
        {
            "14": ("Business, Management and Accounting", "Social Sciences"),
            "17": ("Computer Science", "Physical Sciences"),
        },
        overrides,
    ).iloc[0]

    assert crosswalk["official_asjc_codes"] == "1404"
    assert crosswalk["asjc_codes"] == "1408; 1710"
