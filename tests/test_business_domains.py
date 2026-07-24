import pandas as pd
import pytest

from aecsp.corpus.business_domains import (
    build_entrepreneurship_domain_assignments,
    build_registered_query_domain_assignments,
    summarize_entrepreneurship_domain_journals,
)


def test_builds_two_disjoint_entrepreneurship_domains():
    corpus = pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "Source title": "Core Journal",
                "in_query_3": "1",
                "in_query_4": "0",
            },
            {
                "paper_id": "p2",
                "Source title": "Other Journal",
                "in_query_3": "0",
                "in_query_4": "1",
            },
            {
                "paper_id": "p3",
                "Source title": "Outside Journal",
                "in_query_3": "0",
                "in_query_4": "0",
            },
        ]
    )
    result = build_entrepreneurship_domain_assignments(corpus)

    assert result[["paper_id", "domain_id"]].to_dict("records") == [
        {"paper_id": "p1", "domain_id": "core_entrepreneurship"},
        {"paper_id": "p2", "domain_id": "other_entrepreneurship"},
    ]
    assert set(result.loc[result["domain_id"].eq("core_entrepreneurship"), "domain_label"]) == {
        "Leading entrepreneurship journals"
    }
    summary = summarize_entrepreneurship_domain_journals(result)
    assert summary["papers"].tolist() == [1, 1]


def test_rejects_overlap_between_registered_entrepreneurship_domains():
    corpus = pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "Source title": "Journal",
                "in_query_3": 1,
                "in_query_4": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="overlap"):
        build_entrepreneurship_domain_assignments(corpus)


def test_ft50_is_a_domain_and_may_overlap_core_entrepreneurship():
    corpus = pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "Source title": "FT50 Core Journal",
                "in_query_2": 1,
                "in_query_3": 1,
                "in_query_4": 0,
            },
            {
                "paper_id": "p2",
                "Source title": "Other Journal",
                "in_query_2": 0,
                "in_query_3": 0,
                "in_query_4": 1,
            },
        ]
    )

    result = build_registered_query_domain_assignments(corpus)

    assert set(
        result.loc[result["paper_id"].eq("p1"), "domain_id"]
    ) == {"ft50", "core_entrepreneurship"}
    assert set(result["domain_id"]) == {
        "ft50",
        "core_entrepreneurship",
        "other_entrepreneurship",
    }
