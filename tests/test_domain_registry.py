import pandas as pd

from aecsp.corpus.domain_registry import (
    build_asjc_domain_assignments,
    build_registry_domain_assignments,
)


def test_registry_assigns_only_papers_already_in_the_corpus():
    corpus = pd.DataFrame(
        [
            {"paper_id": "p1", "Source title": "Journal A"},
            {"paper_id": "p2", "Source title": "Journal and B"},
            {"paper_id": "p3", "Source title": "Outside Journal"},
        ]
    )
    domains = {
        "domain_a": {
            "label": "Domain A",
            "registry_field": "Field_A",
            "journals": ["Journal A", "Journal & B", "Absent Journal"],
        }
    }
    aliases = pd.DataFrame(
        [
            {
                "registered_title": "Journal & B",
                "corpus_title": "Journal and B",
                "review_status": "approved",
            }
        ]
    )

    assignments, sources = build_registry_domain_assignments(
        corpus, domains, aliases
    )

    assert assignments["paper_id"].tolist() == ["p1", "p2"]
    assert set(assignments["paper_id"]).issubset(set(corpus["paper_id"]))
    assert sources["papers"].sum() == 2
    assert sources.loc[sources["source_title"].eq("Journal and B"), "alias_applied"].iloc[0]


def test_asjc_aggregation_is_multilabel_and_traceable_to_codes():
    corpus = pd.DataFrame(
        [
            {"paper_id": "p1", "Source title": "Journal A"},
            {"paper_id": "p2", "Source title": "Journal B"},
            {"paper_id": "p3", "Source title": "Residual Journal"},
        ]
    )
    asjc = pd.DataFrame(
        [
            {"paper_id": "p1", "source_title": "Journal A", "asjc_code": "1405", "asjc_description": "Innovation"},
            {"paper_id": "p1", "source_title": "Journal A", "asjc_code": "1408", "asjc_description": "Strategy"},
            {"paper_id": "p2", "source_title": "Journal B", "asjc_code": "1408", "asjc_description": "Strategy"},
            {"paper_id": "p3", "source_title": "Residual Journal", "asjc_code": "1400", "asjc_description": "General business"},
        ]
    )
    domains = {
        "innovation": {
            "label": "Innovation",
            "mapping_mode": "official_asjc",
            "asjc_codes": {"1405": "Innovation"},
        },
        "strategy": {
            "label": "Strategy",
            "mapping_mode": "official_asjc",
            "asjc_codes": {"1408": "Strategy"},
        },
    }

    assignments, sources = build_asjc_domain_assignments(corpus, asjc, domains)

    assert len(assignments) == 3
    assert assignments.groupby("paper_id")["domain_id"].nunique().to_dict() == {
        "p1": 2,
        "p2": 1,
    }
    assert set(assignments["assignment_basis"]) == {
        "official_scopus_asjc:1405",
        "official_scopus_asjc:1408",
    }
    assert set(sources["mapping_mode"]) == {"official_asjc"}
    assert "p3" not in set(assignments["paper_id"])
