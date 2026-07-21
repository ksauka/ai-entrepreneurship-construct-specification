import pandas as pd

from aecsp.corpus.domain_registry import build_registry_domain_assignments


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
