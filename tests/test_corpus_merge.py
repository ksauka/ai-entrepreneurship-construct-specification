import pytest

from etv_v2.corpus import (
    CorpusMergeError,
    SEARCH_QUERIES,
    merge_query_records,
    publication_dedup_key,
)


def test_july_2026_query_counts_are_recorded() -> None:
    counts = {query.id: query.july_2026_count for query in SEARCH_QUERIES}

    assert counts == {
        "query_1": 29294,
        "query_2": 818,
        "query_3": 1097,
        "query_4": 1509,
    }


def test_merge_query_records_preserves_cross_query_provenance() -> None:
    records = merge_query_records(
        {
            "query_1": [
                {
                    "EID": "2-s2.0-123",
                    "Title": "Artificial intelligence and venture creation",
                    "Year": 2026,
                    "Source title": "Journal of Business Venturing",
                }
            ],
            "query_3": [
                {
                    "eid": "2-s2.0-123",
                    "title": "Artificial intelligence and venture creation",
                    "year": 2026,
                    "doi": "10.1000/example",
                }
            ],
        }
    )

    assert len(records) == 1
    record = records[0]
    assert record["in_query_1"] == 1
    assert record["in_query_2"] == 0
    assert record["in_query_3"] == 1
    assert record["in_query_4"] == 0
    assert record["query_count"] == 2
    assert record["query_sources"] == "query_1;query_3"
    assert record["source_title"] == "Journal of Business Venturing"
    assert record["doi"] == "10.1000/example"


def test_publication_dedup_key_falls_back_to_title_year() -> None:
    key = publication_dedup_key(
        {
            "title": "AI, Entrepreneurship, and Opportunity Evaluation!",
            "year": "2026",
        }
    )

    assert key == "title_year:ai entrepreneurship and opportunity evaluation:2026"


def test_merge_query_records_rejects_records_without_identifier() -> None:
    with pytest.raises(CorpusMergeError):
        merge_query_records({"query_1": [{"abstract": "No title or stable id"}]})
