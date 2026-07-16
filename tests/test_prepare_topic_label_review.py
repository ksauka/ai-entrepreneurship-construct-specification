import pandas as pd
import pytest

from scripts.prepare_topic_label_review import build_review, validate_review


def _candidate_topics():
    return pd.DataFrame(
        {
            "min_topic_size": [50, 50],
            "topic_id": [0, 1],
            "paper_count": [3, 2],
            "topic_label": ["ai | consumer", "green | innovation"],
            "top_terms": ["ai; consumer; marketing", "green; innovation; energy"],
        }
    )


def _representatives(prefix="p"):
    rows = []
    for topic_id in (0, 1):
        for rank in (1, 2, 3):
            rows.append(
                {
                    "topic_id": topic_id,
                    "representative_rank": rank,
                    "paper_id": f"{prefix}{topic_id}{rank}",
                    "title": f"Paper {topic_id}-{rank}",
                    "similarity_to_topic_centroid": 0.9 - rank / 10,
                }
            )
    return pd.DataFrame(rows)


def _enriched():
    return pd.DataFrame(
        {
            "paper_id": ["a", "b", "c", "d", "e", "f"],
            "bertopic_topic": [0, 0, 0, 1, 1, ""],
            "query_1_topic_id": [1, 1, "", 0, 0, ""],
        }
    )


def _inputs():
    return {
        "full_corpus": (_candidate_topics(), _representatives("g"), "bertopic_topic"),
        "query_1": (_candidate_topics(), _representatives("q"), "query_1_topic_id"),
    }


def _expected():
    return {"full_corpus": 2, "query_1": 2}


def test_review_sheet_contains_scope_evidence_and_pending_decisions():
    review = build_review(_inputs(), _enriched())
    validate_review(review, expected_by_scope=_expected())

    assert len(review) == 4
    assert set(review.scope) == {"full_corpus", "query_1"}
    assert review.review_status.tolist() == ["pending"] * 4
    query_topic_0 = review[(review.scope == "query_1") & (review.topic_id == 0)].iloc[0]
    assert query_topic_0.final_assigned_papers == 2
    assert query_topic_0.representative_3_title == "Paper 0-3"
    assert query_topic_0.automatic_label == "ai / consumer"


def test_refresh_preserves_decisions_by_scope_and_topic_id():
    initial = build_review(_inputs(), _enriched())
    selected = (initial.scope == "query_1") & (initial.topic_id == 0)
    initial.loc[selected, "approved_label"] = "Query AI and consumers"
    initial.loc[selected, "review_status"] = "approved"
    initial.loc[selected, "reviewer_notes"] = "Checked"

    refreshed = build_review(_inputs(), _enriched(), previous=initial)
    validate_review(refreshed, expected_by_scope=_expected())
    topic = refreshed[(refreshed.scope == "query_1") & (refreshed.topic_id == 0)].iloc[0]
    global_topic = refreshed[(refreshed.scope == "full_corpus") & (refreshed.topic_id == 0)].iloc[0]
    assert topic.approved_label == "Query AI and consumers"
    assert topic.review_status == "approved"
    assert topic.reviewer_notes == "Checked"
    assert global_topic.approved_label == ""


def test_approved_status_requires_label():
    review = build_review(_inputs(), _enriched())
    review.loc[0, "review_status"] = "approved"
    with pytest.raises(ValueError, match="blank labels"):
        validate_review(review, expected_by_scope=_expected())


def test_scope_topic_key_must_be_unique():
    review = build_review(_inputs(), _enriched())
    review.loc[1, ["scope", "topic_id"]] = review.loc[0, ["scope", "topic_id"]]
    with pytest.raises(ValueError, match="duplicate scope-topic"):
        validate_review(review, expected_by_scope=_expected())
