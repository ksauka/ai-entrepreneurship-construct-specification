"""Tests for explicit approval of BERTopic grid-search selections."""

import json

import pytest

from scripts.run_topics import load_approved_topic_selections


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _recommendations():
    scopes = {}
    for scope in ("full_corpus", "query_1", "query_2", "query_3", "query_4"):
        scopes[scope] = {
            "candidate_min_topic_sizes": [10, 20],
            "recommended_min_topic_size": 20,
        }
    return {"scopes": scopes}


def _approval():
    scopes = {}
    for scope in ("full_corpus", "query_1", "query_2", "query_3", "query_4"):
        scopes[scope] = {
            "selected_min_topic_size": 10,
            "selected_topic_count": 7,
            "decision": "human_approved",
        }
    return {"approval_status": "approved", "scopes": scopes}


def test_load_approved_topic_selections_applies_reviewed_override(tmp_path):
    recommendations_path = tmp_path / "recommendations.json"
    approval_path = tmp_path / "approved.json"
    _write_json(recommendations_path, _recommendations())
    _write_json(approval_path, _approval())

    selected, approval = load_approved_topic_selections(
        recommendations_path, approval_path
    )

    assert approval["approval_status"] == "approved"
    assert selected["full_corpus"]["automatic_recommended_min_topic_size"] == 20
    assert selected["full_corpus"]["recommended_min_topic_size"] == 10
    assert selected["query_4"]["approved_topic_count"] == 7


def test_load_approved_topic_selections_rejects_untested_choice(tmp_path):
    recommendations_path = tmp_path / "recommendations.json"
    approval_path = tmp_path / "approved.json"
    approval = _approval()
    approval["scopes"]["query_3"]["selected_min_topic_size"] = 99
    _write_json(recommendations_path, _recommendations())
    _write_json(approval_path, approval)

    with pytest.raises(ValueError, match="was not in the tested grid"):
        load_approved_topic_selections(recommendations_path, approval_path)


def test_load_approved_topic_selections_blocks_pending_review(tmp_path):
    recommendations_path = tmp_path / "recommendations.json"
    approval_path = tmp_path / "approved.json"
    approval = _approval()
    approval["approval_status"] = "pending_researcher_confirmation"
    _write_json(recommendations_path, _recommendations())
    _write_json(approval_path, approval)

    with pytest.raises(ValueError, match="not marked approved"):
        load_approved_topic_selections(recommendations_path, approval_path)
