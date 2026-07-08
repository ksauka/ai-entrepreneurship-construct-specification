"""Contract tests for the canonical dataset scopes."""

import pandas as pd
import pytest

from etv_v2.corpus.scopes import (
    DATASET_SCOPES,
    ScopeError,
    iter_scopes,
    scope_frame,
)


@pytest.fixture
def master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"paper_id": "P1", "in_query_1": 1, "in_query_2": 0, "in_query_3": 1, "in_query_4": 0, "ai_ent_relevant": 1},
            {"paper_id": "P2", "in_query_1": 1, "in_query_2": 0, "in_query_3": 0, "in_query_4": 0, "ai_ent_relevant": 0},
            {"paper_id": "P3", "in_query_1": 0, "in_query_2": 1, "in_query_3": 0, "in_query_4": 1, "ai_ent_relevant": 1},
        ]
    )


def test_five_required_scopes_are_defined():
    assert [scope.id for scope in DATASET_SCOPES] == [
        "full_corpus",
        "query_1",
        "query_2",
        "query_3",
        "query_4",
    ]


def test_scope_frame_filters_by_one_hot_columns(master):
    assert len(scope_frame(master, "full_corpus")) == 3
    assert list(scope_frame(master, "query_1")["paper_id"]) == ["P1", "P2"]
    assert list(scope_frame(master, "query_3")["paper_id"]) == ["P1"]
    assert list(scope_frame(master, "strict_ai_ent")["paper_id"]) == ["P1", "P3"]


def test_scopes_overlap_rather_than_partition(master):
    views = iter_scopes(master)
    total_across_views = sum(len(frame) for scope, frame in views.items() if scope != "full_corpus")
    assert total_across_views > len(master) - 1  # P1 and P3 each appear in two query views


def test_unknown_scope_raises(master):
    with pytest.raises(ScopeError):
        scope_frame(master, "query_9")
