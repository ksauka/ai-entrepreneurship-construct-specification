"""Contract tests for Layer 5 convergence/divergence/contrast analytics."""

import pandas as pd
import pytest

from aecsp.analytics.convergence import (
    construct_contrast,
    convergence_by,
    group_convergence,
)


@pytest.fixture
def coded() -> pd.DataFrame:
    # Journal A: everyone agrees on role (converges); Journal B: split (diverges).
    return pd.DataFrame(
        [
            {"paper_id": "P1", "journal": "A", "ai_role_function": "AI as tool", "ai_type_form": "predictive AI"},
            {"paper_id": "P2", "journal": "A", "ai_role_function": "AI as tool", "ai_type_form": "generative AI"},
            {"paper_id": "P3", "journal": "B", "ai_role_function": "AI as tool", "ai_type_form": "predictive AI"},
            {"paper_id": "P4", "journal": "B", "ai_role_function": "AI as actor/agent", "ai_type_form": "predictive AI"},
        ]
    )


def test_group_convergence_scores_are_bounded(coded):
    conv = group_convergence(coded, group_label="all")
    assert conv.paper_count == 4
    assert 0.0 <= conv.overall_specification_clarity_score <= 1.0
    assert round(conv.fragmentation_score + conv.overall_specification_clarity_score, 4) == 1.0


def test_unanimous_dimension_converges_fully():
    frame = pd.DataFrame(
        [{"paper_id": f"P{i}", "ai_role_function": "AI as tool"} for i in range(5)]
    )
    conv = group_convergence(frame)
    assert conv.dimension_scores["ai_role_function"] == 1.0
    assert conv.dominant_values["ai_role_function"] == "AI as tool"


def test_convergence_by_journal_ranks_fragmentation(coded):
    result = convergence_by(coded, "journal")
    assert set(result["journal"]) == {"A", "B"}
    a = result[result["journal"] == "A"].iloc[0]
    b = result[result["journal"] == "B"].iloc[0]
    # Journal A agrees on role (score 1.0); Journal B is split (score 0.0).
    assert a["ai_role_function_convergence_score"] == 1.0
    assert b["ai_role_function_convergence_score"] == 0.0


def test_construct_contrast_same_type_different_role(coded):
    # Papers sharing AI type "predictive AI" but differing in role: P3 (tool) vs P4 (actor).
    result = construct_contrast(coded, shared_column="ai_type_form", contrast_column="ai_role_function")
    predictive = result[result["shared_value"] == "predictive AI"].iloc[0]
    assert predictive["contrast_value_count"] == 2
    example_ids = {pid for ids in predictive["example_paper_ids"].values() for pid in ids}
    assert {"P3", "P4"} <= example_ids


def test_convergence_by_requires_known_column(coded):
    with pytest.raises(KeyError):
        convergence_by(coded, "nonexistent")
