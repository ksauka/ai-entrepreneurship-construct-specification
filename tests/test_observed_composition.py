"""Tests for the dynamic observed-composition view."""

import pandas as pd
import pytest

from aecsp.analytics.observed_composition import (
    analyze_observed_composition,
    observed_composition_evidence_mask,
)


@pytest.fixture
def papers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_id": "P1",
                "ai_method_or_phenomenon": "phenomenon",
                "ai_role_function": "AI as tool",
                "ai_type_form": "machine learning",
                "ai_mechanism_analysis": "improves prediction",
                "level_of_analysis": "firm",
                "entrepreneurial_process_stage": "innovation",
                "scope_conditions": "sector-specific",
                "definition_construct_clarity": "partial definition",
            },
            {
                "paper_id": "P2",
                "ai_method_or_phenomenon": "method",
                "ai_role_function": "AI as research method",
                "ai_type_form": "machine learning",
                "ai_mechanism_analysis": "mechanism missing",
                "level_of_analysis": "firm",
                "entrepreneurial_process_stage": "process unspecified",
                "scope_conditions": "generalised without scope",
                "definition_construct_clarity": "no definition",
            },
            {
                "paper_id": "P3",
                "ai_method_or_phenomenon": "both",
                "ai_role_function": "AI as actor/agent",
                "ai_type_form": "generative AI",
                "ai_mechanism_analysis": "supports learning",
                "level_of_analysis": "individual entrepreneur",
                "entrepreneurial_process_stage": "ideation",
                "scope_conditions": "early-stage ventures",
                "definition_construct_clarity": "definition by example only",
            },
            {
                "paper_id": "P4",
                "ai_method_or_phenomenon": "unclear",
                "ai_role_function": "AI as unspecified label",
                "ai_type_form": "unspecified AI",
                "ai_mechanism_analysis": "mechanism missing",
                "level_of_analysis": "unspecified level",
                "entrepreneurial_process_stage": "process unspecified",
                "scope_conditions": "scope missing",
                "definition_construct_clarity": "no definition",
            },
        ]
    )


def test_all_view_reproduces_panel_specific_observed_denominators(papers):
    result = analyze_observed_composition(papers)
    panels = {panel["id"]: panel for panel in result["panels"]}

    assert result["filtered_papers"] == 4
    assert panels["study_status"]["observed_n"] == 3
    assert panels["technical_type"]["observed_n"] == 3
    assert panels["mechanism"]["observed_n"] == 2
    assert panels["scope"]["observed_n"] == 2
    assert panels["definition"]["observed_n"] == 2
    assert panels["technical_type"]["full_n"] == 4
    assert panels["technical_type"]["full_categories"][0]["value"] == "machine learning"
    unspecified = next(
        item
        for item in panels["technical_type"]["comparison_categories"]
        if item["value"] == "unspecified AI"
    )
    assert unspecified["full_count"] == 1
    assert unspecified["full_share"] == 0.25
    assert unspecified["observed_count"] == 0
    assert unspecified["observed_share"] is None
    assert unspecified["is_observed"] is False
    assert panels["technical_type"]["categories"][0] == {
        "value": "machine learning",
        "count": 2,
        "share": 0.666667,
    }


def test_study_status_filter_is_applied_before_every_panel(papers):
    result = analyze_observed_composition(papers, study_status="method")
    panels = {panel["id"]: panel for panel in result["panels"]}

    assert result["filtered_papers"] == 1
    assert panels["study_status"]["categories"] == [
        {"value": "method", "count": 1, "share": 1.0}
    ]
    assert panels["technical_type"]["categories"][0]["value"] == "machine learning"
    assert panels["mechanism"]["observed_n"] == 0
    assert panels["process_stage"]["observed_n"] == 0


def test_table_contains_categories_omitted_from_chart():
    frame = pd.DataFrame(
        [
            {
                "paper_id": f"P{index}",
                "ai_method_or_phenomenon": "phenomenon",
                "ai_type_form": f"type {index}",
            }
            for index in range(10)
        ]
    )
    result = analyze_observed_composition(frame)
    panel = next(item for item in result["panels"] if item["id"] == "technical_type")

    assert len(panel["chart_categories"]) == 8
    assert len(panel["categories"]) == 10
    assert panel["omitted_categories"] == 2
    assert panel["omitted_papers"] == 2
    assert sum(item["count"] for item in panel["categories"]) == panel["observed_n"]
    assert sum(item["count"] for item in panel["full_categories"]) == panel["full_n"]


def test_evidence_mask_respects_status_and_category(papers):
    mask = observed_composition_evidence_mask(
        papers,
        study_status="phenomenon",
        column="ai_type_form",
        value="machine learning",
    )
    assert papers.loc[mask, "paper_id"].tolist() == ["P1"]


def test_invalid_filter_or_column_is_rejected(papers):
    with pytest.raises(ValueError, match="Unknown AI positioning"):
        analyze_observed_composition(papers, study_status="invalid")
    with pytest.raises(ValueError, match="Unknown composition column"):
        observed_composition_evidence_mask(papers, "all", "Title", "Anything")
