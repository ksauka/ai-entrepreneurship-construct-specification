"""Tests for pair-selectable coder-robustness re-estimation."""

from __future__ import annotations

import pandas as pd

from aecsp.api.graph_service import GraphService
from aecsp.analytics.coder_robustness import build_coder_robustness


def _frame() -> pd.DataFrame:
    rows = []
    for index in range(6):
        rows.append(
            {
                "paper_id": f"P{index}",
                "in_query_3": "1" if index < 3 else "0",
                "in_query_4": "0" if index < 3 else "1",
                "ai_method_or_phenomenon": (
                    "phenomenon" if index < 3 else "method"
                ),
                "ai_role_function": (
                    "AI as tool" if index < 3 else "AI as research method"
                ),
                "ai_type_form": "machine learning",
                "ai_mechanism_analysis": "improves prediction",
                "level_of_analysis": "firm",
                "entrepreneurial_process_stage": "innovation",
                "scope_conditions": (
                    "scope missing" if index == 0 else "sector-specific"
                ),
                "definition_construct_clarity": "partial definition",
            }
        )
    return pd.DataFrame(rows)


def test_robustness_recomputes_all_five_registered_analyses():
    primary = _frame()
    alternative = _frame()
    alternative.loc[0, "ai_method_or_phenomenon"] = "both"
    alternative.loc[0, "ai_mechanism_analysis"] = "supports learning"

    result = build_coder_robustness(
        primary,
        alternative,
        primary_model="primary",
        primary_label="Primary",
        alternative_model="alternative",
        alternative_label="Alternative",
        min_support=2,
    )

    assert len(result["aggregate_comparison"]) == 7
    assert len(result["nested_comparison"]) == 21
    assert len(result["entrepreneurship_contrasts"]) == 14
    assert len(result["role_level_comparison"]) == 5
    assert len(result["selected_relations"]) == 6
    assert result["summary"]["aggregate_dimensions"] == 7
    assert result["summary"]["core_nested_cells"] == 15


def test_robustness_uses_platform_observed_exclusions():
    result = build_coder_robustness(
        _frame(),
        _frame(),
        primary_model="primary",
        primary_label="Primary",
        alternative_model="alternative",
        alternative_label="Alternative",
        min_support=2,
    )
    scope = next(
        row
        for row in result["aggregate_comparison"]
        if row["dimension_id"] == "scope"
    )
    assert scope["primary"]["denominator"] == 5
    assert scope["alternative"]["denominator"] == 5


def test_service_robustness_accepts_a_selected_model_pair(monkeypatch):
    service = object.__new__(GraphService)
    frames = {
        "gpt-5.4-mini-2026-03-17": _frame(),
        "gemini-3.1-pro-preview": _frame(),
        "claude-sonnet-5": _frame(),
    }
    frames["gemini-3.1-pro-preview"].loc[
        0, "ai_mechanism_analysis"
    ] = "supports learning"
    monkeypatch.setattr(
        service,
        "composition_models",
        lambda: [
            {
                "id": model_id,
                "label": model_id,
                "coverage_share": 1.0,
            }
            for model_id in frames
        ],
    )
    monkeypatch.setattr(
        service,
        "_composition_model_frame",
        lambda model_id: frames[model_id].copy(),
    )

    result = service.primary_coder_robustness(
        reference_model="claude-sonnet-5",
        comparison_model="gemini-3.1-pro-preview",
        min_support=3,
    )

    assert result["reference_model"]["id"] == "claude-sonnet-5"
    assert result["comparison_model"]["id"] == "gemini-3.1-pro-preview"
    assert result["reference_is_registered_primary"] is False
    assert result["min_support"] == 3
    assert len(result["available_models"]) == 3
