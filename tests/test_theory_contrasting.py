import pandas as pd

from aecsp.analytics.theory_contrasting import (
    distribution,
    recurring_configurations,
    relationship_matrix,
)


def _papers():
    return pd.DataFrame(
        [
            {
                "paper_id": "p1",
                "ai_method_or_phenomenon": "phenomenon",
                "ai_role_function": "AI as tool",
                "ai_type_form": "machine learning",
                "ai_mechanism_analysis": "improves prediction",
                "level_of_analysis": "venture",
                "scope_conditions": "sector-specific",
                "entrepreneurial_process_stage": "venture creation",
            },
            {
                "paper_id": "p2",
                "ai_method_or_phenomenon": "phenomenon",
                "ai_role_function": "AI as actor/agent",
                "ai_type_form": "generative AI",
                "ai_mechanism_analysis": "supports learning",
                "level_of_analysis": "venture",
                "scope_conditions": "sector-specific",
                "entrepreneurial_process_stage": "venture creation",
            },
            {
                "paper_id": "p3",
                "ai_method_or_phenomenon": "method",
                "ai_role_function": "AI as unspecified label",
                "ai_type_form": "unspecified AI",
                "ai_mechanism_analysis": "mechanism missing",
                "level_of_analysis": "unspecified level",
                "scope_conditions": "scope missing",
                "entrepreneurial_process_stage": "process unspecified",
            },
        ]
    )


def test_distribution_switches_denominator_between_full_and_observed():
    full = distribution(_papers(), "ai_role", "full")
    observed = distribution(_papers(), "ai_role", "observed")

    assert full["denominator"] == 3
    assert observed["denominator"] == 2
    assert {item["raw_value"] for item in observed["categories"]} == {
        "AI as tool",
        "AI as actor/agent",
    }


def test_relationship_matrix_reports_column_and_total_shares():
    result = relationship_matrix(
        _papers(), "ai_role", "level", "observed"
    )

    assert result["analyzed_n"] == 2
    assert result["columns"] == ["venture"]
    cells = {item["row_value"]: item for item in result["cells"]}
    assert cells["AI as tool"]["papers"] == 1
    assert cells["AI as tool"]["share_within_column"] == 0.5


def test_configurations_exclude_unspecified_records_in_observed_view():
    result = recurring_configurations(
        _papers(), "observed", min_support=1
    )

    assert result["analyzed_n"] == 2
    assert len(result["configurations"]) == 2
    assert all(
        record["ai_role"] != "AI as unspecified label"
        for record in result["configurations"]
    )
