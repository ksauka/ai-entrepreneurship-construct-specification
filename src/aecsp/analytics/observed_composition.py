"""Calculate observed construct composition under a study-status filter.

Inputs: paper-level specification codes and an optional study-status value.
Outputs: per-dimension observed denominators, category shares, and evidence masks.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from aecsp.specification.schema import AI_STUDY_STATUS_COLUMN


STUDY_STATUS_FILTERS: Final[tuple[str, ...]] = ("all", "phenomenon", "method", "both")

OBSERVED_COMPOSITION_PANELS: Final[tuple[dict, ...]] = (
    {
        "id": "study_status",
        "label": "Study status",
        "column": AI_STUDY_STATUS_COLUMN,
        "denominator_label": "clear codes",
        "excluded": ("unclear",),
        "top_n": 3,
    },
    {
        "id": "ai_role",
        "label": "AI role",
        "column": "ai_role_function",
        "denominator_label": "substantive roles",
        "excluded": ("AI as unspecified label",),
        "top_n": 6,
    },
    {
        "id": "technical_type",
        "label": "Technical type",
        "column": "ai_type_form",
        "denominator_label": "named types",
        "excluded": ("unspecified AI",),
        "top_n": 8,
    },
    {
        "id": "mechanism",
        "label": "Mechanism",
        "column": "ai_mechanism_analysis",
        "fallback_column": "ai_mechanism",
        "denominator_label": "observed mechanisms",
        "excluded": ("mechanism missing",),
        "top_n": 8,
    },
    {
        "id": "level",
        "label": "Level",
        "column": "level_of_analysis",
        "denominator_label": "specified levels",
        "excluded": ("unspecified level",),
        "top_n": 8,
    },
    {
        "id": "process_stage",
        "label": "Process stage",
        "column": "entrepreneurial_process_stage",
        "denominator_label": "specified stages",
        "excluded": ("process unspecified",),
        "top_n": 8,
    },
    {
        "id": "scope",
        "label": "Scope",
        "column": "scope_conditions",
        "denominator_label": "stated boundaries",
        "excluded": ("generalised without scope", "scope missing"),
        "top_n": 8,
    },
    {
        "id": "definition",
        "label": "Definition",
        "column": "definition_construct_clarity",
        "denominator_label": "definitional signals",
        "excluded": ("no definition",),
        "top_n": 4,
    },
)


def _panel_column(frame: pd.DataFrame, panel: dict) -> str:
    column = panel["column"]
    if column not in frame.columns and panel.get("fallback_column") in frame.columns:
        return panel["fallback_column"]
    return column


def _status_frame(frame: pd.DataFrame, study_status: str) -> pd.DataFrame:
    if study_status not in STUDY_STATUS_FILTERS:
        raise ValueError(f"Unknown study status: {study_status}")
    if study_status == "all":
        return frame
    if AI_STUDY_STATUS_COLUMN not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame[AI_STUDY_STATUS_COLUMN].astype(str).str.strip() == study_status]


def analyze_observed_composition(
    frame: pd.DataFrame,
    study_status: str = "all",
) -> dict:
    """Return Figure-14-equivalent distributions from the current dataset.

    The selected study status first filters the analytical population. Each
    panel then removes only that dimension's declared unobserved values and
    calculates shares using its own observed denominator.
    """

    filtered = _status_frame(frame, study_status)
    base_n = len(filtered)
    if AI_STUDY_STATUS_COLUMN in frame.columns:
        status_counts = frame[AI_STUDY_STATUS_COLUMN].astype(str).str.strip().value_counts()
    else:
        status_counts = pd.Series(dtype="int64")

    panels = []
    for definition in OBSERVED_COMPOSITION_PANELS:
        column = _panel_column(filtered, definition)
        if column not in filtered.columns:
            panels.append(
                {
                    **{key: definition[key] for key in ("id", "label", "denominator_label")},
                    "column": column,
                    "observed_n": 0,
                    "observed_share": 0.0,
                    "full_n": base_n,
                    "full_categories": [],
                    "categories": [],
                    "chart_categories": [],
                    "comparison_categories": [],
                    "chart_comparison_categories": [],
                    "omitted_categories": 0,
                    "omitted_papers": 0,
                }
            )
            continue

        values = filtered[column].astype(str).str.strip()
        full_counts = values.value_counts(dropna=False)
        full_categories = [
            {
                "value": "Missing value" if value == "" else str(value),
                "raw_value": str(value),
                "count": int(count),
                "share": round(int(count) / base_n, 6) if base_n else 0.0,
            }
            for value, count in full_counts.items()
        ]
        observed_mask = values.ne("") & ~values.isin(definition["excluded"])
        counts = values[observed_mask].value_counts()
        observed_n = int(counts.sum())
        categories = [
            {
                "value": str(value),
                "count": int(count),
                "share": round(int(count) / observed_n, 6) if observed_n else 0.0,
            }
            for value, count in counts.items()
        ]
        chart_categories = categories[: definition["top_n"]]
        omitted = categories[definition["top_n"] :]
        observed_lookup = {item["value"]: item for item in categories}
        comparison_categories = [
            {
                "value": item["value"],
                "raw_value": item["raw_value"],
                "full_count": item["count"],
                "full_share": item["share"],
                "observed_count": observed_lookup.get(item["raw_value"], {}).get(
                    "count", 0
                ),
                "observed_share": observed_lookup.get(item["raw_value"], {}).get(
                    "share"
                ),
                "is_observed": item["raw_value"] in observed_lookup,
            }
            for item in full_categories
        ]
        chart_values = {
            item["value"] for item in full_categories[: definition["top_n"]]
        } | {item["value"] for item in chart_categories}
        chart_comparison_categories = [
            item for item in comparison_categories if item["value"] in chart_values
        ]
        panels.append(
            {
                **{key: definition[key] for key in ("id", "label", "denominator_label")},
                "column": column,
                "full_n": base_n,
                "full_categories": full_categories,
                "observed_n": observed_n,
                "observed_share": round(observed_n / base_n, 6) if base_n else 0.0,
                "categories": categories,
                "chart_categories": chart_categories,
                "comparison_categories": comparison_categories,
                "chart_comparison_categories": chart_comparison_categories,
                "omitted_categories": int(len(omitted)),
                "omitted_papers": sum(category["count"] for category in omitted),
            }
        )

    return {
        "study_status": study_status,
        "scope_papers": len(frame),
        "filtered_papers": base_n,
        "status_options": [
            {
                "value": status,
                "label": status.capitalize(),
                "papers": int(status_counts.get(status, 0)),
            }
            for status in STUDY_STATUS_FILTERS[1:]
        ],
        "panels": panels,
    }


def observed_composition_evidence_mask(
    frame: pd.DataFrame,
    study_status: str,
    column: str,
    value: str,
) -> pd.Series:
    """Select evidence for one displayed composition category."""

    allowed_columns = {
        candidate
        for panel in OBSERVED_COMPOSITION_PANELS
        for candidate in (panel["column"], panel.get("fallback_column"))
        if candidate
    }
    if column not in allowed_columns:
        raise ValueError(f"Unknown composition column: {column}")
    status_mask = pd.Series(True, index=frame.index)
    if study_status not in STUDY_STATUS_FILTERS:
        raise ValueError(f"Unknown study status: {study_status}")
    if study_status != "all":
        if AI_STUDY_STATUS_COLUMN not in frame.columns:
            return pd.Series(False, index=frame.index)
        status_mask = (
            frame[AI_STUDY_STATUS_COLUMN].astype(str).str.strip() == study_status
        )
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return status_mask & (frame[column].astype(str).str.strip() == value)
