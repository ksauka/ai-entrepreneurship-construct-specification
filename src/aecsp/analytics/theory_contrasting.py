"""Reusable calculations for the construct-contrasting platform view.

Inputs are paper-level specification records and, for horizontal contrasts,
paper-to-domain assignments. Outputs retain raw filter values so every cell or
configuration can be traced back to its supporting papers.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from aecsp.analytics.observed_composition import (
    OBSERVED_COMPOSITION_PANELS,
    STUDY_STATUS_FILTERS,
)
from aecsp.specification.schema import AI_STUDY_STATUS_COLUMN


MISSING_LABEL: Final[str] = "Missing value"
VALID_DISTRIBUTIONS: Final[tuple[str, ...]] = ("full", "observed")

DIMENSIONS: Final[tuple[dict, ...]] = tuple(
    {
        "id": panel["id"],
        "label": panel["label"],
        "column": panel["column"],
        "fallback_column": panel.get("fallback_column"),
        "excluded": tuple(panel["excluded"]),
    }
    for panel in OBSERVED_COMPOSITION_PANELS
)
DIMENSION_BY_ID: Final[dict[str, dict]] = {
    definition["id"]: definition for definition in DIMENSIONS
}

# Vertical contrasting always crosses one specification dimension with level of
# analysis.  Either axis may carry level so researchers can choose the most
# readable orientation without changing the methodological meaning.
VERTICAL_ROW_DIMENSIONS: Final[tuple[str, ...]] = tuple(
    definition["id"] for definition in DIMENSIONS
)
STRUCTURING_PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("ai_role", "mechanism", "AI role by mechanism"),
    ("ai_role", "level", "AI role by level"),
    ("mechanism", "level", "Mechanism by level"),
    ("ai_role", "scope", "AI role by scope"),
)
CONFIGURATION_DIMENSIONS: Final[tuple[str, ...]] = (
    "ai_role",
    "mechanism",
    "level",
    "scope",
    "process_stage",
)


def dimension_definition(dimension_id: str) -> dict:
    """Return one registered dimension or raise a user-facing error."""

    try:
        return DIMENSION_BY_ID[dimension_id]
    except KeyError as error:
        raise ValueError(f"Unknown specification dimension: {dimension_id}") from error


def dimension_column(frame: pd.DataFrame, dimension_id: str) -> str:
    """Resolve a dimension's analysis column, including the mechanism fallback."""

    definition = dimension_definition(dimension_id)
    column = str(definition["column"])
    fallback = definition.get("fallback_column")
    if column not in frame.columns and fallback and fallback in frame.columns:
        return str(fallback)
    return column


def filter_study_status(frame: pd.DataFrame, study_status: str) -> pd.DataFrame:
    """Apply the shared study-status filter without changing the input frame."""

    if study_status not in STUDY_STATUS_FILTERS:
        raise ValueError(f"Unknown study status: {study_status}")
    if study_status == "all":
        return frame.copy()
    if AI_STUDY_STATUS_COLUMN not in frame.columns:
        return frame.iloc[0:0].copy()
    values = frame[AI_STUDY_STATUS_COLUMN].astype(str).str.strip()
    return frame.loc[values.eq(study_status)].copy()


def observed_mask(frame: pd.DataFrame, dimension_id: str) -> pd.Series:
    """Identify records with a substantive code for one dimension."""

    definition = dimension_definition(dimension_id)
    column = dimension_column(frame, dimension_id)
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column].astype(str).str.strip()
    return values.ne("") & ~values.isin(definition["excluded"])


def distribution(
    frame: pd.DataFrame,
    dimension_id: str,
    distribution_view: str,
    study_status: str = "all",
) -> dict:
    """Return a full or observed category distribution for one dimension."""

    if distribution_view not in VALID_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {distribution_view}")
    work = filter_study_status(frame, study_status)
    definition = dimension_definition(dimension_id)
    column = dimension_column(work, dimension_id)
    if column not in work.columns:
        return {
            "dimension_id": dimension_id,
            "dimension_label": definition["label"],
            "column": column,
            "distribution": distribution_view,
            "full_n": len(work),
            "denominator": 0,
            "categories": [],
        }
    values = work[column].astype(str).str.strip()
    full_n = len(work)
    if distribution_view == "observed":
        mask = observed_mask(work, dimension_id)
        values = values.loc[mask]
    counts = values.value_counts(dropna=False)
    denominator = int(counts.sum())
    categories = [
        {
            "value": MISSING_LABEL if str(value) == "" else str(value),
            "raw_value": str(value),
            "papers": int(count),
            "share": round(int(count) / denominator, 6) if denominator else 0.0,
        }
        for value, count in counts.items()
    ]
    return {
        "dimension_id": dimension_id,
        "dimension_label": definition["label"],
        "column": column,
        "distribution": distribution_view,
        "full_n": full_n,
        "denominator": denominator,
        "categories": categories,
    }


def relationship_matrix(
    frame: pd.DataFrame,
    row_dimension: str,
    column_dimension: str,
    distribution_view: str,
    study_status: str = "all",
) -> dict:
    """Return a traceable two-dimensional count and percentage matrix."""

    if distribution_view not in VALID_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {distribution_view}")
    work = filter_study_status(frame, study_status)
    row_definition = dimension_definition(row_dimension)
    column_definition = dimension_definition(column_dimension)
    row_column = dimension_column(work, row_dimension)
    column_column = dimension_column(work, column_dimension)
    if row_column not in work.columns or column_column not in work.columns:
        return {
            "row_dimension": row_dimension,
            "row_label": row_definition["label"],
            "row_column": row_column,
            "column_dimension": column_dimension,
            "column_label": column_definition["label"],
            "column_column": column_column,
            "distribution": distribution_view,
            "full_n": len(work),
            "analyzed_n": 0,
            "rows": [],
            "columns": [],
            "cells": [],
        }
    if distribution_view == "observed":
        work = work.loc[
            observed_mask(work, row_dimension)
            & observed_mask(work, column_dimension)
        ].copy()
    row_values = work[row_column].astype(str).str.strip()
    column_values = work[column_column].astype(str).str.strip()
    row_display = row_values.mask(row_values.eq(""), MISSING_LABEL)
    column_display = column_values.mask(column_values.eq(""), MISSING_LABEL)
    table = pd.crosstab(row_display, column_display, dropna=False)
    row_order = table.sum(axis=1).sort_values(ascending=False).index.tolist()
    column_order = table.sum(axis=0).sort_values(ascending=False).index.tolist()
    table = table.reindex(index=row_order, columns=column_order, fill_value=0)
    analyzed_n = len(work)
    cells = []
    for row_value in row_order:
        row_total = int(table.loc[row_value].sum())
        for column_value in column_order:
            count = int(table.loc[row_value, column_value])
            column_total = int(table[column_value].sum())
            cells.append(
                {
                    "row_value": str(row_value),
                    "row_raw_value": "" if row_value == MISSING_LABEL else str(row_value),
                    "column_value": str(column_value),
                    "column_raw_value": "" if column_value == MISSING_LABEL else str(column_value),
                    "papers": count,
                    "share_of_analyzed": round(count / analyzed_n, 6) if analyzed_n else 0.0,
                    "share_within_row": round(count / row_total, 6) if row_total else 0.0,
                    "share_within_column": round(count / column_total, 6) if column_total else 0.0,
                }
            )
    return {
        "row_dimension": row_dimension,
        "row_label": row_definition["label"],
        "row_column": row_column,
        "column_dimension": column_dimension,
        "column_label": column_definition["label"],
        "column_column": column_column,
        "distribution": distribution_view,
        "full_n": len(filter_study_status(frame, study_status)),
        "analyzed_n": analyzed_n,
        "rows": [str(value) for value in row_order],
        "columns": [str(value) for value in column_order],
        "cells": cells,
    }


def recurring_configurations(
    frame: pd.DataFrame,
    distribution_view: str,
    study_status: str = "all",
    min_support: int = 10,
) -> dict:
    """Return recurring role-mechanism-level-scope-stage combinations."""

    if min_support < 1:
        raise ValueError("Minimum support must be at least 1")
    if distribution_view not in VALID_DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution: {distribution_view}")
    work = filter_study_status(frame, study_status)
    resolved = {
        dimension_id: dimension_column(work, dimension_id)
        for dimension_id in CONFIGURATION_DIMENSIONS
    }
    missing = [column for column in resolved.values() if column not in work.columns]
    if missing:
        return {
            "distribution": distribution_view,
            "full_n": len(work),
            "analyzed_n": 0,
            "min_support": min_support,
            "dimensions": list(CONFIGURATION_DIMENSIONS),
            "configurations": [],
        }
    if distribution_view == "observed":
        mask = pd.Series(True, index=work.index)
        for dimension_id in CONFIGURATION_DIMENSIONS:
            mask &= observed_mask(work, dimension_id)
        work = work.loc[mask].copy()
    columns = [resolved[dimension_id] for dimension_id in CONFIGURATION_DIMENSIONS]
    values = work[columns].astype(str).apply(lambda column: column.str.strip())
    display = values.mask(values.eq(""), MISSING_LABEL)
    counts = (
        display.value_counts(dropna=False)
        .rename("papers")
        .reset_index()
    )
    counts = counts.loc[counts["papers"].ge(min_support)].copy()
    analyzed_n = len(work)
    records = []
    for _, row in counts.iterrows():
        record = {
            "papers": int(row["papers"]),
            "share": round(int(row["papers"]) / analyzed_n, 6) if analyzed_n else 0.0,
        }
        filters = {}
        for dimension_id, column in resolved.items():
            display_value = str(row[column])
            record[dimension_id] = display_value
            filters[column] = "" if display_value == MISSING_LABEL else display_value
        record["filters"] = filters
        records.append(record)
    return {
        "distribution": distribution_view,
        "full_n": len(filter_study_status(frame, study_status)),
        "analyzed_n": analyzed_n,
        "min_support": min_support,
        "dimensions": list(CONFIGURATION_DIMENSIONS),
        "configurations": records,
    }
