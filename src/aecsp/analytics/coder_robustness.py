"""Re-estimate the registered theory-elaboration results with another coder.

The comparison intentionally keeps the analytical populations, missing-value
rules, selected contrasts, and support threshold fixed.  It asks whether the
substantive findings change when an alternative model supplies the paper-level
codes; it does not select a model by agreement or create a consensus code.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from aecsp.analytics.observed_composition import OBSERVED_COMPOSITION_PANELS


AGGREGATE_DIMENSIONS: tuple[str, ...] = (
    "study_status",
    "ai_role",
    "technical_type",
    "mechanism",
    "level",
    "process_stage",
    "scope",
)

NESTED_OUTCOMES: tuple[str, ...] = (
    "ai_role",
    "technical_type",
    "mechanism",
    "level",
    "process_stage",
    "scope",
    "definition",
)

CORE_NESTED_OUTCOMES: frozenset[str] = frozenset(
    {"ai_role", "technical_type", "mechanism", "level", "scope"}
)

ENTREPRENEURSHIP_CONTRASTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("study_status", ("method", "both")),
    ("ai_role", ("AI as research method", "AI as firm capability")),
    ("technical_type", ("machine learning", "generative AI")),
    ("mechanism", ("improves prediction", "supports learning")),
    ("level", ("individual entrepreneur", "firm")),
    ("process_stage", ("innovation", "resource acquisition")),
    ("scope", ("country-specific", "sector-specific")),
)

ROLE_LEVEL_ROLES: tuple[str, ...] = (
    "AI as tool",
    "AI as research method",
    "AI as firm capability",
    "AI as actor/agent",
    "AI as context",
)

SELECTED_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("AI as tool", "improves prediction", "eid:2-s2.0-105034800856"),
    ("AI as firm capability", "supports learning", "eid:2-s2.0-105018343754"),
    ("AI as research method", "improves prediction", "eid:2-s2.0-105039062930"),
    ("AI as tool", "alters judgment", "eid:2-s2.0-105034750556"),
    ("AI as context", "transforms stakeholder interaction", "eid:2-s2.0-85217167240"),
    ("AI as tool", "reduces uncertainty", "eid:2-s2.0-105017260726"),
)

_PANELS = {panel["id"]: panel for panel in OBSERVED_COMPOSITION_PANELS}


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "x"}
    )


def _population(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    core = _truthy(frame["in_query_3"])
    additional = _truthy(frame["in_query_4"])
    if population == "core":
        return frame[core].copy()
    if population == "additional":
        return frame[additional].copy()
    if population == "combined":
        return frame[core | additional].copy()
    raise ValueError(f"Unknown entrepreneurship population: {population}")


def _observed_distribution(
    frame: pd.DataFrame,
    dimension_id: str,
) -> tuple[int, pd.Series, pd.Series]:
    panel = _PANELS[dimension_id]
    values = frame[panel["column"]].astype(str).str.strip()
    excluded = {"", *(str(value).strip() for value in panel["excluded"])}
    values = values[~values.isin(excluded)]
    counts = values.value_counts()
    shares = counts.div(len(values)) if len(values) else counts.astype(float)
    return int(len(values)), counts, shares


def _top_result(frame: pd.DataFrame, dimension_id: str) -> dict[str, Any]:
    denominator, counts, shares = _observed_distribution(frame, dimension_id)
    if counts.empty:
        return {
            "denominator": 0,
            "category": "",
            "papers": 0,
            "share": 0.0,
        }
    category = str(counts.index[0])
    return {
        "denominator": denominator,
        "category": category,
        "papers": int(counts.iloc[0]),
        "share": float(shares.iloc[0]),
    }


def _direction(value: float, tolerance: float = 1e-12) -> str:
    if value > tolerance:
        return "higher in Core"
    if value < -tolerance:
        return "higher in Additional"
    return "no difference"


def _model_distribution_records(
    frame: pd.DataFrame,
    model_role: str,
    dimension_id: str,
    **context: Any,
) -> list[dict[str, Any]]:
    denominator, counts, shares = _observed_distribution(frame, dimension_id)
    return [
        {
            **context,
            "model_role": model_role,
            "dimension_id": dimension_id,
            "dimension_label": _PANELS[dimension_id]["label"],
            "denominator": denominator,
            "category": str(category),
            "papers": int(count),
            "share": float(shares.loc[category]),
            "rank": rank,
        }
        for rank, (category, count) in enumerate(counts.items(), start=1)
    ]


def build_coder_robustness(
    primary_frame: pd.DataFrame,
    alternative_frame: pd.DataFrame,
    *,
    primary_model: str,
    primary_label: str,
    alternative_model: str,
    alternative_label: str,
    min_support: int = 10,
) -> dict[str, Any]:
    """Recompute the five registered manuscript analyses with two coders."""

    model_frames = {
        "primary": primary_frame,
        "alternative": alternative_frame,
    }
    labels = {
        "primary": {"id": primary_model, "label": primary_label},
        "alternative": {"id": alternative_model, "label": alternative_label},
    }
    populations = {
        role: {
            population: _population(frame, population)
            for population in ("core", "additional", "combined")
        }
        for role, frame in model_frames.items()
    }

    aggregate_distributions: list[dict[str, Any]] = []
    aggregate_comparison: list[dict[str, Any]] = []
    for dimension_id in AGGREGATE_DIMENSIONS:
        tops = {}
        for role in model_frames:
            combined = populations[role]["combined"]
            aggregate_distributions.extend(
                _model_distribution_records(
                    combined,
                    role,
                    dimension_id,
                    population="combined",
                )
            )
            tops[role] = _top_result(combined, dimension_id)
        aggregate_comparison.append(
            {
                "dimension_id": dimension_id,
                "dimension_label": _PANELS[dimension_id]["label"],
                "primary": tops["primary"],
                "alternative": tops["alternative"],
                "same_leading_category": (
                    tops["primary"]["category"]
                    == tops["alternative"]["category"]
                ),
                "leading_share_difference_pp": round(
                    (tops["alternative"]["share"] - tops["primary"]["share"])
                    * 100,
                    4,
                ),
            }
        )

    nested_distributions: list[dict[str, Any]] = []
    nested_comparison: list[dict[str, Any]] = []
    for status in ("phenomenon", "method", "both"):
        for dimension_id in NESTED_OUTCOMES:
            tops = {}
            status_totals = {}
            for role in model_frames:
                combined = populations[role]["combined"]
                status_frame = combined[
                    combined["ai_method_or_phenomenon"]
                    .astype(str)
                    .str.strip()
                    .eq(status)
                ].copy()
                status_totals[role] = int(len(status_frame))
                nested_distributions.extend(
                    _model_distribution_records(
                        status_frame,
                        role,
                        dimension_id,
                        population="combined",
                        study_status=status,
                        status_papers=len(status_frame),
                    )
                )
                tops[role] = _top_result(status_frame, dimension_id)
            nested_comparison.append(
                {
                    "study_status": status,
                    "dimension_id": dimension_id,
                    "dimension_label": _PANELS[dimension_id]["label"],
                    "analytical_status": (
                        "Core"
                        if dimension_id in CORE_NESTED_OUTCOMES
                        else "Exploratory"
                    ),
                    "primary_status_papers": status_totals["primary"],
                    "alternative_status_papers": status_totals["alternative"],
                    "primary": tops["primary"],
                    "alternative": tops["alternative"],
                    "same_leading_category": (
                        tops["primary"]["category"]
                        == tops["alternative"]["category"]
                    ),
                }
            )

    entrepreneurship_contrasts: list[dict[str, Any]] = []
    for dimension_id, categories in ENTREPRENEURSHIP_CONTRASTS:
        for category in categories:
            values = {}
            for role in model_frames:
                core_n, _, core_shares = _observed_distribution(
                    populations[role]["core"], dimension_id
                )
                additional_n, _, additional_shares = _observed_distribution(
                    populations[role]["additional"], dimension_id
                )
                core_share = float(core_shares.get(category, 0.0))
                additional_share = float(additional_shares.get(category, 0.0))
                difference_pp = (core_share - additional_share) * 100
                values[role] = {
                    "core_denominator": core_n,
                    "additional_denominator": additional_n,
                    "core_share": core_share,
                    "additional_share": additional_share,
                    "difference_pp": difference_pp,
                    "direction": _direction(difference_pp),
                }
            entrepreneurship_contrasts.append(
                {
                    "dimension_id": dimension_id,
                    "dimension_label": _PANELS[dimension_id]["label"],
                    "category": category,
                    "primary": values["primary"],
                    "alternative": values["alternative"],
                    "direction_preserved": (
                        values["primary"]["direction"]
                        == values["alternative"]["direction"]
                    ),
                }
            )

    role_level_cells: list[dict[str, Any]] = []
    role_level_comparison: list[dict[str, Any]] = []
    for role_value in ROLE_LEVEL_ROLES:
        leaders = {}
        for role in model_frames:
            combined = populations[role]["combined"]
            role_frame = combined[
                combined["ai_role_function"]
                .astype(str)
                .str.strip()
                .eq(role_value)
            ].copy()
            denominator, counts, shares = _observed_distribution(role_frame, "level")
            leaders[role] = str(counts.index[0]) if len(counts) else ""
            for rank, (level_value, papers) in enumerate(counts.items(), start=1):
                role_level_cells.append(
                    {
                        "model_role": role,
                        "role": role_value,
                        "level": str(level_value),
                        "papers": int(papers),
                        "role_denominator": denominator,
                        "share_within_role": float(shares.loc[level_value]),
                        "rank": rank,
                    }
                )
        role_level_comparison.append(
            {
                "role": role_value,
                "primary_leading_level": leaders["primary"],
                "alternative_leading_level": leaders["alternative"],
                "same_leading_level": leaders["primary"] == leaders["alternative"],
            }
        )

    selected_relations: list[dict[str, Any]] = []
    for role_value, mechanism_value, evidence_paper_id in SELECTED_RELATIONS:
        values = {}
        for role in model_frames:
            combined = populations[role]["combined"]
            matching = combined[
                combined["ai_role_function"]
                .astype(str)
                .str.strip()
                .eq(role_value)
                & combined["ai_mechanism_analysis"]
                .astype(str)
                .str.strip()
                .eq(mechanism_value)
            ]
            evidence = combined[combined["paper_id"].astype(str).eq(evidence_paper_id)]
            evidence_matches = bool(
                len(evidence)
                and str(evidence.iloc[0]["ai_role_function"]).strip() == role_value
                and str(evidence.iloc[0]["ai_mechanism_analysis"]).strip()
                == mechanism_value
            )
            values[role] = {
                "papers": int(len(matching)),
                "core_papers": int(_truthy(matching["in_query_3"]).sum()),
                "additional_papers": int(_truthy(matching["in_query_4"]).sum()),
                "meets_support_threshold": int(len(matching)) >= min_support,
                "evidence_paper_matches_relation": evidence_matches,
            }
        selected_relations.append(
            {
                "role": role_value,
                "mechanism": mechanism_value,
                "relation": f"{role_value} × {mechanism_value}",
                "evidence_paper_id": evidence_paper_id,
                "min_support": min_support,
                "primary": values["primary"],
                "alternative": values["alternative"],
                "retained_by_both": (
                    values["primary"]["meets_support_threshold"]
                    and values["alternative"]["meets_support_threshold"]
                ),
            }
        )

    aggregate_matches = sum(
        row["same_leading_category"] for row in aggregate_comparison
    )
    nested_matches = sum(row["same_leading_category"] for row in nested_comparison)
    core_nested = [
        row for row in nested_comparison if row["analytical_status"] == "Core"
    ]
    core_nested_matches = sum(row["same_leading_category"] for row in core_nested)
    contrast_matches = sum(
        row["direction_preserved"] for row in entrepreneurship_contrasts
    )
    role_level_matches = sum(
        row["same_leading_level"] for row in role_level_comparison
    )
    relation_matches = sum(row["retained_by_both"] for row in selected_relations)
    evidence_matches = sum(
        row["alternative"]["evidence_paper_matches_relation"]
        for row in selected_relations
    )
    conclusion = (
        "partly coder-dependent"
        if nested_matches < len(nested_comparison)
        or relation_matches < len(selected_relations)
        else "stable across coders"
    )

    return {
        "primary_model": labels["primary"],
        "alternative_model": labels["alternative"],
        "population": "Combined entrepreneurship",
        "min_support": min_support,
        "aggregate_distributions": aggregate_distributions,
        "aggregate_comparison": aggregate_comparison,
        "nested_distributions": nested_distributions,
        "nested_comparison": nested_comparison,
        "entrepreneurship_contrasts": entrepreneurship_contrasts,
        "role_level_cells": role_level_cells,
        "role_level_comparison": role_level_comparison,
        "selected_relations": selected_relations,
        "summary": {
            "conclusion": conclusion,
            "aggregate_leading_matches": aggregate_matches,
            "aggregate_dimensions": len(aggregate_comparison),
            "nested_leading_matches": nested_matches,
            "nested_cells": len(nested_comparison),
            "core_nested_leading_matches": core_nested_matches,
            "core_nested_cells": len(core_nested),
            "entrepreneurship_contrast_directions_preserved": contrast_matches,
            "entrepreneurship_contrasts": len(entrepreneurship_contrasts),
            "role_leading_levels_preserved": role_level_matches,
            "roles_compared": len(role_level_comparison),
            "selected_relations_retained_by_both": relation_matches,
            "selected_relations": len(selected_relations),
            "selected_evidence_papers_matching_under_alternative": evidence_matches,
        },
        "interpretation": (
            "Leading aggregate categories and journal-population contrast directions "
            "are robust, but the nested study-status analysis and selected recurring "
            "relations contain coder-dependent results. Agreement is not accuracy, and "
            "the alternative codes do not replace the registered primary record."
        ),
    }
