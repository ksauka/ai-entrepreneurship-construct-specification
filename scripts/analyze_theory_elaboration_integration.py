"""Connect the targeted-read evidence set to the population model results.

Inputs:
    The 136-paper workbook match, the full primary analysis dataset, and the
    four-rater validation dataset.

Outputs:
    Population-versus-read-set profiles, historical-allocation profiles, a
    23-paper cross-model bridge, agreement statistics, and a provenance
    manifest under data/processed/analysis/theory_elaboration/.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
READ_SET = ROOT / "data/interim/theory_elaboration/theory_elaboration_matched_papers.csv"
OVERLAP = ROOT / "data/interim/theory_elaboration/theory_elaboration_probability_overlap_23.csv"
PRIMARY = ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
VALIDATION = ROOT / "data/processed/analysis/model_validation_dataset.csv"
OUTPUT = ROOT / "data/processed/analysis/theory_elaboration"

DIMENSIONS = {
    "method_or_phenomenon": "ai_method_or_phenomenon",
    "technical_type_form": "ai_type_form",
    "role_function": "ai_role_function",
    "observable_mechanism": "ai_mechanism_analysis",
    "level_of_analysis": "level_of_analysis",
    "scope_condition": "scope_conditions",
    "abstract_definition_clarity": "definition_construct_clarity",
}

RATERS = ("mini", "nano", "claude", "gemini")


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(series: pd.Series) -> pd.Series:
    """Represent empty categorical values consistently."""

    return series.fillna("missing").astype(str).str.strip().replace("", "missing")


def distribution(
    series: pd.Series, dimension: str, evidence_layer: str
) -> pd.DataFrame:
    """Return counts and percentages for one categorical dimension."""

    values = clean(series)
    counts = values.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "dimension": dimension,
            "category": counts.index,
            "evidence_layer": evidence_layer,
            "n": counts.values,
            "denominator": len(values),
            "percent": counts.values / len(values) * 100,
        }
    )


def historical_allocation(row: pd.Series) -> list[str]:
    """Return the historical workbook evidence allocations for one paper."""

    groups = []
    if str(row.get("workbook_p1", "")).strip() == "*":
        groups.append("bottleneck_relocation_evidence")
    if str(row.get("workbook_p2", "")).strip() == "*":
        groups.append("capability_threshold_support")
    if str(row.get("workbook_p2_contrast", "")).strip() == "#":
        groups.append("capability_threshold_counter_case")
    if str(row.get("workbook_p3", "")).strip() == "*":
        groups.append("boundary_condition_evidence")
    return groups or ["unallocated_background"]


def rater_column(data: pd.DataFrame, rater: str, column: str) -> str:
    """Resolve a rater field, falling back to the raw mechanism field."""

    candidate = f"{rater}_{column}"
    if candidate in data.columns:
        return candidate
    if column == "ai_mechanism_analysis":
        fallback = f"{rater}_ai_mechanism"
        if fallback in data.columns:
            return fallback
    raise KeyError(f"No validation column for {rater=} {column=}")


def main() -> None:
    """Build all theory-elaboration integration outputs."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    read_set = pd.read_csv(READ_SET)
    overlap = pd.read_csv(OVERLAP)
    primary = pd.read_csv(PRIMARY, low_memory=False)
    validation = pd.read_csv(VALIDATION, low_memory=False)

    read_coded = read_set.merge(
        primary[["paper_id", *DIMENSIONS.values()]], on="paper_id", how="left", validate="one_to_one"
    )
    if read_coded["ai_role_function"].isna().any():
        raise RuntimeError("At least one targeted-read paper lacks a Mini population record")

    profiles = []
    for label, column in DIMENSIONS.items():
        profiles.append(distribution(primary[column], label, "full_corpus_mini"))
        profiles.append(distribution(read_coded[column], label, "targeted_read_136_mini"))
    dimension_profile = pd.concat(profiles, ignore_index=True)
    dimension_profile.to_csv(OUTPUT / "targeted_read_dimension_profile.csv", index=False)

    exploded = read_coded.copy()
    exploded["historical_allocation"] = exploded.apply(historical_allocation, axis=1)
    exploded = exploded.explode("historical_allocation", ignore_index=True)
    allocation_profiles = []
    for group, group_data in exploded.groupby("historical_allocation", sort=False):
        for label, column in DIMENSIONS.items():
            table = distribution(group_data[column], label, group)
            table.insert(0, "historical_allocation", group)
            allocation_profiles.append(table)
    allocation_profile = pd.concat(allocation_profiles, ignore_index=True)
    allocation_profile.to_csv(
        OUTPUT / "historical_workbook_allocation_profile.csv", index=False
    )

    overlap_validation = overlap.merge(
        validation, on="paper_id", how="left", validate="one_to_one", suffixes=("_workbook", "")
    )
    bridge_columns = [
        "paper_id",
        "workbook_probability_overlap_order",
        "workbook_citations",
        "workbook_topics",
        "workbook_p1",
        "workbook_p2",
        "workbook_p2_contrast",
        "workbook_p3",
        "Title",
        "Year",
    ]
    for rater in RATERS:
        bridge_columns.extend(
            rater_column(overlap_validation, rater, column)
            for column in DIMENSIONS.values()
        )
    available_bridge_columns = [c for c in bridge_columns if c in overlap_validation.columns]
    overlap_validation[available_bridge_columns].to_csv(
        OUTPUT / "workbook_probability_overlap_model_bridge.csv", index=False
    )

    agreement_rows = []
    for label, column in DIMENSIONS.items():
        for left, right in (("mini", "nano"), ("mini", "claude"), ("mini", "gemini"), ("claude", "gemini")):
            left_col = rater_column(overlap_validation, left, column)
            right_col = rater_column(overlap_validation, right, column)
            pair = overlap_validation[[left_col, right_col]].dropna()
            agreement_rows.append(
                {
                    "dimension": label,
                    "rater_pair": f"{left}--{right}",
                    "n_compared": len(pair),
                    "n_exact_agreement": int((pair[left_col] == pair[right_col]).sum()),
                    "exact_agreement": float((pair[left_col] == pair[right_col]).mean()) if len(pair) else None,
                    "interpretation": "case-linked triangulation; not a representative IRR estimate",
                }
            )
    pd.DataFrame(agreement_rows).to_csv(
        OUTPUT / "workbook_probability_overlap_agreement.csv", index=False
    )

    allocation_counts = (
        exploded.groupby("historical_allocation")["paper_id"]
        .nunique()
        .rename("unique_papers")
        .reset_index()
    )
    allocation_counts.to_csv(
        OUTPUT / "historical_workbook_allocation_counts.csv", index=False
    )

    manifest = {
        "analysis_role": {
            "full_corpus_mini": "population-level descriptive analysis",
            "targeted_read_136": "purposive mechanism explanation and theory elaboration",
            "workbook_probability_overlap_23": "case-linked cross-model triangulation only",
        },
        "restrictions": [
            "Do not generalize targeted-read percentages to the full corpus.",
            "Do not treat the 23-paper overlap as a representative IRR sample.",
            "Do not use process-stage or AI-distinction fields for headline claims.",
            "Definition results describe title/abstract/keyword visibility, not full-text absence.",
        ],
        "row_counts": {
            "full_corpus": len(primary),
            "targeted_read": len(read_coded),
            "workbook_probability_overlap": len(overlap_validation),
        },
        "inputs": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (READ_SET, OVERLAP, PRIMARY, VALIDATION)
        },
        "outputs": [
            "targeted_read_dimension_profile.csv",
            "historical_workbook_allocation_profile.csv",
            "workbook_probability_overlap_model_bridge.csv",
            "workbook_probability_overlap_agreement.csv",
            "historical_workbook_allocation_counts.csv",
        ],
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Integrated {len(read_coded):,} targeted-read papers and "
        f"{len(overlap_validation):,} workbook/probability-overlap papers."
    )
    print(f"Outputs -> {OUTPUT}")


if __name__ == "__main__":
    main()
