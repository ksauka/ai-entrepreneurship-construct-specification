"""Build evidence-grounding diagnostics and a stated-evidence Table 8 check.

The production coder could see source journal and publication year as record
metadata, although the frozen instructions permitted coding evidence only from
the title, abstract, and author keywords. This script uses the retained
``evidence_type`` fields to test whether the selected horizontal contrasts hold
when the outcome is restricted to substantive codes supported by stated
evidence. It does not recode papers or alter any frozen model output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from aecsp.analytics.theory_contrasting import (
    DIMENSIONS,
    dimension_column,
    observed_mask,
)


ROOT = Path(__file__).resolve().parents[1]
SPECIFICATION_DIR = ROOT / "data/processed/specification"
MASTER = ROOT / "data/processed/master_corpus.csv"
PRIMARY = ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DOMAIN_DIR = ROOT / "data/processed/analysis/theory_elaboration/domains"
OUTPUT_DIR = ROOT / "reports/analysis/tables/contrasting"
ORIGINAL_HORIZONTAL = OUTPUT_DIR / "horizontal_domain_contrast_full_corpus.csv"

MODEL_FILES = {
    "gpt-5.4-mini-2026-03-17": SPECIFICATION_DIR
    / "paper_specifications_gpt-5.4-mini-2026-03-17_spec-v3.csv",
    "gpt-4.1-nano-2025-04-14": SPECIFICATION_DIR
    / "paper_specifications_gpt-4.1-nano-2025-04-14_spec-v3.csv",
    "claude-sonnet-5": SPECIFICATION_DIR
    / "paper_specifications_claude-sonnet-5_spec-v3.csv",
    "gemini-3.1-pro-preview": SPECIFICATION_DIR
    / "paper_specifications_gemini-3.1-pro-preview_spec-v3.csv",
}

MODEL_LABELS = {
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 Mini",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 Nano",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
}

EVIDENCE_DIMENSIONS = tuple(
    definition for definition in DIMENSIONS if definition["id"] != "study_status"
)

POPULATIONS = (
    ("full_corpus", "Full corpus"),
    ("leading_entrepreneurship", "Leading entrepreneurship journals"),
    ("additional_entrepreneurship", "Additional entrepreneurship journals"),
    ("combined_entrepreneurship", "Combined entrepreneurship"),
    ("ft50", "FT50"),
)

SELECTED_TABLE8_ROWS = (
    ("Management Science and Operations Research", "ai_role", "AI as tool"),
    ("Marketing", "mechanism", "transforms stakeholder interaction"),
    ("Management of Technology and Innovation", "technical_type", "generative AI"),
    ("Organization studies", "mechanism", "alters judgment"),
    ("Finance", "technical_type", "machine learning"),
    ("Environmental and sustainability", "mechanism", "improves prediction"),
    ("Leading entrepreneurship journals", "ai_role", "AI as research method"),
    ("Additional entrepreneurship", "ai_role", "AI as firm capability"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def population_mask(frame: pd.DataFrame, population: str) -> pd.Series:
    """Return the exact registered population mask used elsewhere in analysis."""

    if population == "full_corpus":
        return pd.Series(True, index=frame.index)
    q2 = _truthy(frame["in_query_2"])
    q3 = _truthy(frame["in_query_3"])
    q4 = _truthy(frame["in_query_4"])
    if population == "leading_entrepreneurship":
        return q3
    if population == "additional_entrepreneurship":
        return q4
    if population == "combined_entrepreneurship":
        return q3 | q4
    if population == "ft50":
        return q2
    raise ValueError(f"Unknown population: {population}")


def evidence_column(dimension_id: str) -> str:
    definition = next(item for item in EVIDENCE_DIMENSIONS if item["id"] == dimension_id)
    source_column = str(definition["column"])
    if dimension_id == "mechanism":
        source_column = "ai_mechanism"
    return f"{source_column}_evidence_type"


def load_model_frame(model: str, membership: pd.DataFrame) -> pd.DataFrame:
    """Load only fields required for the grounding diagnostic."""

    columns = {"paper_id"}
    for definition in EVIDENCE_DIMENSIONS:
        columns.add(str(definition["column"]))
        fallback = definition.get("fallback_column")
        if fallback:
            columns.add(str(fallback))
        columns.add(evidence_column(str(definition["id"])))
    available = set(pd.read_csv(MODEL_FILES[model], nrows=0).columns)
    frame = pd.read_csv(
        MODEL_FILES[model],
        usecols=sorted(columns & available),
        dtype=str,
        keep_default_na=False,
    )
    return frame.merge(membership, on="paper_id", how="left", validate="one_to_one")


def build_evidence_type_distribution() -> pd.DataFrame:
    membership = pd.read_csv(
        MASTER,
        usecols=["paper_id", "in_query_2", "in_query_3", "in_query_4"],
        dtype=str,
        keep_default_na=False,
    ).drop_duplicates("paper_id")
    rows: list[dict] = []
    for model in MODEL_FILES:
        frame = load_model_frame(model, membership)
        for population, population_label in POPULATIONS:
            selected = frame.loc[population_mask(frame, population)].copy()
            for definition in EVIDENCE_DIMENSIONS:
                dimension_id = str(definition["id"])
                evidence = selected[evidence_column(dimension_id)].astype(str).str.strip()
                evidence = evidence.mask(evidence.eq(""), "missing evidence label")
                for evidence_view, view_frame, view_evidence in (
                    ("all_codes", selected, evidence),
                    (
                        "observed_substantive",
                        selected.loc[observed_mask(selected, dimension_id)],
                        evidence.loc[observed_mask(selected, dimension_id)],
                    ),
                ):
                    counts = view_evidence.value_counts(dropna=False)
                    denominator = int(len(view_frame))
                    for evidence_type, count in counts.items():
                        rows.append(
                            {
                                "model": model,
                                "model_label": MODEL_LABELS[model],
                                "population": population,
                                "population_label": population_label,
                                "dimension_id": dimension_id,
                                "dimension_label": definition["label"],
                                "evidence_view": evidence_view,
                                "denominator": denominator,
                                "evidence_type": str(evidence_type),
                                "papers": int(count),
                                "share": round(int(count) / denominator, 6)
                                if denominator
                                else 0.0,
                            }
                        )
    return pd.DataFrame(rows)


def load_primary_and_assignments() -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"paper_id"}
    for definition in EVIDENCE_DIMENSIONS:
        required.add(str(definition["column"]))
        fallback = definition.get("fallback_column")
        if fallback:
            required.add(str(fallback))
        required.add(evidence_column(str(definition["id"])))
    frame = pd.read_csv(
        PRIMARY,
        usecols=lambda column: column in required,
        dtype=str,
        keep_default_na=False,
    )
    business = pd.read_csv(
        DOMAIN_DIR / "business_domain_assignments.csv",
        dtype=str,
        keep_default_na=False,
    )
    registered = pd.read_csv(
        DOMAIN_DIR / "registered_query_domain_assignments.csv",
        dtype=str,
        keep_default_na=False,
    )
    assignments = pd.concat([business, registered], ignore_index=True)
    assignments = assignments.drop_duplicates(["paper_id", "domain_id"])
    return frame, assignments


def build_stated_horizontal() -> pd.DataFrame:
    frame, assignments = load_primary_and_assignments()
    group_definitions = (
        assignments[["domain_id", "domain_label", "assignment_basis"]]
        .drop_duplicates("domain_id")
        .to_dict("records")
    )
    combined_ids = set(
        assignments.loc[
            assignments["domain_id"].isin(
                ["core_entrepreneurship", "other_entrepreneurship"]
            ),
            "paper_id",
        ]
    )
    rows: list[dict] = []
    for definition in EVIDENCE_DIMENSIONS:
        dimension_id = str(definition["id"])
        column = dimension_column(frame, dimension_id)
        evidence = evidence_column(dimension_id)
        stated_mask = frame[evidence].astype(str).str.strip().eq("stated")
        substantive_mask = observed_mask(frame, dimension_id)
        baseline = frame.loc[stated_mask & substantive_mask].copy()
        baseline_counts = baseline[column].astype(str).str.strip().value_counts()
        baseline_n = int(len(baseline))

        groups = list(group_definitions) + [
            {
                "domain_id": "combined_entrepreneurship",
                "domain_label": "Combined entrepreneurship",
                "assignment_basis": "union of leading and additional entrepreneurship",
            }
        ]
        for group in groups:
            if group["domain_id"] == "combined_entrepreneurship":
                paper_ids = combined_ids
            else:
                paper_ids = set(
                    assignments.loc[
                        assignments["domain_id"].eq(group["domain_id"]), "paper_id"
                    ]
                )
            selected = baseline.loc[baseline["paper_id"].isin(paper_ids)]
            counts = selected[column].astype(str).str.strip().value_counts()
            denominator = int(len(selected))
            categories = set(baseline_counts.index) | set(counts.index)
            for category in sorted(categories):
                papers = int(counts.get(category, 0))
                baseline_papers = int(baseline_counts.get(category, 0))
                share = papers / denominator if denominator else 0.0
                baseline_share = baseline_papers / baseline_n if baseline_n else 0.0
                rows.append(
                    {
                        "evidence_restriction": "stated substantive evidence only",
                        "baseline_denominator": baseline_n,
                        "domain_id": group["domain_id"],
                        "domain_label": group["domain_label"],
                        "assignment_basis": group["assignment_basis"],
                        "dimension_id": dimension_id,
                        "dimension_label": definition["label"],
                        "denominator": denominator,
                        "category": category,
                        "papers": papers,
                        "share": round(share, 6),
                        "baseline_category_papers": baseline_papers,
                        "baseline_share": round(baseline_share, 6),
                        "percentage_point_difference": round(
                            (share - baseline_share) * 100, 4
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def build_table8_sensitivity(stated: pd.DataFrame) -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_HORIZONTAL)
    rows: list[dict] = []
    for domain, dimension_id, category in SELECTED_TABLE8_ROWS:
        original_group = original[
            original["domain_label"].eq(domain)
            & original["dimension_id"].eq(dimension_id)
            & original["distribution"].eq("observed")
        ].copy()
        stated_group = stated[
            stated["domain_label"].eq(domain)
            & stated["dimension_id"].eq(dimension_id)
        ].copy()
        if original_group.empty or stated_group.empty:
            raise RuntimeError(f"Missing sensitivity group: {domain} / {dimension_id}")
        original_row = original_group.loc[original_group["category"].eq(category)]
        stated_row = stated_group.loc[stated_group["category"].eq(category)]
        if len(original_row) != 1 or len(stated_row) != 1:
            raise RuntimeError(
                f"Expected one category row: {domain} / {dimension_id} / {category}"
            )
        original_row = original_row.iloc[0]
        stated_row = stated_row.iloc[0]
        original_leader = str(
            original_group.sort_values(["papers", "category"], ascending=[False, True])
            .iloc[0]["category"]
        )
        stated_leader = str(
            stated_group.sort_values(["papers", "category"], ascending=[False, True])
            .iloc[0]["category"]
        )
        original_difference = float(original_row["percentage_point_difference"])
        stated_difference = float(stated_row["percentage_point_difference"])
        rows.append(
            {
                "domain": domain,
                "dimension_id": dimension_id,
                "dimension": original_row["dimension_label"],
                "selected_category": category,
                "original_domain_denominator": int(original_row["denominator"]),
                "stated_domain_denominator": int(stated_row["denominator"]),
                "original_baseline_denominator": int(original_row["baseline_denominator"]),
                "stated_baseline_denominator": int(stated_row["baseline_denominator"]),
                "original_domain_share": float(original_row["share"]),
                "stated_domain_share": float(stated_row["share"]),
                "original_baseline_share": float(original_row["baseline_share"]),
                "stated_baseline_share": float(stated_row["baseline_share"]),
                "original_difference_pp": original_difference,
                "stated_difference_pp": stated_difference,
                "direction_retained": _sign(original_difference)
                == _sign(stated_difference),
                "original_leading_category": original_leader,
                "stated_leading_category": stated_leader,
                "leading_category_retained": original_leader == stated_leader,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence_type_distribution()
    stated = build_stated_horizontal()
    selected = build_table8_sensitivity(stated)
    outputs = {
        "evidence_type_distribution_by_model_population.csv": evidence,
        "horizontal_domain_contrast_stated_evidence.csv": stated,
        "table8_stated_evidence_sensitivity.csv": selected,
    }
    manifest_outputs = {}
    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        manifest_outputs[filename] = {
            "rows": int(len(frame)),
            "sha256": _sha256(path),
        }
        print(f"Wrote {path}: {len(frame):,} rows")
    summary = {
        "method": (
            "Existing evidence_type fields; stated substantive codes only; no recoding"
        ),
        "selected_table8_rows": int(len(selected)),
        "directions_retained": int(selected["direction_retained"].sum()),
        "leading_categories_retained": int(
            selected["leading_category_retained"].sum()
        ),
        "outputs": manifest_outputs,
    }
    summary_path = OUTPUT_DIR / "evidence_type_sensitivity_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
