"""Select evidence papers for retained role-mechanism relations.

The relation itself is retained by the theory-elaboration analysis. This
script makes the choice of one paper *within* each retained relation
deterministic and auditable.

Eligibility is restricted to the documented Combined-entrepreneurship
close-reading base. A candidate must:

1. receive the retained role-mechanism relation from the primary coder;
2. have stated primary evidence for both role and mechanism;
3. contain non-empty primary mechanism logic;
4. pass primary-record quality control.

Eligible candidates are ranked lexicographically by:

1. exact relation agreement among Mini, Claude, and Gemini (descending);
2. agreeing models that also mark both evidence fields stated (descending);
3. VOSviewer total-link-strength percentile rank within the relevant
   entrepreneurship population (ascending);
4. minimum primary confidence across role and mechanism (descending); and
5. stable paper identifier (ascending).

The complete candidate audit is written so that the selected paper can always
be reconstructed from the recorded inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
READING_AUDIT = (
    ROOT
    / "reports/analysis/tables/contrasting/"
    "close_reading_current_population_audit.csv"
)
OUTPUT_DIR = ROOT / "reports/analysis/tables/contrasting"
OUTPUT_CSV = OUTPUT_DIR / "objective_relation_evidence_selection.csv"
OUTPUT_JSON = OUTPUT_DIR / "objective_relation_evidence_selection_manifest.json"

MODEL_FILES = {
    "GPT-5.4 Mini": (
        ROOT
        / "data/processed/specification/"
        "paper_specifications_gpt-5.4-mini-2026-03-17_spec-v3.csv"
    ),
    "Claude Sonnet 5": (
        ROOT
        / "data/processed/specification/"
        "paper_specifications_claude-sonnet-5_spec-v3.csv"
    ),
    "Gemini 3.1 Pro Preview": (
        ROOT
        / "data/processed/specification/"
        "paper_specifications_gemini-3.1-pro-preview_spec-v3.csv"
    ),
}

RETAINED_RELATIONS = (
    ("AI as tool", "improves prediction"),
    ("AI as firm capability", "supports learning"),
    ("AI as research method", "improves prediction"),
    ("AI as tool", "alters judgment"),
    ("AI as context", "transforms stakeholder interaction"),
    ("AI as tool", "reduces uncertainty"),
)

def truthy(series: pd.Series) -> pd.Series:
    """Return a boolean mask for the project's accepted true values."""

    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "x"}
    )


def load_model(path: Path, label: str) -> pd.DataFrame:
    """Load and prefix the fields required for exact relation agreement."""

    columns = [
        "paper_id",
        "ai_role_function",
        "ai_role_function_evidence_type",
        "ai_role_function_confidence",
        "ai_mechanism_analysis",
        "ai_mechanism_evidence_type",
        "ai_mechanism_confidence",
    ]
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)
    prefix = label.lower().replace(" ", "_").replace("-", "_").replace(".", "")
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in columns[1:]}
    )


def main() -> None:
    primary = pd.read_csv(PRIMARY, dtype=str, keep_default_na=False)
    reading = pd.read_csv(READING_AUDIT, dtype=str, keep_default_na=False)
    reading = reading[truthy(reading["in_combined_entrepreneurship"])].copy()

    reading_columns = [
        "paper_id",
        "Title",
        "Source title",
        "Year",
        "DOI",
        "Link",
        "current_topic_population",
        "current_topic_id",
        "current_topic_label",
        "vos_population",
        "vos_cluster",
        "vos_total_link_strength",
        "vos_tls_rank",
        "vos_population_nodes",
    ]
    base = reading[reading_columns].merge(
        primary[
            [
                "paper_id",
                "in_query_3",
                "in_query_4",
                "ai_role_function",
                "ai_role_function_evidence",
                "ai_role_function_evidence_type",
                "ai_role_function_confidence",
                "ai_mechanism_analysis",
                "ai_mechanism_evidence",
                "ai_mechanism_evidence_type",
                "ai_mechanism_confidence",
                "ai_mechanism_logic",
                "qc_validation_passed",
            ]
        ],
        on="paper_id",
        how="inner",
        validate="one_to_one",
    )

    model_prefixes: dict[str, str] = {}
    for label, path in MODEL_FILES.items():
        model = load_model(path, label)
        base = base.merge(model, on="paper_id", how="left", validate="one_to_one")
        model_prefixes[label] = (
            label.lower().replace(" ", "_").replace("-", "_").replace(".", "")
        )

    combined_mask = truthy(primary["in_query_3"]) | truthy(primary["in_query_4"])
    combined = primary[combined_mask].copy()
    audit_rows: list[dict[str, object]] = []
    selected_ids: dict[str, str] = {}

    for role, mechanism in RETAINED_RELATIONS:
        relation = f"{role} × {mechanism}"
        relation_population = combined[
            combined["ai_role_function"].eq(role)
            & combined["ai_mechanism_analysis"].eq(mechanism)
        ].copy()
        support_n = len(relation_population)
        leading_n = int(truthy(relation_population["in_query_3"]).sum())
        additional_n = int(truthy(relation_population["in_query_4"]).sum())

        candidates = base[
            base["ai_role_function"].eq(role)
            & base["ai_mechanism_analysis"].eq(mechanism)
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"No close-reading candidates for {relation}")

        candidates["primary_min_confidence"] = candidates[
            ["ai_role_function_confidence", "ai_mechanism_confidence"]
        ].apply(pd.to_numeric, errors="coerce").min(axis=1)
        candidates["primary_both_evidence_stated"] = (
            candidates["ai_role_function_evidence_type"].eq("stated")
            & candidates["ai_mechanism_evidence_type"].eq("stated")
        )
        candidates["primary_mechanism_logic_present"] = (
            candidates["ai_mechanism_logic"].astype(str).str.strip().ne("")
        )
        candidates["primary_qc_passed"] = truthy(
            candidates["qc_validation_passed"]
        )
        candidates["eligible"] = (
            candidates["primary_both_evidence_stated"]
            & candidates["primary_mechanism_logic_present"]
            & candidates["primary_qc_passed"]
        )

        match_columns: list[str] = []
        stated_match_columns: list[str] = []
        for label, prefix in model_prefixes.items():
            match_column = f"{prefix}_exact_relation_match"
            stated_match_column = f"{prefix}_stated_exact_relation_match"
            candidates[match_column] = (
                candidates[f"{prefix}_ai_role_function"].eq(role)
                & candidates[f"{prefix}_ai_mechanism_analysis"].eq(mechanism)
            )
            candidates[stated_match_column] = (
                candidates[match_column]
                & candidates[f"{prefix}_ai_role_function_evidence_type"].eq(
                    "stated"
                )
                & candidates[f"{prefix}_ai_mechanism_evidence_type"].eq(
                    "stated"
                )
            )
            match_columns.append(match_column)
            stated_match_columns.append(stated_match_column)

        candidates["exact_relation_agreement_models"] = candidates[
            match_columns
        ].sum(axis=1)
        candidates["stated_exact_relation_agreement_models"] = candidates[
            stated_match_columns
        ].sum(axis=1)
        candidates["agreeing_model_names"] = candidates.apply(
            lambda row: "; ".join(
                label
                for label, prefix in model_prefixes.items()
                if bool(row[f"{prefix}_exact_relation_match"])
            ),
            axis=1,
        )
        candidates["vos_tls_rank_numeric"] = pd.to_numeric(
            candidates["vos_tls_rank"], errors="coerce"
        ).fillna(float("inf"))
        candidates["vos_population_nodes_numeric"] = pd.to_numeric(
            candidates["vos_population_nodes"], errors="coerce"
        )
        candidates["vos_tls_rank_fraction"] = (
            candidates["vos_tls_rank_numeric"]
            / candidates["vos_population_nodes_numeric"]
        ).fillna(float("inf"))

        eligible = candidates[candidates["eligible"]].copy()
        if eligible.empty:
            raise RuntimeError(
                f"No candidate passes the objective evidence rule for {relation}"
            )
        eligible = eligible.sort_values(
            [
                "exact_relation_agreement_models",
                "stated_exact_relation_agreement_models",
                "vos_tls_rank_fraction",
                "primary_min_confidence",
                "paper_id",
            ],
            ascending=[False, False, True, False, True],
            kind="mergesort",
        )
        selected_id = str(eligible.iloc[0]["paper_id"])
        selected_ids[relation] = selected_id

        candidates["selected"] = candidates["paper_id"].eq(selected_id)
        candidates = candidates.sort_values(
            [
                "eligible",
                "exact_relation_agreement_models",
                "stated_exact_relation_agreement_models",
                "vos_tls_rank_fraction",
                "primary_min_confidence",
                "paper_id",
            ],
            ascending=[False, False, False, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        candidates["selection_rank"] = range(1, len(candidates) + 1)

        for row in candidates.to_dict(orient="records"):
            audit_rows.append(
                {
                    "relation": relation,
                    "role": role,
                    "mechanism": mechanism,
                    "relation_support_combined": support_n,
                    "relation_support_leading": leading_n,
                    "relation_support_additional": additional_n,
                    "selection_rank": row["selection_rank"],
                    "selected": row["selected"],
                    "eligible": row["eligible"],
                    "paper_id": row["paper_id"],
                    "title": row["Title"],
                    "source_title": row["Source title"],
                    "year": row["Year"],
                    "doi": row["DOI"],
                    "link": row["Link"],
                    "entrepreneurship_population": row[
                        "current_topic_population"
                    ],
                    "topic_id": row["current_topic_id"],
                    "topic_label": row["current_topic_label"],
                    "vos_cluster": row["vos_cluster"],
                    "vos_total_link_strength": row[
                        "vos_total_link_strength"
                    ],
                    "vos_tls_rank": row["vos_tls_rank"],
                    "vos_population_nodes": row["vos_population_nodes"],
                    "vos_tls_rank_fraction": row[
                        "vos_tls_rank_fraction"
                    ],
                    "primary_role_evidence": row[
                        "ai_role_function_evidence"
                    ],
                    "primary_mechanism_evidence": row[
                        "ai_mechanism_evidence"
                    ],
                    "primary_mechanism_logic": row["ai_mechanism_logic"],
                    "primary_role_evidence_type": row[
                        "ai_role_function_evidence_type"
                    ],
                    "primary_mechanism_evidence_type": row[
                        "ai_mechanism_evidence_type"
                    ],
                    "primary_role_confidence": row[
                        "ai_role_function_confidence"
                    ],
                    "primary_mechanism_confidence": row[
                        "ai_mechanism_confidence"
                    ],
                    "primary_min_confidence": row[
                        "primary_min_confidence"
                    ],
                    "primary_qc_passed": row["primary_qc_passed"],
                    "exact_relation_agreement_models": row[
                        "exact_relation_agreement_models"
                    ],
                    "stated_exact_relation_agreement_models": row[
                        "stated_exact_relation_agreement_models"
                    ],
                    "agreeing_model_names": row["agreeing_model_names"],
                }
            )

    output = pd.DataFrame(audit_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_CSV, index=False)

    manifest = {
        "purpose": (
            "Deterministic evidence-paper selection within retained "
            "role-mechanism relations"
        ),
        "close_reading_population": "Combined entrepreneurship",
        "close_reading_papers": int(len(reading)),
        "primary_coder": "GPT-5.4 Mini",
        "agreement_models": list(MODEL_FILES),
        "eligibility": [
            "Paper belongs to the documented 124-paper Combined-entrepreneurship close-reading base.",
            "GPT-5.4 Mini assigns the retained role and mechanism.",
            "GPT-5.4 Mini marks both role and mechanism evidence as stated.",
            "GPT-5.4 Mini supplies non-empty mechanism logic.",
            "The primary analytical record passes quality control.",
        ],
        "ranking": [
            "Exact relation agreement count among Mini, Claude, and Gemini, descending.",
            "Agreeing models with stated evidence for both fields, descending.",
            "VOSviewer total-link-strength percentile rank within the relevant entrepreneurship population, ascending.",
            "Minimum primary confidence across role and mechanism, descending.",
            "Stable paper identifier, ascending.",
        ],
        "selected_papers": selected_ids,
        "candidate_audit": str(OUTPUT_CSV.relative_to(ROOT)),
    }
    OUTPUT_JSON.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    selected = output[truthy(output["selected"])][
        [
            "relation",
            "relation_support_combined",
            "relation_support_leading",
            "relation_support_additional",
            "paper_id",
            "title",
            "exact_relation_agreement_models",
            "agreeing_model_names",
            "vos_tls_rank",
            "vos_population_nodes",
        ]
    ]
    print(selected.to_string(index=False))
    print(f"\nWrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
