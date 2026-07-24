"""Freeze the full-coverage four-model IRR and consensus research artifacts.

The platform calculates model reliability live from exact paper-ID joins. This
script writes the same calculation to stable CSV and JSON artifacts for the
manuscript and supplementary appendix. It does not alter any model output.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aecsp.api.graph_service import (  # noqa: E402
    GraphService,
    IRR_UNOBSERVED_VALUES,
)
from aecsp.specification.paths import specification_csv_path  # noqa: E402


OUTPUT_DIR = ROOT / "reports/analysis/tables/model_validation"
COVERAGE = OUTPUT_DIR / "full_corpus_model_coverage.csv"
PAIR_SUMMARY = OUTPUT_DIR / "full_corpus_pairwise_irr_core_summary.csv"
PAIR_DIMENSIONS = OUTPUT_DIR / "full_corpus_pairwise_irr_dimensions.csv"
CONSENSUS = OUTPUT_DIR / "full_corpus_dimension_consensus.csv"
MANIFEST = OUTPUT_DIR / "full_corpus_model_irr_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> None:
    service = GraphService()
    result = service.composition_irr_matrix("full_corpus")
    models = result["models"]
    model_ids = [item["id"] for item in models]
    labels = {item["id"]: item["label"] for item in models}

    coverage_rows = [
        {
            "model": item["id"],
            "model_label": item["label"],
            "study_role": item["role"],
            "corpus_papers": item["corpus_papers"],
            "successful_corpus_papers": item["coded_papers"],
            "missing_corpus_papers": item["missing_papers"],
            "coverage_share": item["coverage_share"],
            "comparison_reference_model": result["reference_model"],
            "comparison_reference_papers": result["reference_cohort_papers"],
            "balanced_common_papers": result["balanced_common_papers"],
        }
        for item in models
    ]
    pair_rows = []
    dimension_rows = []
    for pair in result["pairs"]:
        pair_label = f"{pair['left_label']} / {pair['right_label']}"
        pair_rows.append(
            {
                "left_model": pair["left_model"],
                "left_label": pair["left_label"],
                "right_model": pair["right_model"],
                "right_label": pair["right_label"],
                "model_pair": pair_label,
                "balanced_common_papers": pair["balanced_common_papers"],
                "core_dimensions": result["core_dimension_count"],
                "mean_exact_agreement": pair["mean_percent_agreement"],
                "mean_krippendorff_alpha": pair["mean_krippendorff_alpha"],
                "summary_method": result["summary_method"],
            }
        )
        for dimension in pair["dimensions"]:
            dimension_rows.append(
                {
                    "left_model": pair["left_model"],
                    "left_label": pair["left_label"],
                    "right_model": pair["right_model"],
                    "right_label": pair["right_label"],
                    "model_pair": pair_label,
                    "balanced_common_papers": pair["balanced_common_papers"],
                    "dimension": dimension["column"],
                    "dimension_label": dimension["label"],
                    "classification": dimension["classification"],
                    "comparable_papers": dimension["comparable_papers"],
                    "agreements": dimension["agreements"],
                    "disagreements": dimension["disagreements"],
                    "exact_agreement": dimension["percent_agreement"],
                    "krippendorff_alpha": dimension["krippendorff_alpha"],
                    "observability_comparable_papers": dimension[
                        "observability_comparable_papers"
                    ],
                    "observability_agreements": dimension[
                        "observability_agreements"
                    ],
                    "observability_exact_agreement": dimension[
                        "observability_percent_agreement"
                    ],
                    "observability_krippendorff_alpha": dimension[
                        "observability_krippendorff_alpha"
                    ],
                    "jointly_observed_papers": dimension["jointly_observed_papers"],
                    "observed_category_agreements": dimension[
                        "observed_category_agreements"
                    ],
                    "observed_category_exact_agreement": dimension[
                        "observed_category_percent_agreement"
                    ],
                    "observed_category_krippendorff_alpha": dimension[
                        "observed_category_krippendorff_alpha"
                    ],
                }
            )

    exports = {
        model_id: service.composition_export(
            "full_corpus", model_id, study_status="all"
        ).set_index("paper_id", drop=False)
        for model_id in model_ids
    }
    common_ids = sorted(set.intersection(*[set(frame.index) for frame in exports.values()]))
    if len(common_ids) != result["balanced_common_papers"]:
        raise RuntimeError(
            "Frozen consensus intersection does not match the platform IRR intersection"
        )
    preferred_ids = [
        str(item)
        for item in service._comparison_config.get("preferred_agreement_models", [])
    ]
    if not preferred_ids or not set(preferred_ids).issubset(exports):
        raise RuntimeError("Preferred agreement models are unavailable")

    consensus_rows = []
    for dimension in result["dimensions"]:
        column = dimension["column"]
        preferred_values = pd.concat(
            [
                exports[model_id]
                .loc[common_ids, column]
                .fillna("")
                .astype(str)
                .str.strip()
                for model_id in preferred_ids
            ],
            axis=1,
        )
        preferred_values.columns = preferred_ids
        all_values = pd.concat(
            [
                exports[model_id]
                .loc[common_ids, column]
                .fillna("")
                .astype(str)
                .str.strip()
                for model_id in model_ids
            ],
            axis=1,
        )
        all_values.columns = model_ids
        preferred_match = preferred_values.nunique(axis=1, dropna=False).eq(1)
        all_match = all_values.nunique(axis=1, dropna=False).eq(1)
        excluded = IRR_UNOBSERVED_VALUES.get(column, frozenset())
        preferred_unobserved = preferred_match & preferred_values.iloc[:, 0].isin(
            excluded
        )
        all_unobserved = all_match & all_values.iloc[:, 0].isin(excluded)
        preferred_observed = preferred_match & ~preferred_values.iloc[:, 0].isin(
            excluded
        )
        all_observed = all_match & ~all_values.iloc[:, 0].isin(excluded)
        consensus_rows.append(
            {
                "dimension": column,
                "dimension_label": dimension["label"],
                "classification": dimension["classification"],
                "balanced_common_papers": len(common_ids),
                "preferred_models": " | ".join(labels[item] for item in preferred_ids),
                "preferred_trio_agreement_papers": int(preferred_match.sum()),
                "preferred_trio_agreement_share": float(preferred_match.mean()),
                "preferred_trio_unobserved_agreement_papers": int(
                    preferred_unobserved.sum()
                ),
                "preferred_trio_observed_agreement_papers": int(
                    preferred_observed.sum()
                ),
                "all_four_agreement_papers": int(all_match.sum()),
                "all_four_agreement_share": float(all_match.mean()),
                "all_four_unobserved_agreement_papers": int(all_unobserved.sum()),
                "all_four_observed_agreement_papers": int(all_observed.sum()),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(COVERAGE, index=False)
    pd.DataFrame(pair_rows).to_csv(PAIR_SUMMARY, index=False)
    pd.DataFrame(dimension_rows).to_csv(PAIR_DIMENSIONS, index=False)
    pd.DataFrame(consensus_rows).to_csv(CONSENSUS, index=False)

    input_paths = {
        model_id: specification_csv_path(service.processed_dir, model=model_id)
        for model_id in model_ids
    }
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "full_corpus",
        "corpus_papers": len(service.papers),
        "models": model_ids,
        "comparison_reference_model": result["reference_model"],
        "comparison_reference_papers": result["reference_cohort_papers"],
        "balanced_common_papers": result["balanced_common_papers"],
        "comparison_cohort": result["comparison_cohort"],
        "cohort_rule": result["cohort_rule"],
        "irr_rule": result["irr_rule"],
        "dimension_count": result["dimension_count"],
        "core_dimension_count": result["core_dimension_count"],
        "summary_method": result["summary_method"],
        "preferred_agreement_models": preferred_ids,
        "preferred_agreement_label": service._comparison_config.get(
            "preferred_agreement_label", ""
        ),
        "interpretive_boundary": (
            "Agreement is exact coding convergence on the same paper and dimension; "
            "it is not accuracy or ground truth."
        ),
        "agreement_layers": {
            "full_category": (
                "All balanced papers; unobserved values remain categories."
            ),
            "observability": (
                "All balanced papers collapsed to observed versus unobserved."
            ),
            "observed_category": (
                "Only papers where both models assigned an observed category."
            ),
        },
        "inputs": {
            model_id: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for model_id, path in input_paths.items()
        },
        "outputs": {},
    }
    for path in (COVERAGE, PAIR_SUMMARY, PAIR_DIMENSIONS, CONSENSUS):
        manifest["outputs"][path.name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(pd.read_csv(path)),
            "sha256": sha256(path),
        }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {COVERAGE}")
    print(f"Wrote {PAIR_SUMMARY}")
    print(f"Wrote {PAIR_DIMENSIONS}")
    print(f"Wrote {CONSENSUS}")
    print(f"Wrote {MANIFEST}")


if __name__ == "__main__":
    build()
