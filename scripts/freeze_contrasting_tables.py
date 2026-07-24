"""Freeze the construct-contrasting results into checksummed manuscript tables.

This script does not compute anything new. It drives the same GraphService
methods that back the platform's contrasting pages, then serialises their output
to tidy CSV tables plus a checksummed manifest. Because the platform and the
manuscript tables read from one code path, they cannot disagree.

Read-only with respect to all frozen inputs. It reads the primary analysis
dataset and the frozen domain assignments and writes only new files under
``reports/analysis/tables/contrasting/``.

Covered analyses (per Minika's 17 July request):
  1. Construct specification, per entrepreneurship population (top-tier / other /
     combined) and the full corpus, full and observed distributions.
  2. Horizontal contrasting across the 12 registered analytical groups, observed
     distributions with percentage-point differences from the full corpus, plus
     the FT50-only robustness replication.
  3. Vertical contrasting: the AI-role-by-level matrix and the
     mechanism-by-level matrix.
  4. Structuring: recurring role/mechanism/level/scope/process-stage
     configurations, and the four structuring pair matrices.

Run from the repo root with the ``graphrag`` environment active:

    python scripts/freeze_contrasting_tables.py

Optionally restrict the rater with ``--model``; the default is the registered
primary rater.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

from aecsp.analytics.observed_composition import OBSERVED_COMPOSITION_PANELS
from aecsp.analytics.theory_contrasting import (
    dimension_column,
    distribution as theory_distribution,
    relationship_matrix,
)
from aecsp.api.graph_service import GraphService, THEORY_POPULATIONS
from aecsp.specification.paths import resolve_primary_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports/analysis/tables/contrasting"

# Dimensions carried through the contrast tables, in reporting order. Definition
# clarity is retained for completeness but is genre-confounded and never a
# headline claim; the results narrative treats it as diagnostic only.
DIMENSION_IDS: tuple[str, ...] = tuple(panel["id"] for panel in OBSERVED_COMPOSITION_PANELS)

# Structuring pair matrices exposed by the platform.
STRUCTURING_PAIRS: tuple[str, ...] = (
    "ai_role__mechanism",
    "ai_role__level",
    "mechanism__level",
    "ai_role__scope",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(frame: pd.DataFrame, name: str, outputs: dict[str, dict]) -> None:
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False)
    outputs[name] = {"rows": int(len(frame)), "sha256": _sha256(path)}
    print(f"  wrote {name}: {len(frame)} rows")


def _distribution_rows(result: dict, **context: object) -> list[dict]:
    rows = []
    for category in result.get("categories", []):
        rows.append(
            {
                **context,
                "dimension_id": result.get("dimension_id"),
                "dimension_label": result.get("dimension_label"),
                "distribution": result.get("distribution"),
                "denominator": result.get("denominator"),
                "category": category.get("value"),
                "raw_value": category.get("raw_value"),
                "papers": category.get("papers"),
                "share": category.get("share"),
                "percentage_point_difference": category.get("percentage_point_difference"),
            }
        )
    return rows


def build_construct_specification(service: GraphService, model: str, outputs: dict) -> None:
    print("construct specification (per population)...")
    rows: list[dict] = []
    populations = [pid for pid, _ in THEORY_POPULATIONS]
    for population in populations:
        for view in ("full", "observed"):
            result = service.theory_construct_specification(
                model, population, journal_scope="all", study_status="all"
            )
            label = result.get("population_label")
            for panel in result.get("panels", []):
                categories = (
                    panel.get("full_categories", [])
                    if view == "full"
                    else panel.get("categories", [])
                )
                denominator = (
                    panel.get("full_n") if view == "full" else panel.get("observed_n")
                )
                for category in categories:
                    rows.append(
                        {
                            "population": population,
                            "population_label": label,
                            "dimension_id": panel.get("id"),
                            "dimension_label": panel.get("label"),
                            "distribution": view,
                            "denominator": denominator,
                            "category": category.get("value"),
                            "raw_value": category.get("raw_value", category.get("value")),
                            "papers": category.get("count"),
                            "share": category.get("share"),
                        }
                    )
    _write(pd.DataFrame(rows), "construct_specification_by_population.csv", outputs)


def build_horizontal(service: GraphService, model: str, outputs: dict) -> None:
    for journal_scope, suffix in (("all", "full_corpus"), ("ft50", "ft50")):
        print(f"horizontal contrast (baseline={suffix})...")
        rows: list[dict] = []
        for dimension_id in DIMENSION_IDS:
            result = service.theory_horizontal_contrast(
                model,
                dimension_id,
                distribution_view="observed",
                journal_scope=journal_scope,
                study_status="all",
            )
            baseline_label = result.get("baseline_label")
            baseline = result.get("baseline", {})
            baseline_by_value = {
                str(item.get("raw_value", "")): item
                for item in baseline.get("categories", [])
            }
            coverage = result.get("baseline_domain_coverage", {})
            for group in result.get("groups", []):
                if not group.get("eligible", True):
                    continue
                group_rows = _distribution_rows(
                    group,
                    baseline_label=baseline_label,
                    baseline_denominator=baseline.get("denominator"),
                    baseline_outside_selected_domain_papers=coverage.get(
                        "outside_selected_business_domains"
                    ),
                    baseline_outside_selected_domain_percent=coverage.get(
                        "outside_selected_business_domains_percent"
                    ),
                    domain_id=group.get("id"),
                    domain_label=group.get("label"),
                    assignment_basis=group.get("assignment_basis"),
                )
                for row in group_rows:
                    baseline_category = baseline_by_value.get(
                        str(row.get("raw_value", "")), {}
                    )
                    row["baseline_category_papers"] = baseline_category.get("papers", 0)
                    row["baseline_share"] = baseline_category.get("share", 0.0)
                rows.extend(group_rows)
        _write(
            pd.DataFrame(rows),
            f"horizontal_domain_contrast_{suffix}.csv",
            outputs,
        )


def build_vertical(service: GraphService, model: str, outputs: dict) -> None:
    print("vertical contrast (dimension by level)...")
    rows: list[dict] = []
    for row_dimension in ("ai_role", "mechanism"):
        result = service.theory_vertical_contrast(
            model,
            population="combined",
            row_dimension=row_dimension,
            column_dimension="level",
            distribution_view="observed",
            journal_scope="all",
            study_status="all",
        )
        for cell in result.get("cells", []):
            rows.append(
                {
                    "row_dimension": result.get("row_dimension"),
                    "row_label": result.get("row_label"),
                    "column_dimension": result.get("column_dimension"),
                    "column_label": result.get("column_label"),
                    "analyzed_n": result.get("analyzed_n"),
                    "row_value": cell.get("row_value"),
                    "column_value": cell.get("column_value"),
                    "papers": cell.get("papers"),
                    "share_within_row": cell.get("share_within_row"),
                    "share_within_column": cell.get("share_within_column"),
                    "share_of_analyzed": cell.get("share_of_analyzed"),
                }
            )
    _write(pd.DataFrame(rows), "vertical_dimension_by_level.csv", outputs)


def build_structuring(service: GraphService, model: str, outputs: dict) -> None:
    print("structuring (recurring configurations + pair matrices)...")
    config_rows: list[dict] = []
    pair_rows: list[dict] = []
    for pair in STRUCTURING_PAIRS:
        result = service.theory_structuring(
            model,
            population="combined",
            pair_id=pair,
            distribution_view="observed",
            journal_scope="all",
            study_status="all",
            min_support=10,
        )
        matrix = result.get("matrix", {})
        for cell in matrix.get("cells", []):
            pair_rows.append(
                {
                    "pair": pair,
                    "row_dimension": matrix.get("row_dimension"),
                    "column_dimension": matrix.get("column_dimension"),
                    "row_value": cell.get("row_value"),
                    "column_value": cell.get("column_value"),
                    "papers": cell.get("papers"),
                    "share_within_row": cell.get("share_within_row"),
                }
            )
        if not config_rows:
            configurations = result.get("configurations", {})
            dims = configurations.get("dimensions", [])
            for record in configurations.get("configurations", []):
                config_rows.append(
                    {
                        "papers": record.get("papers"),
                        "share": record.get("share"),
                        **{dim: record.get(dim) for dim in dims},
                    }
                )
    _write(pd.DataFrame(config_rows), "structuring_configurations.csv", outputs)
    _write(pd.DataFrame(pair_rows), "structuring_pair_matrices.csv", outputs)


def _nested_populations(
    service: GraphService,
    model: str,
) -> dict[str, tuple[str, pd.DataFrame]]:
    """Return the four populations shared by the platform and manuscript."""

    frame = service._composition_model_frame(model)
    populations = {"full_corpus": ("Full corpus", frame)}
    for population_id, label in THEORY_POPULATIONS:
        populations[population_id] = (
            label,
            service._theory_population_frame(frame, population_id),
        )
    return populations


def build_nested_dimension_analysis(
    service: GraphService,
    model: str,
    outputs: dict,
) -> None:
    """Freeze the platform's arbitrary dimension filters and pair matrices.

    The interactive Construct Specification page lets the researcher condition
    every distribution on one exact value of any registered dimension and then
    cross any two remaining dimensions.  The manuscript previously consumed
    only the unconditioned portrait.  These tidy tables preserve the complete
    analytical capability without forcing hundreds of matrices into the paper.
    """

    print("nested construct specification (all controls and dimension pairs)...")
    nested_rows: list[dict] = []
    matrix_rows: list[dict] = []
    inventory_rows: list[dict] = []
    labels = {panel["id"]: panel["label"] for panel in OBSERVED_COMPOSITION_PANELS}

    for population_id, (population_label, frame) in _nested_populations(
        service, model
    ).items():
        for control_id in DIMENSION_IDS:
            control_column = dimension_column(frame, control_id)
            if control_column not in frame.columns:
                continue
            control_values = frame[control_column].astype(str).str.strip()
            control_counts = control_values.value_counts(dropna=False)
            for control_value, control_n in control_counts.items():
                raw_control = str(control_value)
                controlled = frame.loc[control_values.eq(raw_control)].copy()
                for outcome_id in DIMENSION_IDS:
                    if outcome_id == control_id:
                        continue
                    for distribution_view in ("full", "observed"):
                        result = theory_distribution(
                            controlled,
                            outcome_id,
                            distribution_view,
                            study_status="all",
                        )
                        for category in result.get("categories", []):
                            nested_rows.append(
                                {
                                    "population": population_id,
                                    "population_label": population_label,
                                    "population_papers": len(frame),
                                    "control_dimension": control_id,
                                    "control_dimension_label": labels[control_id],
                                    "control_column": control_column,
                                    "control_value": (
                                        "Missing value" if raw_control == "" else raw_control
                                    ),
                                    "control_raw_value": raw_control,
                                    "control_papers": int(control_n),
                                    "outcome_dimension": outcome_id,
                                    "outcome_dimension_label": result.get(
                                        "dimension_label"
                                    ),
                                    "outcome_column": result.get("column"),
                                    "distribution": distribution_view,
                                    "denominator": result.get("denominator"),
                                    "category": category.get("value"),
                                    "raw_value": category.get("raw_value"),
                                    "papers": category.get("papers"),
                                    "share": category.get("share"),
                                }
                            )

        for row_dimension, column_dimension in combinations(DIMENSION_IDS, 2):
            for distribution_view in ("full", "observed"):
                result = relationship_matrix(
                    frame,
                    row_dimension,
                    column_dimension,
                    distribution_view,
                    study_status="all",
                )
                nonzero = [
                    cell for cell in result.get("cells", []) if cell.get("papers", 0)
                ]
                strongest = max(
                    nonzero,
                    key=lambda cell: (
                        cell.get("papers", 0),
                        cell.get("share_of_analyzed", 0),
                    ),
                    default={},
                )
                inventory_rows.append(
                    {
                        "population": population_id,
                        "population_label": population_label,
                        "population_papers": len(frame),
                        "distribution": distribution_view,
                        "row_dimension": row_dimension,
                        "row_label": result.get("row_label"),
                        "column_dimension": column_dimension,
                        "column_label": result.get("column_label"),
                        "analyzed_n": result.get("analyzed_n"),
                        "nonzero_cells": len(nonzero),
                        "strongest_row_value": strongest.get("row_value"),
                        "strongest_column_value": strongest.get("column_value"),
                        "strongest_cell_papers": strongest.get("papers", 0),
                        "strongest_cell_share": strongest.get(
                            "share_of_analyzed", 0.0
                        ),
                    }
                )
                for cell in result.get("cells", []):
                    matrix_rows.append(
                        {
                            "population": population_id,
                            "population_label": population_label,
                            "population_papers": len(frame),
                            "distribution": distribution_view,
                            "row_dimension": row_dimension,
                            "row_label": result.get("row_label"),
                            "row_column": result.get("row_column"),
                            "row_value": cell.get("row_value"),
                            "row_raw_value": cell.get("row_raw_value"),
                            "column_dimension": column_dimension,
                            "column_label": result.get("column_label"),
                            "column_column": result.get("column_column"),
                            "column_value": cell.get("column_value"),
                            "column_raw_value": cell.get("column_raw_value"),
                            "analyzed_n": result.get("analyzed_n"),
                            "papers": cell.get("papers"),
                            "share_of_analyzed": cell.get("share_of_analyzed"),
                            "share_within_row": cell.get("share_within_row"),
                            "share_within_column": cell.get("share_within_column"),
                        }
                    )

    nested = pd.DataFrame(nested_rows)
    matrices = pd.DataFrame(matrix_rows)
    inventory = pd.DataFrame(inventory_rows)
    _write(nested, "nested_dimension_distributions.csv", outputs)
    _write(
        nested.loc[
            nested["control_dimension"].eq("study_status")
            & nested["control_raw_value"].isin(["phenomenon", "method", "both"])
        ].reset_index(drop=True),
        "study_status_conditioned_specification.csv",
        outputs,
    )
    _write(matrices, "nested_dimension_pair_matrices.csv", outputs)
    _write(inventory, "nested_dimension_pair_inventory.csv", outputs)

    release_path = OUTPUT_DIR / "nested_specification_release.zip"
    readme = (
        "Nested construct-specification release\n\n"
        "Purpose: complete frozen export of the Construct Specification page's "
        "nested dimension filters and pair matrices.\n"
        f"Model: {model}\n"
        "Denominators: Full retains all controlled papers; observed removes only "
        "the registered missing or unspecified values for the displayed dimension.\n"
        "Interpretation: descriptive composition only; cells do not establish "
        "causal or temporal relations.\n"
    )
    release_files = [
        "nested_dimension_distributions.csv",
        "study_status_conditioned_specification.csv",
        "nested_dimension_pair_matrices.csv",
        "nested_dimension_pair_inventory.csv",
    ]
    with zipfile.ZipFile(release_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.txt", readme)
        for name in release_files:
            archive.write(OUTPUT_DIR / name, arcname=name)
    outputs[release_path.name] = {
        "rows": int(len(nested) + len(matrices) + len(inventory)),
        "sha256": _sha256(release_path),
    }
    print(f"  wrote {release_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Rater; default is the primary.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    service = GraphService()
    model = args.model or resolve_primary_model()
    print(f"Primary rater: {model}")
    print(f"Analytical rows loaded: {len(service.papers)}")

    outputs: dict[str, dict] = {}
    build_construct_specification(service, model, outputs)
    build_horizontal(service, model, outputs)
    build_vertical(service, model, outputs)
    build_structuring(service, model, outputs)
    build_nested_dimension_analysis(service, model, outputs)

    primary_dataset = PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source_dataset": {
            "path": str(primary_dataset.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(primary_dataset) if primary_dataset.exists() else None,
        },
        "dimensions": list(DIMENSION_IDS),
        "structuring_pairs": list(STRUCTURING_PAIRS),
        "outputs": outputs,
        "note": (
            "Frozen from the GraphService contrasting methods that back the "
            "platform. Domain memberships overlap; domain rows must not be summed. "
            "Shares are conditional on the observed denominator disclosed per row."
        ),
    }
    manifest_path = OUTPUT_DIR / "contrasting_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
