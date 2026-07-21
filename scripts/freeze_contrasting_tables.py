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
  2. Horizontal contrasting across the 13 business domains, observed
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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aecsp.analytics.observed_composition import OBSERVED_COMPOSITION_PANELS
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
            for group in result.get("groups", []):
                if not group.get("eligible", True):
                    continue
                rows.extend(
                    _distribution_rows(
                        group,
                        baseline_label=baseline_label,
                        domain_id=group.get("id"),
                        domain_label=group.get("label"),
                        assignment_basis=group.get("assignment_basis"),
                    )
                )
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
