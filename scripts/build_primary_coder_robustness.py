"""Freeze Mini-to-Gemini robustness checks for the five manuscript analyses."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aecsp.analytics.coder_robustness import build_coder_robustness  # noqa: E402
from aecsp.api.graph_service import GraphService  # noqa: E402


PRIMARY_MODEL = "gpt-5.4-mini-2026-03-17"
ALTERNATIVE_MODEL = "gemini-3.1-pro-preview"
OUTPUT_DIR = ROOT / "reports/analysis/tables/model_validation/coder_robustness"


def _flatten(prefix: str, value: Any, target: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}_{key}" if prefix else key, item, target)
    else:
        target[prefix] = value


def _wide_rows(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        flat: dict[str, Any] = {}
        _flatten("", record, flat)
        rows.append(flat)
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    service = GraphService()
    result = build_coder_robustness(
        service._composition_model_frame(PRIMARY_MODEL),
        service._composition_model_frame(ALTERNATIVE_MODEL),
        primary_model=PRIMARY_MODEL,
        primary_label="GPT-5.4 Mini",
        alternative_model=ALTERNATIVE_MODEL,
        alternative_label="Gemini 3.1 Pro Preview",
        min_support=20,
    )

    tables = {
        "aggregate_distributions.csv": pd.DataFrame(
            result["aggregate_distributions"]
        ),
        "aggregate_leading_categories.csv": _wide_rows(
            result["aggregate_comparison"]
        ),
        "nested_distributions_by_study_status.csv": pd.DataFrame(
            result["nested_distributions"]
        ),
        "nested_leading_categories_by_study_status.csv": _wide_rows(
            result["nested_comparison"]
        ),
        "core_additional_selected_contrasts.csv": _wide_rows(
            result["entrepreneurship_contrasts"]
        ),
        "role_by_level_cells.csv": pd.DataFrame(result["role_level_cells"]),
        "role_by_level_leading_comparison.csv": pd.DataFrame(
            result["role_level_comparison"]
        ),
        "selected_recurring_relations.csv": _wide_rows(
            result["selected_relations"]
        ),
    }
    for name, frame in tables.items():
        frame.to_csv(OUTPUT_DIR / name, index=False)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "primary_model": result["primary_model"],
                "alternative_model": result["alternative_model"],
                "population": result["population"],
                "min_support": result["min_support"],
                "summary": result["summary"],
                "interpretation": result["interpretation"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    files = [*sorted(tables), summary_path.name]
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_model": PRIMARY_MODEL,
        "alternative_model": ALTERNATIVE_MODEL,
        "population_rule": "Exact Core and Additional entrepreneurship union",
        "observed_rule": (
            "Dimension-specific missing and unspecified categories are excluded "
            "using the same registry as the platform."
        ),
        "support_threshold": result["min_support"],
        "raw_model_records_changed": False,
        "summary": result["summary"],
        "files": [
            {
                "path": name,
                "rows": int(len(tables[name])) if name in tables else None,
                "sha256": _sha256(OUTPUT_DIR / name),
            }
            for name in files
        ],
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote robustness artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
