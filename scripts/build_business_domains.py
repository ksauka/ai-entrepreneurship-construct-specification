"""Recreate registered business domains from journals represented in the corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.asjc import file_sha256  # noqa: E402
from aecsp.corpus.domain_registry import (  # noqa: E402
    build_registry_domain_assignments,
)


DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/analysis/theory_elaboration/domains"
ASJC_ALIGNED_REGISTRY = (
    PROJECT_ROOT
    / "configs/legacy_taxonomies/domains_asjc_aligned_14_fields_2025-09-19.py"
)
CUSTOM_REGISTRY = (
    PROJECT_ROOT
    / "configs/legacy_taxonomies/domains_custom_10_fields_2025-09-08.py"
)
ALIASES = PROJECT_ROOT / "configs/business_domain_journal_aliases.csv"

DOMAIN_SOURCES = {
    "innovation": ("Innovation", "Technology_and_Innovation_Management", "asjc"),
    "strategy": ("Strategy", "Strategy_and_Management", "asjc"),
    "marketing": ("Marketing", "Marketing", "asjc"),
    "information_systems": (
        "Information systems",
        "Management_Information_Systems",
        "asjc",
    ),
    "finance": ("Finance", "Finance_and_Financial_Economics", "asjc"),
    "operations": ("Operations", "Supply_Chain_and_Operations", "asjc"),
    "organization_studies": (
        "Organization studies",
        "Organizational_Behavior_and_Human_Resource_Management",
        "asjc",
    ),
    "environmental_and_sustainability": (
        "Environmental and sustainability",
        "Environmental_and_Sustainability",
        "custom",
    ),
    "ethics_and_corporate_social_responsibility": (
        "Ethics and CSR",
        "Ethics_CSR_and_Governance",
        "asjc",
    ),
    "tourism": ("Tourism", "Tourism_Leisure_Hospitality", "asjc"),
}


def load_registry(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load journal registry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    asjc_registry = load_registry(ASJC_ALIGNED_REGISTRY, "etv_asjc_domains")
    custom_registry = load_registry(CUSTOM_REGISTRY, "etv_custom_domains")
    corpus = pd.read_csv(
        corpus_path,
        usecols=["paper_id", "Source title"],
        dtype=str,
        keep_default_na=False,
    )
    aliases = pd.read_csv(ALIASES, dtype=str, keep_default_na=False)
    domain_journals = {}
    for domain_id, (label, field, source) in DOMAIN_SOURCES.items():
        registry = asjc_registry if source == "asjc" else custom_registry
        domain_journals[domain_id] = {
            "label": label,
            "registry_field": field,
            "journals": registry.FIELDS[field],
        }
    assignments, source_summary = build_registry_domain_assignments(
        corpus, domain_journals, aliases
    )
    counts = assignments.groupby("domain_id")["paper_id"].nunique().to_dict()
    missing_domains = sorted(set(DOMAIN_SOURCES) - set(counts))
    if missing_domains:
        raise RuntimeError(f"Registered domains have no represented papers: {missing_domains}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "business_domain_assignments.csv"
    sources_path = output_dir / "business_domain_source_titles.csv"
    manifest_path = output_dir / "business_domain_manifest.json"
    assignments.to_csv(assignments_path, index=False)
    source_summary.to_csv(sources_path, index=False)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "construction_rule": (
            "Apply the preserved journal-domain registries only to source titles "
            "represented in the existing corpus; do not retrieve or add papers."
        ),
        "corpus": {
            "path": relative(corpus_path),
            "sha256": file_sha256(corpus_path),
            "papers": len(corpus),
        },
        "registries": {
            "asjc_aligned": {
                "path": relative(ASJC_ALIGNED_REGISTRY),
                "sha256": file_sha256(ASJC_ALIGNED_REGISTRY),
            },
            "custom_environmental": {
                "path": relative(CUSTOM_REGISTRY),
                "sha256": file_sha256(CUSTOM_REGISTRY),
            },
            "aliases": {
                "path": relative(ALIASES),
                "sha256": file_sha256(ALIASES),
                "approved_rows": int(aliases["review_status"].eq("approved").sum()),
            },
        },
        "domains": {
            domain_id: {
                "label": DOMAIN_SOURCES[domain_id][0],
                "registry_field": DOMAIN_SOURCES[domain_id][1],
                "papers": int(counts[domain_id]),
                "represented_source_titles": int(
                    source_summary["domain_id"].eq(domain_id).sum()
                ),
            }
            for domain_id in DOMAIN_SOURCES
        },
        "validation": {
            "input_papers_unchanged": True,
            "assignment_paper_ids_are_subset_of_corpus": bool(
                set(assignments["paper_id"]).issubset(set(corpus["paper_id"]))
            ),
            "paper_domain_rows": len(assignments),
            "unique_assigned_papers": int(assignments["paper_id"].nunique()),
            "multi_domain_papers": int(
                assignments.groupby("paper_id")["domain_id"].nunique().gt(1).sum()
            ),
        },
        "outputs": {
            "paper_assignments": relative(assignments_path),
            "paper_assignments_sha256": file_sha256(assignments_path),
            "represented_sources": relative(sources_path),
            "represented_sources_sha256": file_sha256(sources_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Classified existing corpus papers into {len(counts)} business domains.")
    for domain_id in DOMAIN_SOURCES:
        print(f"  {domain_id}: {counts[domain_id]:,} papers")
    print(f"Assignments: {assignments_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
