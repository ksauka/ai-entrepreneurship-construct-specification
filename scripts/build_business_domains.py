"""Build analytical business domains from official Scopus ASJC assignments.

The primary path aggregates the completed paper-level Scopus ASJC assignments
through an explicit, versioned code-to-domain map. A reviewed source-title
overlay is permitted only where the configuration documents that Scopus has no
direct category corresponding to the analytical domain. No papers are retrieved
or added by this script.
"""

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
import yaml  # noqa: E402

from aecsp.corpus.asjc import file_sha256  # noqa: E402
from aecsp.corpus.domain_registry import (  # noqa: E402
    build_asjc_domain_assignments,
    build_registry_domain_assignments,
)

DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DEFAULT_ASJC = (
    PROJECT_ROOT
    / "data/processed/analysis/theory_elaboration/asjc/paper_asjc_assignments_long.csv"
)
DEFAULT_QUERY_DOMAINS = (
    PROJECT_ROOT
    / "data/processed/analysis/theory_elaboration/domains/registered_query_domain_assignments.csv"
)
DEFAULT_CONFIG = PROJECT_ROOT / "configs/asjc_business_domain_aggregation.yaml"
DEFAULT_ALIASES = PROJECT_ROOT / "configs/business_domain_journal_aliases.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/analysis/theory_elaboration/domains"
)


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


def source_overlay_assignments(
    corpus: pd.DataFrame,
    domains: dict[str, dict],
    aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build only the explicitly configured reviewed source-title overlays."""

    overlay_definitions: dict[str, dict[str, object]] = {}
    for domain_id, definition in domains.items():
        if definition.get("mapping_mode") != "reviewed_source_overlay":
            continue
        registry_path = PROJECT_ROOT / str(definition["registry_path"])
        registry = load_registry(registry_path, f"overlay_{domain_id}")
        registry_field = str(definition["registry_field"])
        overlay_definitions[domain_id] = {
            "label": str(definition["label"]),
            "registry_field": registry_field,
            "journals": registry.FIELDS[registry_field],
        }
    assignments, sources = build_registry_domain_assignments(
        corpus, overlay_definitions, aliases
    )
    if not sources.empty:
        sources = sources.rename(
            columns={
                "registry_field": "overlay_registry_field",
            }
        )
        sources["mapping_mode"] = "reviewed_source_overlay"
        sources["asjc_codes"] = ""
        sources["asjc_descriptions"] = ""
        sources["assignment_basis"] = sources["overlay_registry_field"].map(
            lambda value: f"reviewed_source_overlay:{value}"
        )
        sources = sources[
            [
                "domain_id",
                "domain_label",
                "mapping_mode",
                "asjc_codes",
                "asjc_descriptions",
                "source_title",
                "papers",
                "assignment_basis",
                "overlay_registry_field",
                "registered_source_title",
                "alias_applied",
            ]
        ]
        assignments["assignment_basis"] = assignments["assignment_basis"].str.replace(
            "journal_registry:", "reviewed_source_overlay:", regex=False
        )
    return assignments, sources


def residual_outputs(
    corpus: pd.DataFrame,
    paper_asjc: pd.DataFrame,
    assigned_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a paper ledger and source-title audit for the genuine residual."""

    residual = corpus[~corpus["paper_id"].astype(str).isin(assigned_ids)][
        ["paper_id", "Source title", "ISSN"]
    ].copy()
    code_summary = (
        paper_asjc.groupby("paper_id", sort=False)
        .agg(
            asjc_codes=("asjc_code", lambda values: ";".join(sorted(set(map(str, values))))),
            asjc_descriptions=(
                "asjc_description",
                lambda values: "; ".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    residual = residual.merge(code_summary, on="paper_id", how="left")
    residual["residual_reason"] = (
        "Official ASJC source codes do not map to a selected analytical domain"
    )
    residual = residual.sort_values(["Source title", "paper_id"], kind="stable")
    sources = (
        residual.groupby(
            ["Source title", "asjc_codes", "asjc_descriptions", "residual_reason"],
            dropna=False,
        )["paper_id"]
        .nunique()
        .rename("papers")
        .reset_index()
        .sort_values(["papers", "Source title"], ascending=[False, True])
    )
    return residual.reset_index(drop=True), sources.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--asjc", type=Path, default=DEFAULT_ASJC)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--query-domains", type=Path, default=DEFAULT_QUERY_DOMAINS)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    asjc_path = args.asjc.resolve()
    config_path = args.config.resolve()
    query_path = args.query_domains.resolve()
    alias_path = args.aliases.resolve()
    output_dir = args.output_dir.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    aggregation = config["aggregation"]
    domains = config["domains"]
    corpus = pd.read_csv(
        corpus_path,
        usecols=["paper_id", "Source title", "ISSN"],
        dtype=str,
        keep_default_na=False,
    )
    paper_asjc = pd.read_csv(asjc_path, dtype=str, keep_default_na=False)
    aliases = pd.read_csv(alias_path, dtype=str, keep_default_na=False)

    official_assignments, official_sources = build_asjc_domain_assignments(
        corpus, paper_asjc, domains
    )
    overlay_assignments, overlay_sources = source_overlay_assignments(
        corpus, domains, aliases
    )
    assignments = pd.concat(
        [official_assignments, overlay_assignments], ignore_index=True
    )
    assignments = assignments.drop_duplicates(["paper_id", "domain_id"]).sort_values(
        ["domain_id", "source_title", "paper_id"], kind="stable"
    )
    sources = pd.concat(
        [official_sources, overlay_sources], ignore_index=True, sort=False
    ).fillna("")
    sources = sources.sort_values(
        ["domain_id", "papers", "source_title"],
        ascending=[True, False, True],
        kind="stable",
    )
    assigned_ids = set(assignments["paper_id"].astype(str))
    residual, residual_sources = residual_outputs(corpus, paper_asjc, assigned_ids)

    corpus_ids = set(corpus["paper_id"].astype(str))
    query_assignments = pd.read_csv(query_path, dtype=str, keep_default_na=False)
    query_ids = set(query_assignments["paper_id"].astype(str))
    query_domain_count = int(query_assignments["domain_id"].nunique())
    business_domain_count = len(domains)
    registered_group_count = business_domain_count + query_domain_count
    all_domain_ids = assigned_ids | query_ids
    multi_domain = assignments.groupby("paper_id")["domain_id"].nunique().gt(1)
    counts = assignments.groupby("domain_id")["paper_id"].nunique().to_dict()
    missing_domains = sorted(set(domains) - set(counts))
    if missing_domains:
        raise RuntimeError(f"Configured domains have no represented papers: {missing_domains}")
    if set(paper_asjc["paper_id"].astype(str)) != corpus_ids:
        raise RuntimeError("Official ASJC assignments do not cover the complete corpus")

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "business_domain_assignments.csv"
    sources_path = output_dir / "business_domain_source_titles.csv"
    residual_path = output_dir / "business_domain_residual_papers.csv"
    residual_sources_path = output_dir / "business_domain_residual_source_titles.csv"
    manifest_path = output_dir / "business_domain_manifest.json"
    assignments.to_csv(assignments_path, index=False)
    sources.to_csv(sources_path, index=False)
    residual.to_csv(residual_path, index=False)
    residual_sources.to_csv(residual_sources_path, index=False)

    domain_manifest = {}
    for domain_id, definition in domains.items():
        domain_manifest[domain_id] = {
            "label": str(definition["label"]),
            "mapping_mode": str(definition["mapping_mode"]),
            "asjc_codes": dict(definition.get("asjc_codes", {})),
            "registry_field": str(definition.get("registry_field", "")),
            "rationale": str(definition["rationale"]),
            "papers": int(counts[domain_id]),
            "represented_source_titles": int(
                sources.loc[sources["domain_id"].eq(domain_id), "source_title"].nunique()
            ),
        }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "aggregation": {
            **aggregation,
            "config_path": relative(config_path),
            "config_sha256": file_sha256(config_path),
        },
        "corpus": {
            "path": relative(corpus_path),
            "sha256": file_sha256(corpus_path),
            "papers": len(corpus),
        },
        "official_asjc": {
            "path": relative(asjc_path),
            "sha256": file_sha256(asjc_path),
            "papers_with_assignment": int(paper_asjc["paper_id"].nunique()),
            "distinct_codes": int(paper_asjc["asjc_code"].nunique()),
        },
        "domains": domain_manifest,
        "validation": {
            "input_papers_unchanged": True,
            "assignment_paper_ids_are_subset_of_corpus": bool(
                set(assignments["paper_id"]).issubset(corpus_ids)
            ),
            "all_corpus_papers_have_official_asjc": bool(
                set(paper_asjc["paper_id"].astype(str)) == corpus_ids
            ),
            "paper_domain_rows": len(assignments),
            "unique_assigned_papers": len(assigned_ids),
            "assigned_papers_percent": round(len(assigned_ids) / len(corpus) * 100, 4),
            "multi_domain_papers": int(multi_domain.sum()),
            "residual_papers": len(residual),
            "residual_papers_percent": round(len(residual) / len(corpus) * 100, 4),
            "residual_source_titles": int(residual["Source title"].nunique()),
            "business_domain_count": business_domain_count,
            "registered_query_domain_count": query_domain_count,
            "all_registered_group_count": registered_group_count,
            "all_registered_group_unique_papers": len(all_domain_ids),
            "all_registered_group_residual_papers": len(corpus_ids - all_domain_ids),
            "all_registered_group_residual_percent": round(
                len(corpus_ids - all_domain_ids) / len(corpus) * 100, 4
            ),
        },
        "outputs": {
            "paper_assignments": relative(assignments_path),
            "paper_assignments_sha256": file_sha256(assignments_path),
            "represented_sources": relative(sources_path),
            "represented_sources_sha256": file_sha256(sources_path),
            "residual_papers": relative(residual_path),
            "residual_papers_sha256": file_sha256(residual_path),
            "residual_sources": relative(residual_sources_path),
            "residual_sources_sha256": file_sha256(residual_sources_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Assigned {len(assigned_ids):,}/{len(corpus):,} papers to "
        f"{business_domain_count} domains."
    )
    for domain_id, definition in domains.items():
        print(f"  {definition['label']}: {counts[domain_id]:,} papers")
    print(
        f"Genuine {business_domain_count}-domain residual: {len(residual):,} papers "
        f"({len(residual) / len(corpus):.2%}); all papers retain official ASJC codes."
    )
    print(
        f"Residual outside all {registered_group_count} registered groups: "
        f"{len(corpus_ids - all_domain_ids):,} papers."
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
