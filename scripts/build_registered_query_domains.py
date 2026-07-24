"""Build the FT50, Leading, and Additional entrepreneurship domains."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.asjc import file_sha256  # noqa: E402
from aecsp.corpus.business_domains import (  # noqa: E402
    REGISTERED_QUERY_DOMAIN_RULES,
    build_registered_query_domain_assignments,
    summarize_registered_query_domain_sources,
)


DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/analysis/theory_elaboration/domains"
)
EXPECTED_COUNTS = {
    "ft50": 438,
    "core_entrepreneurship": 646,
    "other_entrepreneurship": 986,
}
EXPECTED_PAIRWISE_OVERLAPS = {
    "ft50__core_entrepreneurship": 212,
    "ft50__other_entrepreneurship": 0,
    "core_entrepreneurship__other_entrepreneurship": 0,
}


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pairwise_overlaps(assignments: pd.DataFrame) -> dict[str, int]:
    members = {
        domain_id: set(group["paper_id"])
        for domain_id, group in assignments.groupby("domain_id")
    }
    ordered = [rule["domain_id"] for rule in REGISTERED_QUERY_DOMAIN_RULES]
    return {
        f"{left}__{right}": len(
            members.get(left, set()) & members.get(right, set())
        )
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    if not corpus_path.exists():
        parser.error(f"Corpus not found: {corpus_path}")

    usecols = [
        "paper_id",
        "Source title",
        *(rule["flag_column"] for rule in REGISTERED_QUERY_DOMAIN_RULES),
    ]
    corpus = pd.read_csv(
        corpus_path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
    )
    assignments = build_registered_query_domain_assignments(corpus)
    source_summary = summarize_registered_query_domain_sources(assignments)
    counts = assignments.groupby("domain_id")["paper_id"].nunique().to_dict()
    overlaps = pairwise_overlaps(assignments)
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Registered domain counts changed: {counts}")
    if overlaps != EXPECTED_PAIRWISE_OVERLAPS:
        raise RuntimeError(f"Registered domain overlaps changed: {overlaps}")

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "registered_query_domain_assignments.csv"
    sources_path = output_dir / "registered_query_domain_source_titles.csv"
    manifest_path = output_dir / "registered_query_domain_manifest.json"
    assignments.to_csv(assignments_path, index=False)
    source_summary.to_csv(sources_path, index=False)

    domains = {}
    for rule in REGISTERED_QUERY_DOMAIN_RULES:
        domain_id = rule["domain_id"]
        domains[domain_id] = {
            "display_name": rule["domain_label"],
            "basis": rule["flag_column"],
            "papers": counts[domain_id],
            "source_title_strings": int(
                source_summary.loc[
                    source_summary["domain_id"].eq(domain_id), "source_title"
                ].nunique()
            ),
        }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": relative(corpus_path),
            "sha256": file_sha256(corpus_path),
            "papers": len(corpus),
        },
        "domains": domains,
        "validation": {
            "paper_domain_rows": len(assignments),
            "unique_papers": int(assignments["paper_id"].nunique()),
            "pairwise_overlaps": overlaps,
            "counts_match_registered_corpus": counts == EXPECTED_COUNTS,
            "overlaps_match_registered_corpus": (
                overlaps == EXPECTED_PAIRWISE_OVERLAPS
            ),
        },
        "outputs": {
            "paper_assignments": relative(assignments_path),
            "paper_assignments_sha256": file_sha256(assignments_path),
            "source_title_summary": relative(sources_path),
            "source_title_summary_sha256": file_sha256(sources_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "Built registered query domains: "
        f"FT50 {counts['ft50']:,}; "
        f"Leading entrepreneurship journals {counts['core_entrepreneurship']:,}; "
        f"Additional entrepreneurship {counts['other_entrepreneurship']:,}."
    )
    print(f"Pairwise overlaps: {overlaps}")
    print(f"Assignments: {assignments_path}")
    print(f"Source-title summary: {sources_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
