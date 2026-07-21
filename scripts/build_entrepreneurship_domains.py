"""Build the two registered entrepreneurship-domain assignment tables."""

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
    build_entrepreneurship_domain_assignments,
    summarize_entrepreneurship_domain_journals,
)


DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/analysis/theory_elaboration/domains"
)
EXPECTED_COUNTS = {
    "core_entrepreneurship": 646,
    "other_entrepreneurship": 986,
}


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
    if not corpus_path.exists():
        parser.error(f"Corpus not found: {corpus_path}")

    corpus = pd.read_csv(
        corpus_path,
        usecols=[
            "paper_id",
            "Source title",
            "in_query_3",
            "in_query_4",
        ],
        dtype=str,
        keep_default_na=False,
    )
    assignments = build_entrepreneurship_domain_assignments(corpus)
    journal_summary = summarize_entrepreneurship_domain_journals(assignments)
    counts = assignments.groupby("domain_id")["paper_id"].nunique().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Entrepreneurship-domain counts changed: {counts}; "
            f"expected {EXPECTED_COUNTS}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = output_dir / "entrepreneurship_domain_assignments.csv"
    journals_path = output_dir / "entrepreneurship_domain_journals.csv"
    manifest_path = output_dir / "entrepreneurship_domain_manifest.json"
    assignments.to_csv(assignments_path, index=False)
    journal_summary.to_csv(journals_path, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": {
            "path": relative(corpus_path),
            "sha256": file_sha256(corpus_path),
            "papers": len(corpus),
        },
        "domains": {
            "core_entrepreneurship": {
                "display_name": "Core entrepreneurship",
                "basis": "in_query_3",
                "papers": counts["core_entrepreneurship"],
                "source_title_strings": int(
                    journal_summary.loc[
                        journal_summary["domain_id"].eq("core_entrepreneurship"),
                        "source_title",
                    ].nunique()
                ),
            },
            "other_entrepreneurship": {
                "display_name": "Additional entrepreneurship",
                "basis": "in_query_4",
                "papers": counts["other_entrepreneurship"],
                "source_title_strings": int(
                    journal_summary.loc[
                        journal_summary["domain_id"].eq("other_entrepreneurship"),
                        "source_title",
                    ].nunique()
                ),
            },
        },
        "validation": {
            "paper_domain_rows": len(assignments),
            "unique_papers": int(assignments["paper_id"].nunique()),
            "cross_domain_overlap": int(
                assignments.duplicated("paper_id", keep=False).sum()
            ),
            "counts_match_registered_corpus": counts == EXPECTED_COUNTS,
        },
        "outputs": {
            "paper_assignments": relative(assignments_path),
            "paper_assignments_sha256": file_sha256(assignments_path),
            "journal_summary": relative(journals_path),
            "journal_summary_sha256": file_sha256(journals_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "Built entrepreneurship domains: "
        f"Core {counts['core_entrepreneurship']:,}; "
        f"Other {counts['other_entrepreneurship']:,}; overlap 0"
    )
    print(f"Assignments: {assignments_path}")
    print(f"Journal summary: {journals_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
