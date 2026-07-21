"""Build official Scopus ASJC assignments for every paper in the corpus."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.asjc import (  # noqa: E402
    build_source_crosswalk,
    expand_asjc_assignments,
    file_sha256,
    normalize_issn,
    read_asjc_classifications,
    read_scopus_source_list,
)


DEFAULT_SOURCE_LIST = (
    PROJECT_ROOT / "data/external/scopus/ext_list_Jun_2026.xlsx"
)
DEFAULT_CORPUS = (
    PROJECT_ROOT / "data/processed/analysis/primary_analysis_dataset.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data/processed/analysis/theory_elaboration/asjc"
)
DEFAULT_REVIEWED_OVERRIDES = (
    PROJECT_ROOT / "configs/scopus_asjc_reviewed_overrides.csv"
)
OFFICIAL_DOWNLOAD_URL = (
    "https://downloads.ctfassets.net/o78em1y1w4i4/"
    "7xtaTxNiNcWRTeZkV86eNy/"
    "710bfd3c7f7c7c9c88eeb3638ba4be43/ext_list_Jun_2026.xlsx"
)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-list", type=Path, default=DEFAULT_SOURCE_LIST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reviewed-overrides",
        type=Path,
        default=DEFAULT_REVIEWED_OVERRIDES,
    )
    args = parser.parse_args()
    source_list = args.source_list.resolve()
    corpus_path = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    reviewed_overrides_path = args.reviewed_overrides.resolve()
    if not source_list.exists():
        parser.error(f"Scopus source list not found: {source_list}")
    if not corpus_path.exists():
        parser.error(f"Corpus not found: {corpus_path}")
    if not reviewed_overrides_path.exists():
        parser.error(
            f"Reviewed ASJC overrides not found: {reviewed_overrides_path}"
        )

    print(f"Reading official source list: {source_list.name}")
    official_sources = read_scopus_source_list(source_list)
    asjc_labels, asjc_groups = read_asjc_classifications(source_list)
    corpus = pd.read_csv(
        corpus_path,
        usecols=["paper_id", "Source title", "ISSN"],
        dtype=str,
        keep_default_na=False,
    )
    if corpus["paper_id"].duplicated().any():
        parser.error("paper_id must be unique in the corpus")
    corpus["_normalized_issn"] = corpus["ISSN"].map(normalize_issn)
    reviewed_overrides = pd.read_csv(
        reviewed_overrides_path,
        dtype=str,
        keep_default_na=False,
    )

    crosswalk = build_source_crosswalk(
        corpus[["Source title", "ISSN"]],
        official_sources,
        asjc_labels,
        asjc_groups,
        reviewed_overrides,
    )
    paper_assignments = corpus.merge(
        crosswalk,
        left_on=["Source title", "_normalized_issn"],
        right_on=["source_title", "corpus_issn"],
        how="left",
        validate="many_to_one",
    )
    if len(paper_assignments) != len(corpus):
        raise RuntimeError("Paper assignment merge changed the corpus row count")
    if paper_assignments["match_status"].ne("matched").any():
        unresolved = paper_assignments.loc[
            paper_assignments["match_status"].ne("matched"),
            ["paper_id", "Source title", "ISSN"],
        ]
        raise RuntimeError(
            f"{len(unresolved):,} papers remain without an ASJC source match"
        )
    if paper_assignments["asjc_codes"].fillna("").eq("").any():
        raise RuntimeError("At least one matched paper has no ASJC code")

    paper_assignments = paper_assignments.drop(
        columns=["Source title", "ISSN", "_normalized_issn", "paper_count"]
    )
    paper_assignments_long = expand_asjc_assignments(
        paper_assignments,
        asjc_labels,
        asjc_groups,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    crosswalk_path = output_dir / "scopus_asjc_source_crosswalk.csv"
    paper_path = output_dir / "paper_asjc_assignments.csv"
    long_path = output_dir / "paper_asjc_assignments_long.csv"
    manifest_path = output_dir / "asjc_assignment_manifest.json"
    crosswalk.to_csv(crosswalk_path, index=False)
    paper_assignments.to_csv(paper_path, index=False)
    paper_assignments_long.to_csv(long_path, index=False)

    match_methods = {
        key: int(value)
        for key, value in paper_assignments["match_method"].value_counts().items()
    }
    source_match_methods = {
        key: int(value)
        for key, value in crosswalk["match_method"].value_counts().items()
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_source": {
            "publisher": "Elsevier",
            "product": "Scopus",
            "source_list_release": "June 2026",
            "source_list_path": relative(source_list),
            "source_list_sha256": file_sha256(source_list),
            "download_url": OFFICIAL_DOWNLOAD_URL,
            "classification_unit": "Scopus source",
            "classification_system": "All Science Journal Classification Codes",
        },
        "corpus": {
            "path": relative(corpus_path),
            "sha256": file_sha256(corpus_path),
            "papers": len(corpus),
            "unique_paper_ids": int(corpus["paper_id"].nunique()),
            "distinct_source_title_issn_pairs": len(crosswalk),
        },
        "matching_protocol": {
            "priority": [
                "exact normalized source title and ISSN",
                "reviewed non-exact source override",
                "review required for unique title-only or ISSN-only suggestion",
                "unresolved",
            ],
            "reviewed_overrides_path": relative(reviewed_overrides_path),
            "reviewed_overrides_sha256": file_sha256(
                reviewed_overrides_path
            ),
            "title_normalization": (
                "Unicode casefold; non-alphanumeric runs replaced by one space"
            ),
            "issn_normalization": (
                "uppercase; all characters except digits and X removed"
            ),
            "source_pair_match_methods": source_match_methods,
            "paper_match_methods": match_methods,
            "identifier_conflict_source_pairs": int(
                crosswalk["identifier_conflict"].sum()
            ),
        },
        "outputs": {
            "source_crosswalk": relative(crosswalk_path),
            "source_crosswalk_sha256": file_sha256(crosswalk_path),
            "paper_assignments": relative(paper_path),
            "paper_assignments_sha256": file_sha256(paper_path),
            "paper_assignments_long": relative(long_path),
            "paper_assignments_long_sha256": file_sha256(long_path),
        },
        "validation": {
            "all_papers_matched": bool(
                paper_assignments["match_status"].eq("matched").all()
            ),
            "all_matched_papers_have_asjc": bool(
                paper_assignments["asjc_codes"].fillna("").ne("").all()
            ),
            "paper_assignment_rows": len(paper_assignments),
            "paper_asjc_long_rows": len(paper_assignments_long),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Assigned ASJC codes to {len(paper_assignments):,}/{len(corpus):,} papers"
    )
    print(f"Paper match methods: {match_methods}")
    print(f"Source-pair match methods: {source_match_methods}")
    print(f"Source crosswalk: {crosswalk_path}")
    print(f"Paper assignments: {paper_path}")
    print(f"Long assignments: {long_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
