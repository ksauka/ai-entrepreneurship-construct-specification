"""Cross-check the proprietary sample against master and raw Scopus exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.provenance_audit import AUDIT_FIELDS, audit_sample_provenance  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        default=PROJECT_ROOT / "data/interim/proprietary_validation/proprietary_probability_sample_2235_papers.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/interim/proprietary_validation/provenance_audit",
    )
    args = parser.parse_args()
    args.sample = args.sample.resolve()
    args.output_dir = args.output_dir.resolve()
    master_path = PROJECT_ROOT / "data/processed/master_corpus.csv"
    query_paths = sorted((PROJECT_ROOT / "data/queries").glob("*.csv"))
    if not query_paths:
        parser.error("no original Scopus query CSV files found")
    sample = pd.read_csv(args.sample, dtype=str, keep_default_na=False, low_memory=False)
    master = pd.read_csv(master_path, dtype=str, keep_default_na=False, low_memory=False)
    raw_frames = []
    for path in query_paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        frame["raw_source_file"] = path.name
        frame["raw_source_row"] = range(2, len(frame) + 2)
        raw_frames.append(frame)
    raw = pd.concat(raw_frames, ignore_index=True)
    audit, discrepancies = audit_sample_provenance(sample, master, raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / "paper_provenance_audit.csv"
    discrepancy_path = args.output_dir / "provenance_discrepancies.csv"
    audit.to_csv(audit_path, index=False)
    discrepancies.to_csv(discrepancy_path, index=False)
    summary = {
        "sample_path": str(args.sample.relative_to(PROJECT_ROOT)),
        "sample_sha256": digest(args.sample),
        "master_path": str(master_path.relative_to(PROJECT_ROOT)),
        "master_sha256": digest(master_path),
        "original_query_files": {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in query_paths},
        "audited_fields": list(AUDIT_FIELDS),
        "sample_n": len(sample),
        "unique_sample_ids": int(sample["paper_id"].nunique()),
        "master_found_n": int(audit["master_found"].sum()),
        "raw_eid_found_n": int(audit["raw_found"].sum()),
        "sample_master_all_exact_n": int(audit["sample_master_all_exact"].sum()),
        "master_raw_all_exact_same_row_n": int(audit["master_raw_all_exact"].sum()),
        "master_raw_all_normalized_same_row_n": int(audit["master_raw_all_normalized"].sum()),
        "field_discrepancies_n": len(discrepancies),
        "whitespace_normalizations_n": int((discrepancies.get("discrepancy_type", pd.Series(dtype=str)) == "whitespace_normalization").sum()),
        "substantive_discrepancies_n": int((discrepancies.get("discrepancy_type", pd.Series(dtype=str)) == "substantive").sum()),
        "audit_passed": bool(
            len(audit) == len(sample)
            and audit["master_found"].all()
            and audit["raw_found"].all()
            and audit["sample_master_all_exact"].all()
            and audit["master_raw_all_normalized"].all()
            and not (discrepancies.get("discrepancy_type", pd.Series(dtype=str)) == "substantive").any()
        ),
    }
    summary_path = args.output_dir / "provenance_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Paper audit: {audit_path}")
    print(f"Discrepancies: {discrepancy_path}")


if __name__ == "__main__":
    main()
