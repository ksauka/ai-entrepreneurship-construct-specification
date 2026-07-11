"""Build a reproducible GPT-5.4 mini challenge set from nano results.

Inputs: the spec-v3 GPT-4.1 nano cache, its failure log, and master corpus.
Output: a 50-paper CSV manifest with a documented selection reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aecsp.specification.llm_coder import (  # noqa: E402
    PROTOCOL_ID,
    SPECIFICATION_DIMENSIONS,
)

NANO_MODEL = "gpt-4.1-nano-2025-04-14"
NANO_CACHE = (
    PROJECT_ROOT / "data" / "interim" / "spec_cache" / PROTOCOL_ID / NANO_MODEL
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "interim" / "spec_pilots" / "gpt54mini_challenge_50.csv"
)
SEED = 20260711


def _cache_records() -> pd.DataFrame:
    records = []
    for path in sorted(NANO_CACHE.glob("*.json")):
        if path.name == "protocol_manifest.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        confidence = [
            record.get(f"{dimension.column}_confidence")
            for dimension in SPECIFICATION_DIMENSIONS
        ]
        confidence = [float(value) for value in confidence if value is not None]
        records.append(
            {
                "paper_id": record["paper_id"],
                "mean_confidence": sum(confidence) / len(confidence),
                "needs_full_text_count": len(
                    [value for value in record.get("needs_full_text", "").split(";") if value]
                ),
            }
        )
    return pd.DataFrame(records)


def _content_failures() -> list[str]:
    path = NANO_CACHE / "failures.jsonl"
    if not path.exists():
        return []
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if "Structured response exceeded" in event.get("error", ""):
            failures.append(event["paper_id"])
    return list(dict.fromkeys(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.size < 10:
        parser.error("--size must be at least 10")

    corpus = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "master_corpus.csv",
        dtype=str,
        keep_default_na=False,
    )
    results = _cache_records()
    if results.empty:
        parser.error(f"No nano cache records found under {NANO_CACHE}")

    chosen: dict[str, str] = {}

    def add(paper_ids, reason: str, maximum: int) -> None:
        added = 0
        for paper_id in paper_ids:
            if paper_id in chosen:
                continue
            chosen[paper_id] = reason
            added += 1
            if added == maximum or len(chosen) == args.size:
                break

    failures = _content_failures()
    add(failures, "nano_output_limit_failure", min(10, args.size))
    add(
        results.sort_values(["mean_confidence", "paper_id"])["paper_id"],
        "nano_lowest_mean_confidence",
        min(20, args.size - len(chosen)),
    )
    add(
        results.sort_values(
            ["needs_full_text_count", "paper_id"], ascending=[False, True]
        )["paper_id"],
        "nano_high_full_text_uncertainty",
        min(10, args.size - len(chosen)),
    )
    remaining = [pid for pid in results["paper_id"] if pid not in chosen]
    random.Random(SEED).shuffle(remaining)
    add(remaining, "seeded_completed_control", args.size - len(chosen))

    manifest = pd.DataFrame(
        [{"paper_id": paper_id, "selection_reason": reason} for paper_id, reason in chosen.items()]
    )
    manifest = manifest.merge(
        corpus[["paper_id", "Title", "Year", "Source title"]],
        on="paper_id",
        how="left",
        validate="one_to_one",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False, encoding="utf-8-sig")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Prepared {len(manifest)} papers: {args.output}")
    print(manifest["selection_reason"].value_counts().to_string())
    print(f"SHA256: {digest}")


if __name__ == "__main__":
    main()
