"""Create a blinded, stratified 50-paper human-validation workbook.

Inputs: the master corpus and fixed sample size/seed. Outputs: a private key
linking validation IDs to paper IDs and a model-blind coding CSV containing
title, abstract, keywords, and empty specification-code columns.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "human_validation"
DIMENSIONS = (
    "ai_role_function",
    "ai_type_form",
    "ai_mechanism",
    "level_of_analysis",
    "entrepreneurial_process_stage",
    "scope_conditions",
    "definition_construct_clarity",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.size < 20:
        parser.error("--size must be at least 20")

    corpus = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "master_corpus.csv",
        dtype=str,
        keep_default_na=False,
    ).copy()
    years = pd.to_numeric(corpus["Year"], errors="coerce")
    corpus["validation_era"] = pd.cut(
        years,
        bins=[-float("inf"), 2015, 2020, float("inf")],
        labels=["through_2015", "2016_2020", "2021_plus"],
    ).astype(str)
    lengths = corpus["Abstract"].str.split().str.len()
    corpus["validation_abstract_length"] = pd.qcut(
        lengths.rank(method="first"), 3, labels=["short", "medium", "long"]
    ).astype(str)
    query_columns = [f"in_query_{index}" for index in range(1, 5)]
    corpus["validation_query"] = corpus.apply(
        lambda row: next(
            (column.removeprefix("in_") for column in query_columns if row.get(column) == "1"),
            "unassigned",
        ),
        axis=1,
    )
    corpus["validation_stratum"] = (
        corpus["validation_era"]
        + "|"
        + corpus["validation_query"]
        + "|"
        + corpus["validation_abstract_length"]
    )

    shuffled = corpus.sample(frac=1, random_state=args.seed)
    first = shuffled.groupby("validation_stratum", sort=True).head(1)
    selected = first.head(args.size)
    if len(selected) < args.size:
        remainder = shuffled[~shuffled["paper_id"].isin(selected["paper_id"])]
        selected = pd.concat([selected, remainder.head(args.size - len(selected))])
    selected = selected.reset_index(drop=True)
    selected.insert(0, "validation_id", [f"HV{index:03d}" for index in range(1, len(selected) + 1)])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_columns = [
        "validation_id", "paper_id", "validation_stratum", "validation_era",
        "validation_query", "validation_abstract_length",
    ]
    selected[key_columns].to_csv(args.output_dir / "private_sample_key.csv", index=False)

    blind_columns = [
        "validation_id", "Title", "Abstract", "Author Keywords", "Source title", "Year"
    ]
    blind = selected[blind_columns].copy()
    for dimension in DIMENSIONS:
        blind[dimension] = ""
        blind[f"{dimension}_evidence"] = ""
    blind["ai_mechanism_logic"] = ""
    blind["needs_full_text"] = ""
    blind["coder_id"] = ""
    blind["coder_notes"] = ""
    blind.to_csv(args.output_dir / "blind_coding_template.csv", index=False, encoding="utf-8-sig")

    digest = hashlib.sha256((args.output_dir / "private_sample_key.csv").read_bytes()).hexdigest()
    print(f"Prepared {len(selected)} blinded validation papers in {args.output_dir}")
    print(f"Private key SHA256: {digest}")


if __name__ == "__main__":
    main()
