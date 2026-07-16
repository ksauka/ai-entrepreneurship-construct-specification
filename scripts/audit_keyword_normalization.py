#!/usr/bin/env python3
"""Audit canonical keyword mappings without modifying source keyword fields.

Inputs: the primary analysis dataset and the controlled keyword alias registry.
Outputs: variant-level mappings, unresolved inflection candidates, and a JSON
summary under data/processed/analysis/keyword_normalization/.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import pandas as pd

from aecsp.analytics.keyword_trends import (
    _ORTHOGRAPHIC_TOKEN_ALIASES,
    _base_normalize,
    load_keyword_aliases,
    normalize_keyword,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "processed" / "analysis" / "primary_analysis_dataset.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "analysis" / "keyword_normalization"
)
SOURCE_COLUMNS = {
    "author": "Author Keywords",
    "index": "Index Keywords",
}


def _variants(value: object) -> set[str]:
    return {
        normalized
        for term in str(value or "").split(";")
        if (normalized := _base_normalize(term))
    }


def _possible_plurals(word: str) -> set[str]:
    """Return broad candidates for audit only; never use them as auto-merges."""

    if len(word) < 3 or word.endswith(("s", "ics")):
        return set()
    if re.search(r"[^aeiou]y$", word):
        candidates = {word[:-1] + "ies"}
    elif word.endswith(("ch", "sh", "x", "z")):
        candidates = {word + "es"}
    else:
        candidates = {word + "s"}
    irregular = {
        "analysis": "analyses",
        "criterion": "criteria",
        "index": "indices",
        "matrix": "matrices",
        "medium": "media",
        "phenomenon": "phenomena",
    }
    if word in irregular:
        candidates.add(irregular[word])
    return candidates


def audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    aliases = load_keyword_aliases()
    counts = {"author": Counter(), "index": Counter(), "combined": Counter()}

    for _, row in frame.iterrows():
        per_source = {
            source: _variants(row.get(column, ""))
            for source, column in SOURCE_COLUMNS.items()
        }
        for source, variants in per_source.items():
            counts[source].update(variants)
        counts["combined"].update(per_source["author"] | per_source["index"])

    observed = set(counts["combined"])
    rows = []
    for variant in sorted(observed):
        canonical = normalize_keyword(variant, aliases)
        orthographic = " ".join(
            _ORTHOGRAPHIC_TOKEN_ALIASES.get(word, word)
            for word in variant.split()
        )
        if variant in aliases:
            rule = "exact_alias"
        elif orthographic != variant:
            rule = "approved_orthographic_spelling"
        elif canonical != variant:
            rule = "approved_head_inflection"
        else:
            rule = "unchanged"
        rows.append(
            {
                "source_variant": variant,
                "canonical_keyword": canonical,
                "normalization_rule": rule,
                "author_papers": counts["author"][variant],
                "index_papers": counts["index"][variant],
                "combined_papers": counts["combined"][variant],
            }
        )
    variants = pd.DataFrame(rows)
    group_sizes = variants.groupby("canonical_keyword")["source_variant"].transform("size")
    variants.insert(2, "variants_in_canonical_group", group_sizes)
    variants = variants.sort_values(
        ["combined_papers", "canonical_keyword", "source_variant"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    candidates = []
    for singular in observed:
        words = singular.split()
        for plural_head in _possible_plurals(words[-1]):
            plural = " ".join([*words[:-1], plural_head])
            if plural not in observed:
                continue
            singular_canonical = normalize_keyword(singular, aliases)
            plural_canonical = normalize_keyword(plural, aliases)
            if singular_canonical == plural_canonical:
                continue
            candidates.append(
                {
                    "candidate_a": singular,
                    "candidate_b": plural,
                    "candidate_a_papers": counts["combined"][singular],
                    "candidate_b_papers": counts["combined"][plural],
                    "affected_papers_upper_bound": (
                        counts["combined"][singular] + counts["combined"][plural]
                    ),
                    "decision": "manual_review_required",
                }
            )
    unresolved = pd.DataFrame(candidates).drop_duplicates()
    if not unresolved.empty:
        unresolved = unresolved.sort_values(
            ["affected_papers_upper_bound", "candidate_a", "candidate_b"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    changed = variants[variants["normalization_rule"] != "unchanged"]
    summary = {
        "papers": len(frame),
        "observed_normalized_source_variants": len(variants),
        "variants_mapped": len(changed),
        "exact_alias_variants": int(
            (variants["normalization_rule"] == "exact_alias").sum()
        ),
        "approved_orthographic_spelling_variants": int(
            (
                variants["normalization_rule"]
                == "approved_orthographic_spelling"
            ).sum()
        ),
        "approved_head_inflection_variants": int(
            (variants["normalization_rule"] == "approved_head_inflection").sum()
        ),
        "canonical_keywords": int(variants["canonical_keyword"].nunique()),
        "unresolved_inflection_candidates": len(unresolved),
        "source_columns": SOURCE_COLUMNS,
        "source_fields_mutated": False,
    }
    return variants, unresolved, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    missing = [column for column in SOURCE_COLUMNS.values() if column not in frame.columns]
    if missing:
        raise ValueError(f"Keyword audit input is missing columns: {missing}")

    variants, unresolved, summary = audit(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants.to_csv(args.output_dir / "keyword_variant_mappings.csv", index=False)
    unresolved.to_csv(
        args.output_dir / "unresolved_keyword_variant_candidates.csv", index=False
    )
    (args.output_dir / "keyword_normalization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
