#!/usr/bin/env python3
"""Paired abstract-versus-full-text comparison, within model.

The question this answers is the one that demoted the definition dimensions:
how much apparent under-specification is genuine absence, and how much is the
abstract simply not saying what the paper does state.

The comparison is WITHIN MODEL by construction. The same rater coded the same
papers under the same schema; only the evidentiary boundary differs. So a code
that changes is attributable to the boundary, not to two models disagreeing.

Headline metric per dimension: the RESOLUTION RATE, the share of papers coded
unspecified from the abstract that receive a substantive code once the full
text is read. A high rate means the abstract was withholding; a low rate means
the absence is real and the under-specification claim strengthens.

This is EXPLORATORY under METHODS_LOCK: it was specified after the
abstract-level distributions had been inspected.

Read-only with respect to caches. Writes only under --out-dir.

    python scripts/analyze_fulltext_paired.py
    python scripts/analyze_fulltext_paired.py --model gemini-3.1-pro-preview
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = PROJECT_ROOT / "data" / "processed" / "specification"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "analysis" / "fulltext_paired"

# The code that means "the evidence did not support a substantive value" for
# each dimension. Read from the controlled vocabularies, not invented here.
UNSPECIFIED = {
    "ai_role_function": {"AI as unspecified label"},
    "ai_type_form": {"unspecified AI"},
    "ai_mechanism_analysis": {"mechanism missing"},
    "level_of_analysis": {"unspecified level"},
    "entrepreneurial_process_stage": {"process unspecified"},
    "scope_conditions": {"scope missing", "generalised without scope"},
    "definition_construct_clarity": {"no definition"},
    "ai_definition_present": {"no"},
    "ai_distinction_present": {"no"},
    "process_sequence_specified": {"no"},
}


def load_codes(path: Path, dimensions: list[str], wanted: set[str] | None = None) -> dict:
    """Return {paper_id: {dimension: code}} for the requested papers."""
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("paper_id") or "").strip()
            if not pid or (wanted is not None and pid not in wanted):
                continue
            out[pid] = {d: (row.get(d) or "").strip() for d in dimensions}
            out[pid]["_prompt_tokens"] = row.get("prompt_tokens", "")
            out[pid]["_output_tokens"] = row.get("output_tokens", "")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--input-price", type=float, default=0.75,
                    help="USD per 1M input tokens, for the actual-cost report")
    ap.add_argument("--output-price", type=float, default=4.50)
    args = ap.parse_args()

    slug = args.model.replace("/", "_")
    abstract_csv = SPEC_DIR / f"paper_specifications_{slug}_spec-v3.csv"
    fulltext_csv = SPEC_DIR / f"paper_specifications_{slug}_spec-ft-v1.csv"
    for path in (abstract_csv, fulltext_csv):
        if not path.is_file():
            raise SystemExit(f"Missing coded dataset: {path}")

    dimensions = list(UNSPECIFIED)
    fulltext = load_codes(fulltext_csv, dimensions)
    paired_ids = set(fulltext)
    abstract = load_codes(abstract_csv, dimensions, wanted=paired_ids)

    common = sorted(paired_ids & set(abstract))
    if not common:
        raise SystemExit("No overlapping paper_ids between the two arms.")
    print(f"Model: {args.model}")
    print(f"Full-text papers: {len(paired_ids)} | with abstract codes: {len(common)}")
    if len(common) < len(paired_ids):
        print(f"NOTE: {len(paired_ids) - len(common)} full-text papers have no "
              f"abstract-level code and are excluded rather than imputed.")

    rows = []
    for dimension in dimensions:
        blank = unspec_codes = 0
        counts = Counter()
        for pid in common:
            a, f = abstract[pid].get(dimension, ""), fulltext[pid].get(dimension, "")
            if not a or not f:
                blank += 1
                continue
            a_unspec = a in UNSPECIFIED[dimension]
            f_unspec = f in UNSPECIFIED[dimension]
            unspec_codes += 1
            if a_unspec and not f_unspec:
                counts["resolved"] += 1
            elif not a_unspec and f_unspec:
                counts["introduced"] += 1
            elif a_unspec and f_unspec:
                counts["both_unspecified"] += 1
            elif a == f:
                counts["same_code"] += 1
            else:
                counts["changed_code"] += 1
            counts["abstract_unspecified"] += int(a_unspec)
            counts["fulltext_unspecified"] += int(f_unspec)

        n = unspec_codes
        a_unspec_n = counts["abstract_unspecified"]
        rows.append({
            "dimension": dimension,
            "n_compared": n,
            "blank_either_side": blank,
            "abstract_unspecified": a_unspec_n,
            "abstract_unspecified_pct": round(100 * a_unspec_n / max(n, 1), 1),
            "fulltext_unspecified": counts["fulltext_unspecified"],
            "fulltext_unspecified_pct": round(
                100 * counts["fulltext_unspecified"] / max(n, 1), 1),
            # The headline: of those the abstract could not settle, how many
            # does the full text settle?
            "resolved": counts["resolved"],
            "resolution_rate_pct": round(100 * counts["resolved"] / max(a_unspec_n, 1), 1),
            "introduced": counts["introduced"],
            "both_unspecified": counts["both_unspecified"],
            "same_code": counts["same_code"],
            "changed_code": counts["changed_code"],
            "agreement_pct": round(
                100 * (counts["same_code"] + counts["both_unspecified"]) / max(n, 1), 1),
        })

    rows.sort(key=lambda r: -r["resolution_rate_pct"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / f"paired_{slug}.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Actual cost, from recorded token usage rather than an estimate.
    pt = sum(int(v["_prompt_tokens"]) for v in fulltext.values() if str(v["_prompt_tokens"]).isdigit())
    ot = sum(int(v["_output_tokens"]) for v in fulltext.values() if str(v["_output_tokens"]).isdigit())
    cost = pt / 1e6 * args.input_price + ot / 1e6 * args.output_price

    print("\n=== resolution: abstract 'unspecified' settled by full text ===")
    print(f"{'dimension':<32}{'n':>5}{'abs%':>7}{'ft%':>7}{'resolved':>9}{'rate%':>7}{'new':>5}")
    for r in rows:
        print(f"{r['dimension']:<32}{r['n_compared']:>5}"
              f"{r['abstract_unspecified_pct']:>7}{r['fulltext_unspecified_pct']:>7}"
              f"{r['resolved']:>9}{r['resolution_rate_pct']:>7}{r['introduced']:>5}")

    print(f"\nactual tokens: {pt:,} in / {ot:,} out")
    print(f"actual cost at {args.input_price}/{args.output_price} per 1M: ${cost:.2f}")

    (args.out_dir / f"paired_{slug}_summary.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "protocols": {"abstract": "spec-v3", "full_text": "spec-ft-v1"},
        "within_model": True,
        "label": "exploratory",
        "papers_full_text": len(paired_ids),
        "papers_paired": len(common),
        "prompt_tokens": pt, "output_tokens": ot, "actual_cost_usd": round(cost, 2),
        "dimensions": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
