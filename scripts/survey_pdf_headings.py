#!/usr/bin/env python3
"""Survey the ACTUAL headings used across the full-text PDFs, grouped by journal.

Purpose: decide the extraction rules from evidence instead of guessing. This
script deliberately does NOT use the heading classifier in extract_fulltext.py.
It collects candidates with a permissive typographic rule and reports what is
really there, so the section vocabulary and the size/bold thresholds can be set
from observed practice rather than assumption.

It answers four questions:
  1. What heading strings do these journals actually use, and how often?
  2. Which frequent headings does the current SECTION_WORDS vocabulary MISS?
  3. Per journal, are headings bold, larger, numbered, or upper-case?
  4. Is case a real factor (ALL CAPS, Title Case, Sentence case)?

Read-only. No API calls. Writes only under --out-dir.

Typical use:

    python scripts/survey_pdf_headings.py --manifest data/interim/fulltext_prep/fulltext_manifest.csv
    python scripts/survey_pdf_headings.py --manifest ... --limit 40
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    raise SystemExit("PyMuPDF (fitz) is required: pip install pymupdf")

from extract_fulltext import SECTION_WORDS, page_blocks_in_reading_order

DEFAULT_MANIFEST = "data/interim/fulltext_prep/fulltext_manifest.csv"
DEFAULT_OUT = "data/interim/fulltext_prep/heading_survey"

# The numeral must be followed by a delimiter, otherwise a bare [0-9IVXivx]+
# class eats the leading capital of headings like "Introduction" or "Variables".
_LEAD_NUM = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[IVXLC]{1,5}[.)]\s*)")
_SECTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(SECTION_WORDS, key=len, reverse=True)) + r")\b"
)


def normalize(text: str) -> str:
    t = _LEAD_NUM.sub("", text.strip()).strip().rstrip(":").strip()
    return re.sub(r"\s+", " ", t).lower()


def case_style(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "ALLCAPS"
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) > 3]
    if words and sum(w[0].isupper() for w in words) / len(words) > 0.7:
        return "TitleCase"
    return "Sentence"


def collect_candidates(pdf_path: Path):
    """Permissive candidate collection: short lines that are bold or larger than body."""
    doc = fitz.open(pdf_path)
    spans = []
    for pno, page in enumerate(doc, start=1):
        for block in page_blocks_in_reading_order(page):
            for line in block["lines"]:
                text = "".join(s.get("text", "") for s in line["spans"]).strip()
                if not text:
                    continue
                sizes = [s.get("size", 0) for s in line["spans"]]
                bold = any(
                    (s.get("flags", 0) & 16) or "bold" in str(s.get("font", "")).lower()
                    for s in line["spans"]
                )
                spans.append({"text": text, "size": round(max(sizes) if sizes else 0, 1),
                              "bold": bold, "page": pno})
    doc.close()
    if not spans:
        return 0.0, []

    weighted: Counter = Counter()
    for s in spans:
        weighted[s["size"]] += len(s["text"])
    body = weighted.most_common(1)[0][0] if weighted else 0.0

    cands = []
    for s in spans:
        words = s["text"].split()
        if not (1 <= len(words) <= 14) or len(s["text"]) > 120:
            continue
        ratio = (s["size"] / body) if body else 0.0
        if ratio >= 1.05 or s["bold"]:
            cands.append({**s, "ratio": round(ratio, 3)})
    return body, cands


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=3,
                    help="min occurrences to report a heading as recurrent")
    ap.add_argument("--top", type=int, default=45, help="rows to print per console table")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("ready_for_fulltext") == "1"]
    todo = [r for r in rows if (r.get("staged_pdf_path") or r.get("pdf_path"))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Surveying headings in {len(todo)} PDFs")

    all_rows = []
    freq: Counter = Counter()
    heading_papers = defaultdict(set)
    heading_journals = defaultdict(set)
    jour_stats = defaultdict(lambda: {"papers": set(), "bold": 0, "n": 0,
                                      "ratios": [], "numbered": 0, "allcaps": 0})
    case_counter: Counter = Counter()
    failed = 0

    for i, r in enumerate(todo, 1):
        src = Path(r.get("staged_pdf_path") or r.get("pdf_path"))
        if not src.is_file():
            continue
        journal = (r.get("corpus_source_title") or r.get("zotero_journal") or "UNKNOWN").strip()
        pid = (r.get("paper_id") or "").replace("eid:", "")
        try:
            body, cands = collect_candidates(src)
        except Exception as exc:
            failed += 1
            print(f"  [{i}] FAILED {pid}: {type(exc).__name__}")
            continue

        st = jour_stats[journal]
        st["papers"].add(pid)
        for c in cands:
            norm = normalize(c["text"])
            if not norm or not re.search(r"[a-z]", norm):
                continue
            style = case_style(c["text"])
            numbered = bool(re.match(r"^[0-9]+(\.[0-9]+)*[.)]?\s+\S", c["text"]))
            freq[norm] += 1
            heading_papers[norm].add(pid)
            heading_journals[norm].add(journal)
            st["n"] += 1
            st["bold"] += int(c["bold"])
            st["ratios"].append(c["ratio"])
            st["numbered"] += int(numbered)
            st["allcaps"] += int(style == "ALLCAPS")
            case_counter[style] += 1
            all_rows.append({
                "paper_id": pid, "journal": journal, "page": c["page"],
                "heading_raw": c["text"], "heading_norm": norm,
                "size": c["size"], "body_size": body, "size_ratio": c["ratio"],
                "bold": int(c["bold"]), "numbered": int(numbered), "case_style": style,
                "matched_by_current_vocab": int(
                    norm in SECTION_WORDS or bool(_SECTION_RE.search(norm))
                ),
            })
        if i % 25 == 0:
            print(f"  [{i}/{len(todo)}] ...")

    if not all_rows:
        raise SystemExit("No heading candidates collected.")

    with (out / "heading_candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Recurrent headings and whether the current vocabulary catches them
    freq_rows = []
    for norm, n in freq.most_common():
        if n < args.min_count:
            continue
        matched = norm in SECTION_WORDS or bool(_SECTION_RE.search(norm))
        freq_rows.append({
            "heading_norm": norm, "occurrences": n,
            "n_papers": len(heading_papers[norm]),
            "n_journals": len(heading_journals[norm]),
            "matched_by_current_vocab": int(matched),
        })
    with (out / "heading_frequency.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(freq_rows[0].keys()))
        w.writeheader()
        w.writerows(freq_rows)

    missing = [r for r in freq_rows if not r["matched_by_current_vocab"]]
    with (out / "vocabulary_gaps.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(freq_rows[0].keys()))
        w.writeheader()
        w.writerows(missing)

    jrows = []
    for j, st in sorted(jour_stats.items(), key=lambda kv: -len(kv[1]["papers"])):
        if not st["n"]:
            continue
        jrows.append({
            "journal": j, "papers": len(st["papers"]), "candidates": st["n"],
            "pct_bold": round(100 * st["bold"] / st["n"], 1),
            "pct_numbered": round(100 * st["numbered"] / st["n"], 1),
            "pct_allcaps": round(100 * st["allcaps"] / st["n"], 1),
            "median_size_ratio": round(statistics.median(st["ratios"]), 3),
            "p90_size_ratio": round(sorted(st["ratios"])[int(len(st["ratios"]) * 0.9)], 3),
        })
    with (out / "journal_typography.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jrows[0].keys()))
        w.writeheader()
        w.writerows(jrows)

    coverage = 100 * sum(r["occurrences"] for r in freq_rows if r["matched_by_current_vocab"]) / max(
        sum(r["occurrences"] for r in freq_rows), 1
    )
    (out / "survey_summary.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pdfs_surveyed": len(todo), "failed": failed,
        "candidate_lines": len(all_rows),
        "distinct_normalized_headings": len(freq),
        "recurrent_headings_min_count": args.min_count,
        "recurrent_headings": len(freq_rows),
        "vocabulary_gaps": len(missing),
        "recurrent_coverage_pct_by_current_vocab": round(coverage, 1),
        "case_styles": dict(case_counter),
    }, indent=2), encoding="utf-8")

    print(f"\n=== case styles ===  {dict(case_counter)}")
    print(f"\n=== current vocabulary covers {coverage:.1f}% of recurrent heading occurrences ===")
    print(f"\n=== TOP RECURRENT HEADINGS NOT MATCHED by current vocabulary "
          f"({len(missing)} of {len(freq_rows)}) ===")
    print(f"{'occ':>5} {'papers':>7} {'jrnls':>6}  heading")
    for r in missing[: args.top]:
        print(f"{r['occurrences']:>5} {r['n_papers']:>7} {r['n_journals']:>6}  {r['heading_norm'][:64]}")

    print(f"\n=== journal typography (top {min(12, len(jrows))}) ===")
    print(f"{'papers':>6} {'bold%':>6} {'num%':>6} {'caps%':>6} {'medRatio':>9}  journal")
    for r in jrows[:12]:
        print(f"{r['papers']:>6} {r['pct_bold']:>6} {r['pct_numbered']:>6} "
              f"{r['pct_allcaps']:>6} {r['median_size_ratio']:>9}  {r['journal'][:48]}")

    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
