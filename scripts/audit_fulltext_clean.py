#!/usr/bin/env python3
"""Audit cleaned Markdown for residual publisher furniture and structural faults.

Deliberately INDEPENDENT of clean_fulltext.py's rules. Re-running the cleaner's
own patterns would only confirm what it already removed; this uses broader,
differently-shaped checks to find what LEAKED THROUGH, so cleaning rules can be
written against measured evidence instead of one sampled paper at a time.

Read-only. Writes only under --out-dir. Calls no API.

Checks fall into three families:
  line-level    email, URL, ORCID, ISSN, copyright, download stamp, page number,
                volume/issue, affiliation, author line, date, editor, table
                caption, garbled text, escaped-hash artifacts
  structure     lowercase or single-word headings, tiny sections, duplicate
                headings, non-content sections that survived
  cross-paper   identical lines recurring across many papers, which is how
                journal running heads reveal themselves without knowing the
                journal in advance

Typical use:

    python scripts/audit_fulltext_clean.py
    python scripts/audit_fulltext_clean.py --show 2-s2.0-85068602661
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_IN = "data/interim/fulltext_clean"
DEFAULT_OUT = "data/interim/fulltext_clean/audit"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Line-level checks. Order matters only for which label a line reports first.
LINE_CHECKS = [
    ("email", re.compile(r"[\w.%+-]+@[\w.-]+\.\w{2,}")),
    ("orcid", re.compile(r"orcid\.org", re.I)),
    ("url", re.compile(r"https?://|www\.\w|doi\.org/", re.I)),
    ("issn_isbn", re.compile(r"\bissn\b|\bisbn\b", re.I)),
    ("copyright", re.compile(r"©|\(c\)\s*\d{4}|all rights reserved|springer nature|"
                             r"elsevier (?:ltd|b\.v)|emerald publishing", re.I)),
    ("download_stamp", re.compile(r"downloaded from|this content downloaded|"
                                  r"all use subject to|jstor|researchgate", re.I)),
    ("page_number", re.compile(r"^\s*\d{1,4}\s*$")),
    ("volume_issue", re.compile(r"^\s*(?:vol\.?|volume|no\.?|issue)\s*\d+|^\s*\d{1,3}\s*[,/]\s*\d{1,3}\s*$|"
                                r"^\s*pp?\.\s*\d+\s*[-–]\s*\d+", re.I)),
    ("editor_line", re.compile(r"\b(?:senior|associate|handling|area)\s+editor\b|"
                               r"\baccepted by\b|\bhistory\s*:", re.I)),
    ("date_line", re.compile(r"^\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+"
                             r"\d{1,2},?\s+\d{4}\s*$|^\s*\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|"
                             r"aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\s*$", re.I)),
    ("table_caption", re.compile(r"^\s*(?:table|figure|fig\.?|panel|exhibit)\s*\d+[.:)]?\s", re.I)),
    ("keywords_inline", re.compile(r"^\s*(?:keywords?|jel\s+classification|jel\s+codes?)\b", re.I)),
    ("escaped_hash", re.compile(r"^\s*\\#")),
    ("cid_artifact", re.compile(r"\(cid:\d+\)")),
    ("contact_label", re.compile(r"^\s*(?:contact|correspondence|corresponding author|"
                                 r"supplemental material|declarations?)\s*:", re.I)),
]

# Affiliation: capital after the institution word, so "school of thought" is safe.
_AFFILIATION_RE = re.compile(
    r"\b(?:School|Department|Faculty|College|Institute|Centre|Center|University)\b[,.]?\s+(?:of\s+)?[A-Z]"
)
# Author lines: "Hwang E.H.; Singh P.V." or "Elina H. Hwang,a Param Vir Singh,b"
_AUTHOR_SEMI_RE = re.compile(r"^[A-Z][A-Za-z'\-]+ [A-Z]\.(?:[A-Z]\.)*(?:\s*;\s*[A-Z][A-Za-z'\-]+ [A-Z]\.)+")
_AUTHOR_SUP_RE = re.compile(r"^(?:[A-Z][A-Za-z.'\-]+\s+){1,3}[A-Z][A-Za-z'\-]+,[a-z](?:\s|,|$)")

NON_CONTENT = {
    "references", "bibliography", "acknowledgements", "acknowledgments", "funding",
    "declarations", "declaration of competing interest", "author contributions",
    "credit authorship contribution statement", "conflict of interest",
    "data availability", "supplementary material", "appendix",
}

TINY_SECTION_WORDS = 25


def classify_line(line: str) -> list[str]:
    t = line.strip()
    if not t:
        return []
    hits = []
    for name, rx in LINE_CHECKS:
        if rx.search(t):
            hits.append(name)
    words = t.split()
    if len(words) <= 30 and _AFFILIATION_RE.search(t):
        hits.append("affiliation")
    if _AUTHOR_SEMI_RE.match(t) or _AUTHOR_SUP_RE.match(t):
        hits.append("author_line")
    letters = sum(c.isalpha() for c in t)
    if len(t) > 20 and letters / len(t) < 0.5:
        hits.append("garbled_low_alpha")
    return hits


def parse_md(text: str):
    sections, current = [], {"level": 2, "title": "", "lines": []}
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if current["title"] or current["lines"]:
                sections.append(current)
            current = {"level": len(m.group(1)) - 1, "title": m.group(2).strip(), "lines": []}
        else:
            current["lines"].append(line)
    if current["title"] or current["lines"]:
        sections.append(current)
    return sections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--show", metavar="PAPER_ID", help="print all findings for one paper")
    ap.add_argument("--top", type=int, default=12, help="examples to print per issue")
    ap.add_argument("--repeat-min-papers", type=int, default=4,
                    help="a line recurring in this many papers is flagged as furniture")
    args = ap.parse_args()

    md_dir = Path(args.in_dir) / "md"
    if not md_dir.is_dir():
        raise SystemExit(f"No cleaned Markdown in {md_dir}. Run clean_fulltext.py first.")
    out_dir = Path(args.out_dir)

    files = sorted(md_dir.glob("*.md"))
    if args.show:
        files = [f for f in files if f.stem == args.show]
        if not files:
            raise SystemExit(f"No cleaned Markdown for {args.show}")

    findings = []
    issue_counts: Counter = Counter()
    per_paper: Counter = Counter()
    line_papers = defaultdict(set)
    total_lines = 0
    heading_titles = defaultdict(list)

    for path in files:
        pid = path.stem
        text = path.read_text(encoding="utf-8")
        sections = parse_md(text)
        seen_titles: Counter = Counter()

        for sec in sections:
            title = sec["title"].strip()
            body_words = len(" ".join(sec["lines"]).split())

            # The corpus-supplied Abstract and Keywords sections are the
            # authoritative record. They are legitimately short and legitimately
            # named, so exempt them rather than reporting 131 false positives.
            if title in ("Abstract", "Keywords") and sec["level"] == 1:
                continue

            if title:
                seen_titles[title.lower()] += 1
                heading_titles[title.lower()].append(pid)
                flat = re.sub(r"[^a-z0-9 ]", " ", title.lower()).strip()
                if title[0].islower():
                    findings.append((pid, "heading_lowercase", title))
                if len(flat.split()) == 1 and flat not in NON_CONTENT and len(flat) < 4:
                    findings.append((pid, "heading_single_short_word", title))
                if sec["level"] >= 1 and body_words < TINY_SECTION_WORDS:
                    findings.append((pid, "tiny_section", f"{title} ({body_words}w)"))
                if flat in NON_CONTENT:
                    findings.append((pid, "non_content_section_present", title))
                for h in classify_line(title):
                    findings.append((pid, f"heading_{h}", title))

            for line in sec["lines"]:
                t = line.strip()
                if not t:
                    continue
                total_lines += 1
                if len(t) < 200:
                    line_papers[t].add(pid)
                for h in classify_line(t):
                    findings.append((pid, h, t[:180]))

        for title, n in seen_titles.items():
            if n > 1:
                findings.append((pid, "duplicate_heading", f"{title} x{n}"))

    # Cross-paper repetition: journal furniture reveals itself without knowing
    # the journal, because the same string recurs in unrelated papers.
    # A repeated line is only furniture if it is a PHRASE. Single common words
    # ("Theme", "Technology", "outcomes.") recur across unrelated papers for
    # ordinary reasons and swamped the first run of this audit.
    for line, pids in line_papers.items():
        if (len(pids) >= args.repeat_min_papers
                and 4 <= len(line.split()) <= 20 and len(line) >= 25):
            for pid in pids:
                findings.append((pid, "repeated_across_papers", line[:180]))

    for pid, issue, _ in findings:
        issue_counts[issue] += 1
        per_paper[pid] += 1

    if args.show:
        print(f"=== {files[0].stem}: {len(findings)} findings ===")
        by_issue = defaultdict(list)
        for _, issue, line in findings:
            by_issue[issue].append(line)
        for issue, lines in sorted(by_issue.items(), key=lambda kv: -len(kv[1])):
            print(f"\n[{issue}]  {len(lines)}")
            for ln in lines[: args.top]:
                print(f"    {ln}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "audit_findings.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["paper_id", "issue", "line"])
        w.writerows(findings)

    summary = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "papers": len(files),
        "body_lines_scanned": total_lines,
        "findings": len(findings),
        "findings_per_1000_lines": round(1000 * len(findings) / max(total_lines, 1), 2),
        "issues": dict(issue_counts.most_common()),
        "papers_with_no_findings": sum(1 for f in files if per_paper[f.stem] == 0),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"papers: {len(files)} | body lines: {total_lines:,} | findings: {len(findings)}"
          f" ({summary['findings_per_1000_lines']} per 1k lines)")
    print(f"papers with zero findings: {summary['papers_with_no_findings']}\n")

    print("=== issues by frequency ===")
    by_issue = defaultdict(list)
    for pid, issue, line in findings:
        by_issue[issue].append((pid, line))
    for issue, items in sorted(by_issue.items(), key=lambda kv: -len(kv[1])):
        papers = len({p for p, _ in items})
        print(f"\n[{issue}]  {len(items)} lines in {papers} papers")
        seen = set()
        shown = 0
        for pid, line in items:
            if line in seen:
                continue
            seen.add(line)
            print(f"    {line[:110]}")
            shown += 1
            if shown >= args.top:
                break

    print("\n=== worst papers ===")
    for pid, n in per_paper.most_common(10):
        print(f"  {n:>5}  {pid}")
    print(f"\nwrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
