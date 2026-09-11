#!/usr/bin/env python3
"""Extract staged PDFs into section-aware Markdown plus a JSON sidecar.

One extraction feeds two downstream consumers with opposite needs:

  A. specification coding  -> the WHOLE paper in a single model call, so the
     construct definition (introduction), the mechanism (discussion) and the
     scope condition (limitations) can be connected. Never chunked.
  B. RAG storage           -> the same document chunked into passages for
     embedding and graph loading. Chunking happens in the RAG builder, not
     here; this script only preserves the section boundaries it will use.

Outputs per paper (paper_id = the corpus EID without the "eid:" prefix):

    <out-dir>/raw/<paper_id>.txt     untouched extractor output (audit trail)
    <out-dir>/md/<paper_id>.md       section-aware clean text
    <out-dir>/json/<paper_id>.json   metadata, sections, page map, provenance
    <out-dir>/extraction_report.csv  per-paper quality diagnostics
    <out-dir>/extraction_summary.json

This script is non-mutating and calls no API. Extraction quality is the main
risk in the full-text arm, so every paper carries diagnostic flags and the
report is meant to be READ before any coding run is launched.

Typical use:

    python scripts/extract_fulltext.py --manifest data/interim/fulltext_prep/fulltext_manifest.csv
    python scripts/extract_fulltext.py --manifest ... --limit 10   # sample first
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    raise SystemExit("PyMuPDF (fitz) is required: pip install pymupdf")

DEFAULT_MANIFEST = "data/interim/fulltext_prep/fulltext_manifest.csv"
DEFAULT_OUT = "data/interim/fulltext"

# Canonical IMRaD-ish section names used as a strong heading signal.
SECTION_WORDS = {
    "abstract", "introduction", "background", "literature review", "related work",
    "theory", "theoretical background", "theoretical framework", "hypotheses",
    "hypothesis development", "conceptual framework", "method", "methods",
    "methodology", "data", "data and methods", "research design", "sample",
    "measures", "analysis", "results", "findings", "discussion",
    "general discussion", "implications", "theoretical implications",
    "practical implications", "limitations", "limitations and future research",
    "future research", "conclusion", "conclusions", "concluding remarks",
    "references", "bibliography", "appendix", "acknowledgements",
    "acknowledgments", "declaration of competing interest", "funding",
    # Back matter observed by survey_pdf_headings.py across 154 papers.
    # Recognised so it can be identified and excluded from analysis chunks.
    "keywords", "declarations", "declaration", "data availability",
    "data availability statement", "credit authorship contribution statement",
    "author contributions", "conflict of interest", "conflicts of interest",
    "ethics statement", "supplementary material", "supplementary materials",
    "notes", "endnotes", "author biographies", "abbreviations",
}

# Sections that are apparatus, not argument. Downstream chunking and coding
# should skip these rather than embed boilerplate as if it were content.
NON_CONTENT_SECTIONS = {
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "funding", "declarations", "declaration", "declaration of competing interest",
    "credit authorship contribution statement", "author contributions",
    "conflict of interest", "conflicts of interest", "data availability",
    "data availability statement", "ethics statement", "supplementary material",
    "supplementary materials", "author biographies", "abbreviations", "keywords",
}

# Tokens-per-word heuristic; tiktoken is not installed in the graphrag env.
TOKENS_PER_WORD = 1.35

# Words that end a clause, never a heading. Used to reject prose fragments.
_TRAILING_STOPWORDS = {
    "and", "or", "the", "of", "to", "with", "are", "is", "was", "were", "in",
    "on", "for", "that", "which", "a", "an", "as", "by", "from", "we", "our",
}

# Strip a leading section number ("3.", "2.1", "IV.") from a heading.
# The numeral MUST be followed by a delimiter. A naive [0-9IVXivx]+ class eats
# the first letter of any heading starting with I, V or X, silently turning
# "Introduction" into "ntroduction" and "Variables" into "ariables", which
# breaks every downstream vocabulary match.
_LEAD_NUM = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[IVXLC]{1,5}[.)]\s*)")

# Single alternation over the canonical section vocabulary, longest first so
# "literature review" wins over "review". Compiled once, used per line.
_SECTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(SECTION_WORDS, key=len, reverse=True)) + r")\b"
)

_WS = re.compile(r"[ \t ]+")
_MULTINL = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


# --------------------------------------------------------------- text cleanup


def clean_text(text: str) -> str:
    for lig, rep in _LIGATURES.items():
        text = text.replace(lig, rep)
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # rejoin words split across lines
    text = _WS.sub(" ", text)
    text = _MULTINL.sub("\n\n", text)
    # Escape body lines that begin with "#" so they cannot be re-parsed as
    # Markdown headings downstream. Springer copyright lines are the common
    # case: the "©" glyph extracts as "#", producing a phantom level-1 heading
    # that corrupts the document tree.
    text = re.sub(r"(?m)^(#+)(?=\s|\S)", r"\\\1", text)
    return text.strip()


def looks_like_heading(text: str, size: float, body_size: float, bold: bool) -> bool:
    """Conservative heading test.

    Deliberately strict. In two-column layouts body lines are short and can
    measure marginally larger than the modal font, so a loose "short and a bit
    bigger" rule turns every line into a heading. A candidate must therefore be
    a known section word, or numbered, or clearly emphasised, and must not read
    as a fragment of a running sentence.
    """
    t = text.strip()
    if not t or len(t) > 120 or not re.search(r"[A-Za-z]", t):
        return False

    words = t.split()

    # Prose-rejection guards run FIRST and apply to every path, so that a
    # canonical section word appearing inside running text ("...the results
    # show...") can never promote a body line to a heading.
    if t[0].islower():
        return False
    if t.endswith((",", ";", "-")) or words[-1].lower() in _TRAILING_STOPWORDS:
        return False
    if t.endswith(".") and len(words) > 8:
        return False
    if len(words) > 14:
        return False
    if sum(t.count(ch) for ch in ",;") > 1:
        return False

    stripped = _LEAD_NUM.sub("", t).strip().lower().rstrip(":")
    # Compound headings are the norm ("Data and Variable Operationalization",
    # "Results and Robustness Checks"), so match a canonical word ANYWHERE in
    # the heading rather than requiring the whole string to equal one.
    # An exact canonical heading that has already cleared the prose guards
    # needs no typographic corroboration: a line reading only "Methodology" is
    # a heading even when set at body size and not bold.
    if stripped in SECTION_WORDS:
        return True

    # Table and figure captions are short, often bold, and are NOT sections.
    if re.match(r"^(table|figure|fig\.?|panel|exhibit|appendix table)\s*[0-9ivxa-d]", stripped):
        return False

    canonical = bool(_SECTION_RE.search(stripped))
    numbered = bool(re.match(r"^\d+(\.\d+)*\.?\s+\S", t))
    bigger = size >= body_size * 1.15
    letters = [c for c in t if c.isalpha()]
    allcaps = bool(letters) and all(c.isupper() for c in letters)

    # Thresholds set from survey_pdf_headings.py across 154 papers. Bold at the
    # SAME size is the dominant convention (JKE 99%, JSBM 98.5%, IEMJ 98.1%,
    # SBE 95.9% bold, all at a median size ratio of 1.0 or below), so requiring
    # a size increase discards most real headings. A minority of journals use
    # size or capitals instead (AMR: 0% bold, 60% ALLCAPS, ratio 1.2).
    if bold and len(words) <= 12:
        return True
    if allcaps and len(words) <= 12:
        return True
    if bigger and len(words) <= 12:
        return True
    if canonical and (bold or bigger or numbered or allcaps):
        return True
    return bool(numbered and (bold or bigger))


# Headings that are top-level by convention even when unnumbered. Narrower than
# SECTION_WORDS: back matter and sub-parts are excluded so they cannot claim
# level 1 and orphan the sections that should nest under them.
CORE_SECTIONS = {
    "abstract", "introduction", "background", "literature review", "related work",
    "theory", "theoretical background", "theoretical framework", "conceptual framework",
    "method", "methods", "methodology", "data", "data and methods", "research design",
    "analysis", "results", "findings", "discussion", "general discussion",
    "implications", "limitations", "conclusion", "conclusions", "concluding remarks",
    "references", "bibliography", "appendix",
}

_LEAD_NUM_CAPTURE = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+\S")


def assign_levels(sections: list[dict]) -> None:
    """Infer a heading level for each section, in place.

    Priority, strongest signal first:
      1. Explicit numbering depth. "4 Results" is level 1, "4.1 Sample" level 2.
         This is the most reliable signal and is trusted whenever present.
      2. A canonical top-level section name ("Results", "Discussion").
      3. Relative font size, ranked within the document.

    An unnumbered, non-canonical heading never claims level 1 in a document that
    has level-1 headings, so a bare "SVC" nests under the "4 Results" above it
    rather than being flattened alongside it.
    """
    sizes = sorted({s.get("size", 0.0) for s in sections}, reverse=True)
    rank = {sz: i for i, sz in enumerate(sizes)}

    has_explicit_top = any(
        _LEAD_NUM_CAPTURE.match(s["title"].strip())
        or _LEAD_NUM.sub("", s["title"]).strip().lower().rstrip(":") in CORE_SECTIONS
        for s in sections
    )

    for sec in sections:
        title = sec["title"].strip()
        norm = _LEAD_NUM.sub("", title).strip().lower().rstrip(":")
        m = _LEAD_NUM_CAPTURE.match(title)
        if m:
            sec["level"] = min(m.group(1).count(".") + 1, 4)
        elif norm in CORE_SECTIONS:
            sec["level"] = 1
        else:
            inferred = min(rank.get(sec.get("size", 0.0), 1) + 1, 3)
            sec["level"] = max(2, inferred) if has_explicit_top else inferred

    # Breadcrumb of ancestors, so a retrieved chunk can say "Results > SVC".
    stack: list[tuple[int, str]] = []
    for sec in sections:
        lvl = sec["level"]
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        sec["path"] = " > ".join([t for _, t in stack] + [sec["title"].strip()])
        stack.append((lvl, sec["title"].strip()))


def merge_small_sections(sections: list[dict], min_words: int = 40) -> list[dict]:
    """Fold undersized sections into the previous one.

    A real section has substantive body text. Anything smaller is almost always
    a misfired heading (a bold lead-in, a table label, a caption).
    """
    merged: list[dict] = []
    for sec in sections:
        body_words = len(" ".join(sec["lines"]).split())
        if merged and body_words < min_words:
            merged[-1]["lines"].append(sec["title"])
            merged[-1]["lines"].extend(sec["lines"])
            merged[-1]["page_end"] = sec.get("page_end", merged[-1].get("page_end"))
        else:
            merged.append(sec)
    return merged


# ----------------------------------------------------------------- extraction


def page_blocks_in_reading_order(page) -> list[dict]:
    """Return text blocks sorted into reading order, handling two-column layouts."""
    data = page.get_text("dict")
    blocks = [b for b in data.get("blocks", []) if b.get("type") == 0 and b.get("lines")]
    if not blocks:
        return []
    width = page.rect.width
    centers = [(b["bbox"][0] + b["bbox"][2]) / 2 for b in blocks]
    mid = width / 2
    left = [c for c in centers if c < mid]
    right = [c for c in centers if c >= mid]
    # Treat as two columns only when both sides are well populated and separated.
    two_col = (
        len(left) >= 3 and len(right) >= 3
        and abs(statistics.median(left) - statistics.median(right)) > width * 0.22
    )
    if two_col:
        blocks.sort(key=lambda b: (0 if (b["bbox"][0] + b["bbox"][2]) / 2 < mid else 1, b["bbox"][1]))
    else:
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return blocks


def extract_document(pdf_path: Path, meta: dict | None = None) -> dict:
    doc = fitz.open(pdf_path)
    spans, pages_raw = [], []

    for pno, page in enumerate(doc, start=1):
        pages_raw.append(page.get_text("text"))
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
                spans.append(
                    {"text": text, "size": round(max(sizes), 1) if sizes else 0.0,
                     "bold": bold, "page": pno}
                )
    n_pages = doc.page_count
    doc.close()

    if not spans:
        return {"n_pages": n_pages, "raw": "\n".join(pages_raw), "sections": [],
                "markdown": "", "body_size": 0.0}

    # Weight font sizes by character mass, not line count: body prose dominates
    # the page by volume even when captions and references contribute more lines.
    size_counts: Counter = Counter()
    for s in spans:
        if s["text"]:
            size_counts[s["size"]] += len(s["text"])
    body_size = size_counts.most_common(1)[0][0] if size_counts else 0.0

    # Drop repeated running headers/footers. Digits are masked before comparison
    # so that per-page numbers ("554", "555") collapse to one pattern, and the
    # page threshold is low because running heads often alternate verso/recto.
    def shape(t: str) -> str:
        return re.sub(r"\d", "#", t)

    line_pages: dict[str, set[int]] = {}
    for s in spans:
        if len(s["text"]) < 90:
            line_pages.setdefault(shape(s["text"]), set()).add(s["page"])
    boiler_shapes = {
        t for t, pgs in line_pages.items() if len(pgs) >= max(3, n_pages * 0.34)
    }

    sections, current = [], {"title": "Front matter", "page_start": 1, "size": 0.0, "lines": []}
    for s in spans:
        if shape(s["text"]) in boiler_shapes:
            continue
        if looks_like_heading(s["text"], s["size"], body_size, s["bold"]):
            if current["lines"]:
                current["page_end"] = s["page"]
                sections.append(current)
            current = {"title": s["text"].strip(), "page_start": s["page"],
                       "size": s["size"], "lines": []}
        else:
            current["lines"].append(s["text"])
    if current["lines"]:
        current["page_end"] = spans[-1]["page"]
        sections.append(current)

    sections = merge_small_sections(sections)
    assign_levels(sections)

    parts, offset, out_sections = [], 0, []
    # The paper title is the document root ("#"), so sections nest beneath it.
    # It is taken from the corpus metadata rather than detected from page 1,
    # which is far more reliable across journal layouts. Offsets start after it
    # so section char ranges stay accurate.
    meta = meta or {}
    title = (meta.get("title") or "").strip()
    if title:
        root = "# {}\n\n".format(title)
        parts.append(root)
        offset = len(root)

    # Title, abstract and keywords come from the CORPUS, not the PDF. Two
    # reasons. PDF-derived front matter is unreliable (observed "Abstract"
    # sections of 1,878 and 4,810 words that had absorbed the introduction).
    # And spec-v3 defined its evidentiary unit as exactly title + abstract +
    # author keywords, so reusing that text keeps the full-text arm comparable
    # with the abstract-level study instead of quietly changing the base input.
    for label, key in (("Abstract", "abstract"), ("Keywords", "keywords")):
        body = (meta.get(key) or "").strip()
        if not body:
            continue
        header = "## {}\n\n".format(label)
        chunk = header + body + "\n\n"
        out_sections.append({
            "title": label, "level": 1, "path": label,
            "is_content": 1, "source": "corpus_metadata",
            "page_start": None, "page_end": None,
            "char_start": offset + len(header),
            "char_end": offset + len(chunk) - 2,
            "words": len(body.split()),
        })
        parts.append(chunk)
        offset += len(chunk)

    for sec in sections:
        body = clean_text("\n".join(sec["lines"]))
        if not body:
            continue
        # Markdown depth mirrors the inferred level: level 1 -> "##", 2 -> "###".
        header = "{} {}\n\n".format("#" * (sec["level"] + 1), sec["title"])
        chunk = header + body + "\n\n"
        norm_title_key = _LEAD_NUM.sub("", sec["title"]).strip().lower().rstrip(":")
        out_sections.append({
            "title": sec["title"],
            "level": sec["level"],
            "source": "pdf",
            "path": sec.get("path", sec["title"]),
            "is_content": int(norm_title_key not in NON_CONTENT_SECTIONS),
            "page_start": sec.get("page_start"),
            "page_end": sec.get("page_end"),
            "char_start": offset + len(header),
            "char_end": offset + len(chunk) - 2,
            "words": len(body.split()),
        })
        parts.append(chunk)
        offset += len(chunk)

    return {
        "n_pages": n_pages,
        "raw": "\n".join(pages_raw),
        "sections": out_sections,
        "markdown": "".join(parts).strip(),
        "body_size": body_size,
    }


# ------------------------------------------------------------------ diagnostics


def diagnose(doc: dict, md: str) -> tuple[list[str], str]:
    words = len(md.split())
    flags = []
    if words < 500:
        flags.append("very_little_text")
    if doc["n_pages"] and words / max(doc["n_pages"], 1) < 150:
        flags.append("low_text_density")
    if not doc["sections"]:
        flags.append("no_sections_detected")
    # Match section words ANYWHERE in a heading. Real headings are compound
    # ("Main Model Results", "Implications for Theory"), so exact equality
    # under-reports badly and produces false quality alarms.
    titles_blob = " | ".join(s["title"].lower() for s in doc["sections"])
    if not re.search(r"\b(introduction|background)\b", titles_blob):
        flags.append("no_introduction")
    if not re.search(r"\b(discussion|result|results|finding|findings|conclusion|conclusions)\b",
                     titles_blob):
        flags.append("no_discussion_or_results")
    letters = sum(c.isalpha() for c in md)
    if md and letters / max(len(md), 1) < 0.55:
        flags.append("low_alpha_ratio")
    if re.search(r"\(cid:\d+\)", md):
        flags.append("cid_font_artifacts")

    # Informational only: many management journals leave the opening section
    # unlabelled, so a missing "Introduction" heading says nothing about
    # whether the text extracted correctly.
    info_only = {"no_introduction"}
    serious = [f for f in flags if f not in info_only]

    if "very_little_text" in flags or "no_sections_detected" in flags:
        status = "poor"
    elif serious:
        status = "check"
    else:
        status = "good"
    return flags, status


# ------------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (sampling)")
    ap.add_argument("--only-ready", action="store_true", default=True,
                    help="restrict to rows with ready_for_fulltext=1 (default)")
    ap.add_argument("--include-unmatched", dest="only_ready", action="store_false",
                    help="also extract PDFs with no corpus match")
    ap.add_argument("--overwrite", action="store_true", help="re-extract papers already done")
    args = ap.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}. Run prepare_fulltext_corpus.py first.")

    out = Path(args.out_dir)
    for sub in ("raw", "md", "json"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Complete corpus record per paper, written by prepare_fulltext_corpus.py.
    # Embedded verbatim in each sidecar so the graph and RAG stages never have
    # to re-read the 321MB master corpus.
    meta_path = manifest.parent / "paper_metadata.json"
    corpus_meta = {}
    if meta_path.is_file():
        corpus_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"Corpus metadata: {len(corpus_meta)} records from {meta_path.name}")
    else:
        print(f"NOTE: {meta_path.name} not found; sidecars will carry manifest fields only.")

    todo = []
    for r in rows:
        if args.only_ready and r.get("ready_for_fulltext") != "1":
            continue
        src = r.get("staged_pdf_path") or r.get("pdf_path") or ""
        if not src or not Path(src).is_file():
            continue
        todo.append(r)
    if args.limit:
        todo = todo[: args.limit]

    print(f"Manifest rows: {len(rows)} | to extract: {len(todo)}")
    if not todo:
        raise SystemExit("Nothing to extract. Check ready_for_fulltext / pdf paths.")

    report, stats = [], Counter()
    total_words = 0

    for i, r in enumerate(todo, 1):
        paper_id = (r.get("paper_id") or "").replace("eid:", "") or f"zotero_{r.get('zotero_key')}"
        md_path = out / "md" / f"{paper_id}.md"
        if md_path.exists() and not args.overwrite:
            stats["skipped_existing"] += 1
            continue

        src = Path(r.get("staged_pdf_path") or r.get("pdf_path"))
        record = corpus_meta.get(r.get("paper_id", ""), {})
        doc_meta = {
            "title": record.get("Title") or r.get("corpus_title") or r.get("zotero_title") or "",
            "abstract": record.get("Abstract") or r.get("corpus_abstract") or "",
            "keywords": record.get("Author Keywords") or r.get("corpus_keywords") or "",
        }
        try:
            doc = extract_document(src, doc_meta)
        except Exception as exc:  # keep going; a bad PDF is data, not a crash
            stats["failed"] += 1
            report.append({
                "paper_id": paper_id, "status": "failed", "flags": f"exception:{type(exc).__name__}",
                "pages": "", "words": "", "sections": "", "est_input_tokens": "",
                "title": r.get("corpus_title", ""), "pdf_path": str(src),
            })
            print(f"[{i}/{len(todo)}] FAILED {paper_id}: {exc}")
            continue

        md = doc["markdown"]
        flags, status = diagnose(doc, md)
        words = len(md.split())
        total_words += words
        est_tokens = int(words * TOKENS_PER_WORD)

        (out / "raw" / f"{paper_id}.txt").write_text(doc["raw"], encoding="utf-8")
        md_path.write_text(md, encoding="utf-8")
        (out / "json" / f"{paper_id}.json").write_text(
            json.dumps(
                {
                    "paper_id": r.get("paper_id", ""),
                    "doi": r.get("corpus_doi") or r.get("zotero_doi", ""),
                    "title": doc_meta["title"],
                    "year": r.get("corpus_year") or r.get("zotero_year", ""),
                    "source_title": r.get("corpus_source_title", ""),
                    "authors": record.get("Authors") or r.get("zotero_authors", ""),
                    # Every corpus field, verbatim. The graph stage reads
                    # authors, affiliations, journal, year, keywords and query
                    # membership straight from here.
                    "corpus_metadata": record,
                    "zotero_key": r.get("zotero_key", ""),
                    "in_workbook_previously_read": r.get("in_workbook_previously_read", ""),
                    "n_pages": doc["n_pages"],
                    "words": words,
                    "est_input_tokens": est_tokens,
                    "sections": doc["sections"],
                    "quality_status": status,
                    "quality_flags": flags,
                    "extraction": {
                        "tool": "pymupdf",
                        "tool_version": fitz.__doc__.strip() if fitz.__doc__ else "",
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "source_pdf": str(src),
                        "tokens_per_word_heuristic": TOKENS_PER_WORD,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        stats[status] += 1
        report.append({
            "paper_id": paper_id, "status": status, "flags": ";".join(flags),
            "pages": doc["n_pages"], "words": words, "sections": len(doc["sections"]),
            "est_input_tokens": est_tokens,
            "title": (r.get("corpus_title") or r.get("zotero_title", ""))[:120],
            "pdf_path": str(src),
        })
        if i % 25 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] {paper_id} {status} ({words} words)")

    rep_path = out / "extraction_report.csv"
    if report:
        with rep_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
            w.writeheader()
            w.writerows(report)

    n_ok = stats["good"] + stats["check"]
    est_in = int(total_words * TOKENS_PER_WORD)
    est_out = n_ok * 2000
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "extracted": dict(stats),
        "total_words": total_words,
        "mean_words_per_paper": round(total_words / n_ok, 1) if n_ok else 0,
        "est_input_tokens_total": est_in,
        "est_output_tokens_total": est_out,
        "cost_preview_usd": {
            "note": "estimate only; assumes ~2,000 output tokens per paper",
            "gpt-5.4-mini_live": round(est_in / 1e6 * 0.75 + est_out / 1e6 * 4.50, 2),
            "gpt-5.4-mini_batch_50pct": round((est_in / 1e6 * 0.75 + est_out / 1e6 * 4.50) / 2, 2),
            "gpt-4.1-nano_live": round(est_in / 1e6 * 0.10 + est_out / 1e6 * 0.40, 2),
        },
        "notes": [
            "Token counts use a words*1.35 heuristic; tiktoken is not installed.",
            "Read extraction_report.csv before launching any coding run.",
            "status poor = do not code; status check = inspect before trusting.",
        ],
    }
    (out / "extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- extraction ---")
    for k, v in sorted(stats.items()):
        print(f"{k:>20}: {v}")
    print(f"{'mean words/paper':>20}: {summary['mean_words_per_paper']}")
    print(f"{'est input tokens':>20}: {est_in:,}")
    print("\ncost preview (USD):")
    for k, v in summary["cost_preview_usd"].items():
        if k != "note":
            print(f"  {k:>28}: {v}")
    print(f"\nreport  -> {rep_path}")
    print(f"summary -> {out / 'extraction_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
