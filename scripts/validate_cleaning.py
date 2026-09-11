#!/usr/bin/env python3
"""Stage 4 validation: compare raw extraction against cleaned output, per journal.

The point is to stop tuning cleaning rules from spot checks. Patching a rule
because one paper looked wrong tells you nothing about whether the fix
generalises, and it cannot detect substance that was deleted silently, which is
the failure mode that matters. A mid-document reference block once caused whole
Discussion and Conclusion sections to be dropped, and that surfaced only because
a human happened to read the file.

Ground truth is data/interim/fulltext/raw/<paper_id>.txt, the untouched PyMuPDF
output. Every sentence there either survived into the cleaned Markdown or did
not, and each casualty is classified:

  SUBSTANTIVE_LOST   prose with a finite verb, 8+ words, alphabetic  -> ERROR,
                     this is argument text we destroyed
  furniture_removed  publisher apparatus                            -> correct
  reference_removed  a bibliography entry                           -> correct

and the reverse direction:

  furniture_retained furniture still present in the clean text       -> miss

Results aggregate BY JOURNAL, because furniture is a publisher property: Emerald,
Springer, Wiley and INFORMS each mangle differently. Per-journal precision and
recall is what tells you whether the rules generalise or merely fit the papers
you happened to open.

Read-only. Writes only under --out-dir. Calls no API.

    python scripts/validate_cleaning.py --limit 20     # quick pass
    python scripts/validate_cleaning.py                # full 131
    python scripts/validate_cleaning.py --journal "Technovation"
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

DEFAULT_RAW = "data/interim/fulltext"
DEFAULT_CLEAN = "data/interim/fulltext_clean"
DEFAULT_MANIFEST = "data/interim/fulltext_prep/fulltext_manifest.csv"
DEFAULT_OUT = "data/interim/fulltext_clean/validation"

# Signals that a removed line was apparatus rather than argument.
FURNITURE_RE = re.compile(
    r"""(?ix)
    (?: https?:// | www\. | doi\.org | \borcid\b | \bissn\b | \bisbn\b
      | [\w.%+-]+@[\w.-]+\.\w{2,}
      | © | \(c\)\s*\d{4} | all\s+rights\s+reserved
      | downloaded\s+from | this\s+content\s+downloaded | jstor
      | \bvol\.?\s*\d+ | \bno\.?\s*\d+ | \bpp\.\s*\d+
      | springer | elsevier | emerald | wiley | sage\s+publications
      | senior\s+editor | associate\s+editor
      | electronic\s+copy\s+available
    )"""
)
# A bibliography entry: authors, a year in brackets, then a source.
REFERENCE_RE = re.compile(
    r"^[A-Z][A-Za-z'’\-]+,?\s+(?:[A-Z]\.\s*)+.*\(?(?:19|20)\d{2}[a-z]?\)?[.,]"
)
_SQUASH = re.compile(r"[^a-z0-9]")

# Apparatus prose: acknowledgements, declarations, funding notes, licence terms
# and special-issue calls for papers. All are intended removals.
_BACK_MATTER_PROSE_RE = re.compile(
    r"""(?ix)
    (?: acknowledg(?:e?ments?|e)\b
      | declaration\s+of\s+(?:competing|conflicting)\s+interest
      | conflicts?\s+of\s+interest
      | \bfunding\b.{0,40}\b(?:received|supported|provided|grant)
      | the\s+authors?\s*\(?s?\)?\s+received\s+no\s+financial
      | data\s+availability
      | ethics\s+approval
      | credit\s+authorship
      | we\s+(?:sincerely\s+)?thank\s+(?:the\s+)?(?:authors|reviewers|editors)
      | manuscripts?\s+will\s+undergo
      | double-?blind\s+peer\s+review\s+process
      | special\s+issue\s+(?:submission|deadline|call)
      | the\s+terms\s+on\s+which\s+this\s+article
      | posting\s+of\s+the\s+accepted\s+manuscript
    )"""
)


def squash(s: str) -> str:
    return _SQUASH.sub("", (s or "").lower())


# Furniture is routinely interleaved INTO the middle of a sentence, e.g.
# "This study further highlights 1 3 82 Page 4 of 21 <journal> (2025) 21:82 The
# integration...". Removing it legitimately changes the sentence's raw form, so
# whole-sentence substring matching reports the prose as lost when it is present.
# Splitting on digit runs isolates the surviving prose segments.
_SEGMENT_SPLIT_RE = re.compile(r"\s*\d[\d\s:().,/\-]*\s*")


def sentence_survives(sent: str, clean_sq: str, min_chars: int = 40) -> bool:
    """True if any substantial prose segment of the sentence is in the clean text.

    Matching per segment rather than per sentence is what separates real
    deletions from cosmetic differences caused by removing embedded furniture
    or by splitting a heading that was fused into the sentence during
    extraction.
    """
    sq_whole = squash(sent)
    if len(sq_whole) >= 20 and sq_whole in clean_sq:
        return True
    for segment in _SEGMENT_SPLIT_RE.split(sent):
        sq = squash(segment)
        if len(sq) >= min_chars and sq in clean_sq:
            return True
    # Sliding window. Digit-delimited segments miss the common case where the
    # difference is at the START or END of the sentence: extraction fuses a
    # heading into the following sentence ("Research agenda Building on the
    # themes..."), and cleaning correctly separates or removes that heading, so
    # the raw sentence carries a prefix the clean text does not have. Any long
    # interior run surviving verbatim means the prose was kept.
    if len(sq_whole) >= min_chars:
        step = 20
        for start in range(0, len(sq_whole) - min_chars + 1, step):
            if sq_whole[start:start + min_chars] in clean_sq:
                return True
    return False


def load_spacy():
    try:
        import spacy
    except ImportError:
        raise SystemExit("spaCy required: pip install spacy && "
                         "python -m spacy download en_core_web_sm")
    try:
        # NER is not needed here; POS is, for the finite-verb test.
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        raise SystemExit("python -m spacy download en_core_web_sm")


_BACK_MATTER_HEADING_RE = re.compile(
    r"^\s*(?:references|bibliography|works\s+cited)\s*:?\s*$", re.I | re.M)


def back_matter_offset(raw_text: str) -> int:
    """Character offset where the bibliography begins, or len(text) if none.

    Everything after it is apparatus. Without this, spaCy splits each reference
    entry into sentences and the TITLE of a cited paper looks exactly like
    prose: eight or more words, no URL, and gerunds ("Investigating...",
    "Navigating...") that tag as verbs. Those were being reported as destroyed
    argument text and were inflating the error rate.
    """
    matches = list(_BACK_MATTER_HEADING_RE.finditer(raw_text))
    return matches[-1].start() if matches else len(raw_text)


def classify_removed(sent: str, doc, offset: int = 0, back_matter_at: int = 10**9) -> str:
    """Why did this sentence disappear? SUBSTANTIVE_LOST is the error case."""
    t = sent.strip()
    words = t.split()
    if offset >= back_matter_at:
        return "reference_removed"
    # Back-matter prose is removed on purpose. It contains finite verbs
    # ("We sincerely thank the reviewers...", "The author(s) received no
    # financial support..."), so the prose test alone marks it as destroyed
    # argument text and inflates the error rate.
    if _BACK_MATTER_PROSE_RE.search(t[:120]):
        return "back_matter_prose"
    if len(words) < 8:
        return "short_fragment"
    if FURNITURE_RE.search(t):
        return "furniture_removed"
    if REFERENCE_RE.match(t):
        return "reference_removed"
    letters = sum(c.isalpha() for c in t)
    if letters / max(len(t), 1) < 0.6:
        return "numeric_or_table"
    # Prose has a finite verb. Citation strings and affiliation lines do not.
    has_verb = any(tok.pos_ in ("VERB", "AUX") for tok in doc)
    if not has_verb:
        return "verbless_line"
    return "SUBSTANTIVE_LOST"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=DEFAULT_RAW)
    ap.add_argument("--clean-dir", default=DEFAULT_CLEAN)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--journal", help="restrict to one journal (substring match)")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir) / "raw"
    clean_dir = Path(args.clean_dir) / "md"
    if not raw_dir.is_dir() or not clean_dir.is_dir():
        raise SystemExit(f"Need {raw_dir} and {clean_dir}. Run extract then clean first.")

    journals: dict[str, str] = {}
    mpath = Path(args.manifest)
    if mpath.is_file():
        with mpath.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                pid = (r.get("paper_id") or "").replace("eid:", "")
                if pid:
                    journals[pid] = (r.get("corpus_source_title")
                                     or r.get("zotero_journal") or "UNKNOWN").strip()

    files = sorted(clean_dir.glob("*.md"))
    if args.journal:
        files = [f for f in files if args.journal.lower() in journals.get(f.stem, "").lower()]
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No files to validate.")

    nlp = load_spacy()
    print(f"Validating {len(files)} papers against raw extraction (spaCy sentence + POS)")

    rows, examples = [], []
    per_journal = defaultdict(lambda: Counter())
    journal_papers = defaultdict(set)
    furniture_kept_examples = defaultdict(Counter)
    totals: Counter = Counter()

    for i, cpath in enumerate(files, 1):
        pid = cpath.stem
        rpath = raw_dir / f"{pid}.txt"
        if not rpath.is_file():
            continue
        journal = journals.get(pid, "UNKNOWN")
        journal_papers[journal].add(pid)

        raw_text = rpath.read_text(encoding="utf-8", errors="replace")
        clean_text = cpath.read_text(encoding="utf-8", errors="replace")
        clean_sq = squash(clean_text)

        # Sentence-segment the raw extraction. Cap length: a few PDFs are huge.
        raw_doc = nlp(raw_text[:600000])
        bm_at = back_matter_offset(raw_text)
        counts: Counter = Counter()
        for sent in raw_doc.sents:
            s = re.sub(r"\s+", " ", sent.text).strip()
            if not s or len(s) < 12:
                continue
            counts["raw_sentences"] += 1
            if len(squash(s)) < 20:
                continue
            if sentence_survives(s, clean_sq):
                counts["kept"] += 1
                # A miss only if the FURNITURE ITSELF is still in the clean
                # text. Testing the whole sentence counted false misses, since
                # segment matching lets a sentence "survive" on its prose half
                # while its furniture half was correctly removed.
                fm = FURNITURE_RE.search(s)
                if fm and sent.start_char < bm_at:
                    window = s[max(0, fm.start() - 30): fm.end() + 30]
                    if len(squash(window)) >= 20 and squash(window) in clean_sq:
                        counts["furniture_retained"] += 1
                        furniture_kept_examples[journal][window[:120]] += 1
                continue
            kind = classify_removed(s, sent.as_doc(), sent.start_char, bm_at)
            counts[kind] += 1
            if kind == "SUBSTANTIVE_LOST" and len(examples) < 4000:
                examples.append({"paper_id": pid, "journal": journal, "sentence": s[:300]})

        for k, v in counts.items():
            per_journal[journal][k] += v
            totals[k] += v

        raw_words = len(raw_text.split())
        clean_words = len(clean_text.split())
        rows.append({
            "paper_id": pid, "journal": journal,
            "raw_words": raw_words, "clean_words": clean_words,
            "pct_words_removed": round(100 * (raw_words - clean_words) / max(raw_words, 1), 2),
            "raw_sentences": counts["raw_sentences"], "kept": counts["kept"],
            "substantive_lost": counts["SUBSTANTIVE_LOST"],
            "furniture_removed": counts["furniture_removed"],
            "reference_removed": counts["reference_removed"],
            "furniture_retained": counts["furniture_retained"],
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(files)}]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "validation_by_paper.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "substantive_lost.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["paper_id", "journal", "sentence"])
        w.writeheader()
        w.writerows(examples)

    jrows = []
    for journal, c in per_journal.items():
        kept, lost = c["kept"], c["SUBSTANTIVE_LOST"]
        jrows.append({
            "journal": journal, "papers": len(journal_papers[journal]),
            "raw_sentences": c["raw_sentences"],
            "substantive_lost": lost,
            "substantive_lost_pct": round(100 * lost / max(c["raw_sentences"], 1), 2),
            "furniture_retained": c["furniture_retained"],
            "furniture_removed": c["furniture_removed"],
            "reference_removed": c["reference_removed"],
        })
    jrows.sort(key=lambda r: -r["substantive_lost_pct"])
    with (out_dir / "validation_by_journal.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(jrows[0].keys()))
        w.writeheader()
        w.writerows(jrows)

    (out_dir / "validation_summary.json").write_text(json.dumps({
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "papers": len(rows), "totals": dict(totals),
        "substantive_lost_pct": round(
            100 * totals["SUBSTANTIVE_LOST"] / max(totals["raw_sentences"], 1), 3),
    }, indent=2), encoding="utf-8")

    print("\n=== overall ===")
    print(f"{'raw sentences':>22}: {totals['raw_sentences']:,}")
    print(f"{'kept':>22}: {totals['kept']:,}")
    print(f"{'SUBSTANTIVE LOST':>22}: {totals['SUBSTANTIVE_LOST']:,}  "
          f"({100*totals['SUBSTANTIVE_LOST']/max(totals['raw_sentences'],1):.2f}%)  <-- errors")
    print(f"{'references removed':>22}: {totals['reference_removed']:,}   (correct)")
    print(f"{'furniture removed':>22}: {totals['furniture_removed']:,}   (correct)")
    print(f"{'furniture RETAINED':>22}: {totals['furniture_retained']:,}   <-- misses")

    print("\n=== by journal (worst substantive loss first) ===")
    print(f"{'papers':>6} {'lost%':>7} {'lost':>6} {'furnKept':>9}  journal")
    for r in jrows[:18]:
        print(f"{r['papers']:>6} {r['substantive_lost_pct']:>7} {r['substantive_lost']:>6} "
              f"{r['furniture_retained']:>9}  {r['journal'][:44]}")

    print(f"\n=== examples of SUBSTANTIVE text we deleted (first {args.examples}) ===")
    for e in examples[: args.examples]:
        print(f"  [{e['journal'][:26]:<26}] {e['sentence'][:100]}")

    print("\n=== furniture still present, by journal ===")
    for journal, c in sorted(furniture_kept_examples.items(),
                             key=lambda kv: -sum(kv[1].values()))[:8]:
        print(f"\n  {journal[:50]}  ({sum(c.values())} lines)")
        for line, n in c.most_common(5):
            print(f"      {n:>3}x  {line[:96]}")

    print(f"\nwrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
