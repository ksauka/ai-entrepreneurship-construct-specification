#!/usr/bin/env python3
"""Stage 4: clean extracted Markdown by removing publisher furniture.

Separate from extract_fulltext.py on purpose. Extraction parses PDFs and is
slow (minutes for the corpus); cleaning is pure text and runs in seconds, so
the rules here can be retuned repeatedly without re-parsing a single PDF.

What it removes, and why extraction cannot:
  - running heads such as "EJIM" (the journal acronym) or the journal name;
  - volume/issue stamps such as "24,2" and bare page numbers such as "554";
  - access furniture such as "Downloaded from ... on 19 December 2025";
  - copyright, ISSN, DOI-URL and permission lines.

Extraction drops lines that repeat IDENTICALLY across pages. That misses page
numbers (different every page) and alternating verso/recto running heads, which
is why a pattern-based pass is needed after it.

Inputs  (from extract_fulltext.py):
    <in-dir>/md/<paper_id>.md
    <in-dir>/json/<paper_id>.json
Outputs:
    <out-dir>/md/<paper_id>.md        cleaned Markdown
    <out-dir>/json/<paper_id>.json    updated sidecar
    <out-dir>/cleaning_report.csv     per-paper removal counts
    <out-dir>/removed_lines.csv       every removed line, for auditing the rules
    <out-dir>/cleaning_summary.json

Non-mutating with respect to its inputs, and calls no API.

Typical use:

    python scripts/clean_fulltext.py --dry-run          # preview, writes nothing
    python scripts/clean_fulltext.py --show 2-s2.0-85067243640
    python scripts/clean_fulltext.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_IN = "data/interim/fulltext"
DEFAULT_OUT = "data/interim/fulltext_clean"
DEFAULT_MANIFEST = "data/interim/fulltext_prep/fulltext_manifest.csv"
TOKENS_PER_WORD = 1.35

_STOP_IN_TITLE = {"of", "the", "and", "for", "in", "on", "a", "an", "&"}

# Publisher furniture. Anchored patterns match a WHOLE line; the unanchored
# group matches anywhere, for stamps that wrap unpredictably.
_NOISE_LINE_RE = re.compile(
    r"""(?ix)
    ^\s*(?:
        \d{1,4}                                    # bare page number
      | [ivxlc]{1,6}                               # roman page number
      | \d{1,3}\s*[,/]\s*\d{1,3}                   # "24,2" volume,issue
      | (?:vol\.?|volume|no\.?|issue)\s*\d+.*
      | pp?\.?\s*\d+\s*[-–]\s*\d+.*
      | on\s+\d{1,2}\s+\w+\s+\d{4}
      | at\s+\d{1,2}[:.]\d{2}.*
      | (?:received|revised|accepted|published|available\s+online)\s*:?\s*\d.*
      | issn.*
      | e-?mail\s*:.*
      | https?://\S*
      | www\.\S*
      | doi\s*:.*
      | \d{4}\s+(?:the\s+authors?|elsevier|springer|emerald|wiley|sage|taylor).*
      | (?:keywords?|jel\s+classification|jel\s+codes?)\b.*
      | (?:accepted|received|revised|published\s+online|available\s+online)\s*:?\s*.*
      | \#?\s*springer\s+science.*
      | /?published\s+online.*
      | [A-Z][a-z]+\s+[A-Z][a-z]*\s*\(\*\)\s*$          # author line "J. C. Kaminski (*)"
      | (?:school|department|faculty|college|institute|university)\s+of\s+.*
    )\s*$
    """
)

# Substring patterns. Applied ONLY to short lines: a Scopus abstract is written
# as one long line and routinely ends "© 2022 The Authors", so allowing an
# unanchored match to delete a whole line would destroy entire paragraphs for
# containing a copyright mark. Furniture is short; argument text is not.
_NOISE_SUBSTR_RE = re.compile(
    r"""(?ix)
    (?:downloaded\s+from
      | this\s+content\s+downloaded
      | all\s+use\s+subject\s+to
      | journal\s+homepage
      | copyright\s+holder
      | for\s+individual\s+use
      | express\s+written\s+permission
      | see\s+discussions,\s*stats
      | researchgate
      | publisher.s\s+note
      | ©\s*\d{4}
      | \(c\)\s*\d{4}
      | all\s+rights\s+reserved
      # Publisher licence and copyright lines, measured by validate_cleaning.py
      # as the residue surviving across Elsevier, Wiley and Springer titles.
      | published\s+by\s+(?:elsevier|emerald|wiley|springer|sage|john\s+wiley|taylor)
      | this\s+is\s+an\s+open\s+access\s+article
      | creativecommons\.org
      | see\s+the\s+terms\s+and\s+conditions
      | onlinelibrary\.wiley\.com
      | the\s+authors?\(?s?\)?,?\s+under\s+exclusive\s+licen[cs]e
      | science\s*\+\s*business\s+media
      | open\s+access\s+funding\s+provided
      | corrected\s+publication\s+\d{4}
      | retrieved\s+from\s+https?://
    )
    """
)

# A bibliography entry, detected by SHAPE rather than by sitting under a
# "References" heading. Some journals (Technovation among them) yield no
# detectable heading, so their reference lists survived section-based removal.
# Matching the entry form generalises across publishers instead of needing a
# rule per journal.
_REFERENCE_ENTRY_RE = re.compile(
    r"""(?ix)
    ^\s*(?:
        # "Surname, A.B., 2019. Title. Journal 12 (3), 45-67."
        [A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*){1,4}.*\b(?:19|20)\d{2}\b
        # "Surname, A. B. (2019). Title..."
      | [A-Z][A-Za-z'’\-]+,\s*(?:[A-Z]\.\s*){1,4}\((?:19|20)\d{2}[a-z]?\)
        # Trailing fragment of an entry: "94, 164-184. https://doi.org/10.1016/..."
      | \d{1,4}\s*[,(].{0,80}?\bdoi\b
      | \d{1,4},\s*\d{1,4}[-–]\d{1,4}\.\s*https?://
    )
    """
)
_REFERENCE_MAX_WORDS = 60

# URL-dense lines: footnote runs and link tables such as
#   "2 https://pypi.org/project/afinn/. 3 https://www.prettyscale.com/. 4 ..."
#   "Image and video Vision: https:// cloud.google. com/vision?hl= en Video: ..."
# A whole-line URL rule misses these because prose surrounds the links. Density
# generalises across publishers instead of needing a rule per journal.
# Note the spaces inside "https:// cloud.google. com" — PDF extraction breaks
# URLs, so the pattern tolerates whitespace after the scheme.
_URL_TOKEN_RE = re.compile(r"https?:\s*//\s*\S+|www\.\s*\S+", re.I)


# Furniture that reflow glues INTO a paragraph. Dropping the whole paragraph
# would take the surrounding prose with it, so these spans are excised in place.
# Each matches the complete stamp, not just its opening phrase.
_EXCISE_RES = (
    re.compile(r"See\s+the\s+Terms\s+and\s+Conditions\s*\(?https?://\S*\)?"
               r"[^.]*?(?:rules\s+of\s+use|Wiley\s+Online\s+Library)[^.]*\.?", re.I),
    re.compile(r"\d*\s*[A-Z][A-Z\s]{2,}\s+ET\s+AL\.\s*\d+x?,\s*\d+,\s*"
               r"Downloaded\s+from\s+\S+[^.]*", re.I),
    re.compile(r"Downloaded\s+from\s+https?://\S+[^.]*", re.I),
    re.compile(r"\d\s+\d\s+Page\s+\d+\s+of\s+\d+\s+\d*\s*[A-Z][A-Za-z ]{5,60}"
               r"\(\d{4}\)\s*\d+:\d+", re.I),          # Springer running head
    re.compile(r"©\s*\d{4}[^.]{0,120}(?:reserved|Limited|Ltd|Inc\.?|Nature)\.?", re.I),
    # Publisher page-footer stamps measured by validate_cleaning.py across the
    # full 131. Each is cut in fragments rather than as one long span, so a
    # partial match never swallows adjacent prose.
    re.compile(r"See\s+the\s+Terms\s+and\s+Conditions\s*\(?\s*https?://\S+?\)?(?=\s|$)", re.I),
    re.compile(r"on\s+Wiley\s+Online\s+Library\s+for\s+rules\s+of\s+use[^.]{0,80}\.?", re.I),
    re.compile(r"OA\s+articles\s+are\s+governed\s+by[^.]{0,120}\.?", re.I),
    re.compile(r"wileyonlinelibrary\.com/journal/\w+", re.I),
    re.compile(r"\d{6,}x?,\s*\d{1,4},\s*\d*,?\s*Downloaded\s+from\s+\S+", re.I),
    re.compile(r"For\s+personal\s+use\s+only,\s*all\s+rights\s+reserved\.?", re.I),
    re.compile(r"Downloaded\s+from\s+informs\.org\s+by\s*\[?[\d.\s]{0,30}\]?", re.I),
    re.compile(r"©\s*\d{4}\s+Society\s+for\s+the\s+Advancement[^.]{0,60}\.?", re.I),
    re.compile(r"published\s+by\s+Wiley\s+Periodicals\s+LLC[^.]{0,60}\.?", re.I),
    re.compile(r"\S*\s*Journal\s+published\s+by\s+John\s+Wiley\s*&?\s*Sons\s+Ltd"
               r"[^.]{0,60}\.?", re.I),
)


def excise_furniture(text: str) -> tuple[str, int]:
    """Cut furniture spans out of a paragraph, keeping the prose around them.

    Publisher stamps are interleaved mid-sentence by PDF reading order (Wiley's
    terms notice appears 37 times in one paper, Springer's running head lands
    inside sentences). Removing the paragraph would destroy argument text, and
    leaving it retains furniture, so the span itself is removed.
    """
    out, n = text, 0
    for rx in _EXCISE_RES:
        out, k = rx.subn(" ", out)
        n += k
    return re.sub(r"\s{2,}", " ", out).strip(), n


def url_density(text: str) -> tuple[int, float]:
    """Return (number of URLs, share of characters that are URL text)."""
    hits = _URL_TOKEN_RE.findall(text)
    return len(hits), sum(len(h) for h in hits) / max(len(text), 1)
_NOISE_SUBSTR_MAX_WORDS = 25

# Unambiguous furniture: these phrases never occur in argument text, so they
# apply at ANY line length. Wiley stamps such as
#   "4 BEZ ET AL. 1932443x, 0, Downloaded from https://sms.onlinelibrary..."
# exceed the 25-word cap and were surviving because of it.
_NOISE_ALWAYS_RE = re.compile(
    r"""(?ix)
    (?: downloaded\s+from
      | this\s+content\s+downloaded
      | see\s+the\s+terms\s+and\s+conditions
      | onlinelibrary\.wiley\.com
      | creativecommons\.org
      | this\s+is\s+an\s+open\s+access\s+article
      | published\s+by\s+(?:elsevier|emerald|wiley|springer|sage|john\s+wiley|taylor)
      | the\s+authors?\(?s?\)?,?\s+under\s+exclusive\s+licen[cs]e
      | open\s+access\s+funding\s+provided
      | all\s+use\s+subject\s+to
      | electronic\s+copy\s+available\s+at
    )"""
)

# Page-1 sidebar matter that PyMuPDF's reading order interleaves into the body,
# common in INFORMS journals (Organization Science, ISR). It splits sections and
# injects contact, ORCID and editorial lines into the argument text.
_FRONT_MATTER_RE = re.compile(
    r"""(?ix)
    ^\s*(?:
        (?:contact|history|supplemental\s+material|corresponding\s+author
          |funding\s+information|author\s+contributions|data\s+availability)\s*:.*
      | .*\borcid\.org\b.*
      | .*\b[\w.%+-]+@[\w.-]+\.\w{2,}\b.*                    # any line with an email
      | .*\b(?:senior|associate|handling|area)\s+editor\b.*
      | .*\baccepted\s+by\b.*
      | (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}
      | \d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}
    )\s*$
    """
)

# Affiliation lines. Case-SENSITIVE on the word after "of" so that
# "School of Business" matches but "school of thought" does not.
_AFFILIATION_RE = re.compile(
    r"\b(?:School|Department|Faculty|College|Institute|Centre|Center|University)\b"
    r"[,.]?\s+(?:of\s+)?[A-Z]"
)

# Section names that legitimately recur and must never be deleted as a title
# echo, even when the paper's own title happens to contain the same words.
# Apparatus, not argument. Dropped whole: a bibliography must not be embedded as
# retrievable content or read as the paper's claims, and it is a large share of
# every document's tokens. The unclean copy under data/interim/fulltext/ keeps
# them, so nothing is lost, only excluded from the analysis-ready text.
NON_CONTENT_SECTIONS = {
    "references", "reference", "bibliography", "works cited",
    "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
    "funding", "funding information", "declarations", "declaration",
    "declaration of competing interest", "declaration of conflicting interests",
    "conflict of interest", "conflicts of interest", "competing interests",
    "credit authorship contribution statement", "author contributions",
    "data availability", "data availability statement", "ethics statement",
    "ethical approval", "supplementary material", "supplementary materials",
    "author biographies", "about the authors", "author information",
    "disclosure statement", "notes on contributors", "orcid",
}

# Prefix forms, because real headings carry suffixes: "Appendix A",
# "References and notes", "Acknowledgements and funding".
NON_CONTENT_PREFIXES = (
    "references", "reference list", "bibliography", "works cited",
    "acknowledg", "declaration", "funding", "conflict of interest",
    "competing interest", "data availability", "supplementary",
    "author contribution", "credit authorship", "about the author",
    "author biograph", "author information", "notes on contributor",
    "disclosure statement", "ethics", "ethical approval", "orcid",
)

# A bibliography marks the start of the back matter. Everything after it is
# apparatus, so the tail is dropped wholesale. Dropping only the section titled
# "References" leaves the rest of the list behind whenever a junk heading (a DOI
# fragment, say) splits it into several sections.
BACK_MATTER_STARTS = ("references", "reference list", "bibliography", "works cited")

# Sections that are legitimately long. Everything else on the non-content list
# is a one-or-two-sentence statement, so a large body means the heading has
# ABSORBED real text and the section must not be dropped wholesale. Observed:
# a "Competing interests" section carrying 353 words, which was the paper's
# Conclusion.
LEGITIMATELY_LARGE = ("references", "reference list", "bibliography",
                      "works cited", "appendix")
ABSORBED_CONTENT_WORDS = 150

# Canonical body sections. Their appearance CANCELS back-matter mode.
# Reference blocks routinely surface mid-document (two-column reading order, or
# journals that place them before appendices). Without this escape, a sticky
# back-matter flag deletes every section after such a block, including Methods,
# Discussion and Conclusion. Discussion is where mechanism statements live, so
# that failure silently guts the coding input.
CONTENT_RESUME_PREFIXES = (
    "introduction", "background", "literature", "related work", "theory",
    "theoretical", "conceptual", "hypothes", "method", "methodolog", "data",
    "research design", "sample", "measure", "analys", "result", "finding",
    "discussion", "implication", "limitation", "conclusion", "concluding",
    "robustness", "empirical", "model", "estimation", "validation",
)

# Headings that are really table cells, DOI fragments or link lines. The
# extractor cannot always tell these from real headings, so they are removed
# here and their text promoted into the section above.
_JUNK_HEADING_RE = re.compile(
    r"""(?ix)
    ^\s*(?:
        [\(\[]?[A-Z0-9][A-Z0-9\-_.]{0,6}[\)\]]?          # (XAI), Q3, B2C, SMT, ML
      | [\d./]+                                          # 10.1108/..., 5.
      | .*\bdoi\b.*
      | .*https?://.*
      | (?:to\s+link\s+to\s+this\s+article).*
      | [A-Z]{1,4}-\d{2}-\d{4}-\d{3,}                    # IM-02-2024-0154
      | /?10\.\d{4,}/.*                                  # bare DOI
    )\s*$
    """
)

_CANONICAL_HEADINGS = {
    "abstract", "keywords", "introduction", "background", "literature review",
    "related work", "theory", "theoretical background", "conceptual framework",
    "method", "methods", "methodology", "data", "research design", "analysis",
    "results", "findings", "discussion", "implications", "limitations",
    "conclusion", "conclusions", "references", "appendix",
}
_AFFILIATION_MAX_WORDS = 30

# Leading section number, so "3. References" normalises to "references".
_LEAD_NUM_RE = re.compile(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[IVXLC]{1,5}[.)]\s*)")

# Author bylines carrying affiliation superscripts, e.g.
# "Elina H. Hwang,a Param Vir Singh,b Linda Argoteb" or "Nadia Zahoora,b,*".
_AUTHOR_BYLINE_RE = re.compile(
    r"^(?:[A-Z][A-Za-z.'’\-]+\s+){0,3}[A-Z][A-Za-z'’\-]+[a-z]?"
    r"(?:,\s*[a-z](?:,\s*[a-z])*)?\s*,?\s*\*?\s*(?:,|$)"
)
_AUTHOR_SEMI_RE = re.compile(
    r"^[A-Z][A-Za-z'’\-]+ [A-Z]\.(?:[A-Z]\.)*(?:\s*;\s*[A-Z][A-Za-z'’\-]+ [A-Z]\.)+"
)

# Level 0 is the "#" paper title (the document root); "##" is level 1.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def journal_signature(journal: str) -> tuple[set[str], str]:
    """Return (significant title words, acronym) for a journal name.

    The acronym is what catches running heads like "EJIM" for the European
    Journal of Innovation Management, which no generic pattern would recognise.
    """
    words = [w for w in re.findall(r"[A-Za-z]+", journal or "") if w]
    sig = {w.lower() for w in words if w.lower() not in _STOP_IN_TITLE and len(w) > 2}
    acronym = "".join(w[0] for w in words if w.lower() not in _STOP_IN_TITLE).upper()
    return sig, acronym


def is_noise_line(text: str, jsig: set[str], jacronym: str) -> tuple[bool, str]:
    """Return (is_noise, reason). Reason makes the rules auditable."""
    t = text.strip()
    if not t:
        return False, ""
    if _NOISE_LINE_RE.match(t):
        return True, "pattern"
    if _NOISE_ALWAYS_RE.search(t):
        return True, "pattern"
    if len(t.split()) <= _NOISE_SUBSTR_MAX_WORDS and _NOISE_SUBSTR_RE.search(t):
        return True, "pattern"
    if _FRONT_MATTER_RE.match(t):
        return True, "front_matter"
    # Stray bibliography entries outside a detected References section.
    if len(t.split()) <= _REFERENCE_MAX_WORDS and _REFERENCE_ENTRY_RE.match(t):
        return True, "reference_entry"
    n_urls, url_share = url_density(t)
    if n_urls >= 2 or (n_urls >= 1 and url_share >= 0.35):
        return True, "url_dense"
    if len(t.split()) <= _AFFILIATION_MAX_WORDS and _AFFILIATION_RE.search(t):
        return True, "affiliation"
    # Author bylines: short, capitalised, with affiliation superscripts or a
    # semicolon-separated Scopus-style list. Length-capped so prose is safe.
    if len(t.split()) <= 12 and (_AUTHOR_SEMI_RE.match(t) or
                                 (re.search(r",\s*[a-z]\s*(?:,|\*|$)", t)
                                  and _AUTHOR_BYLINE_RE.match(t))):
        return True, "author_line"
    letters = re.sub(r"[^A-Za-z]", "", t)
    if letters and len(jacronym) >= 3 and letters.upper() == jacronym:
        return True, "journal_acronym"
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", t)]
    if words and len(words) <= 6 and jsig and all(w in jsig for w in words):
        return True, "journal_name"
    return False, ""


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;\"'<>()\[\]]+", re.I)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
# Most entries begin "Surname, A." or "Surname, Firstname" after a full stop.
_REF_SPLIT_RE = re.compile(
    r"(?<=[.\)])\s+(?=(?:\[\d+\]\s*|\d+\.\s+)?[A-Z][A-Za-z'’\-]{1,}\s*,\s*(?:[A-Z]\.|[A-Z][a-z]+))"
)


def parse_references(lines: list[str]) -> list[dict]:
    """Split a bibliography block into entries with an identifier for each.

    References are dropped from the body text but retained here as metadata, so
    the knowledge graph can link papers through shared citations. Each entry
    gets a `ref_key`: its DOI when one is present, otherwise a fingerprint over
    first author, year and the entry text. The fingerprint matches only
    near-identical strings, so cross-journal formatting differences will still
    split some shared references; DOI-based keys are the reliable ones and
    resolving the rest is deliberately left as later work.
    """
    text = re.sub(r"\s+", " ", " ".join(ln.strip() for ln in lines if ln.strip())).strip()
    if not text:
        return []

    refs, seen = [], set()
    for part in _REF_SPLIT_RE.split(text):
        part = part.strip()
        if len(part) < 30:
            continue
        doi_m, year_m = _DOI_RE.search(part), _YEAR_RE.search(part)
        if not (doi_m or year_m):
            continue  # not a reference entry
        doi = doi_m.group(0).rstrip(".,;)") if doi_m else ""
        year = year_m.group(0)[:4] if year_m else ""
        author_m = re.match(r"(?:\[\d+\]\s*|\d+\.\s+)?([A-Z][A-Za-z'’\-]+)", part)
        first_author = author_m.group(1) if author_m else ""
        if doi:
            key = "doi:" + doi.lower()
        else:
            squashed = re.sub(r"[^a-z0-9]", "", part.lower())[:80]
            key = "fp:" + hashlib.sha1(
                (first_author.lower() + year + squashed).encode("utf-8")
            ).hexdigest()[:16]
        if key in seen:
            continue
        seen.add(key)
        refs.append({
            "ref_key": key, "doi": doi, "year": year,
            "first_author": first_author, "raw": part[:1000],
        })
    return refs


def load_spacy():
    """Load spaCy. REQUIRED, not optional.

    It does two jobs no regex does well: repairing paragraph breaks that split a
    sentence after reflow, and recognising author/affiliation bylines by entity
    density rather than by a fixed punctuation shape.
    """
    try:
        import spacy
    except ImportError:
        raise SystemExit(
            "spaCy is required by this stage.\n"
            "  pip install spacy && python -m spacy download en_core_web_sm"
        )
    try:
        return spacy.load("en_core_web_sm", disable=["lemmatizer"])
    except OSError:
        raise SystemExit(
            "spaCy model en_core_web_sm is missing.\n"
            "  python -m spacy download en_core_web_sm"
        )


def reflow(lines: list[str], nlp) -> str:
    """Join PDF column-wrapped lines back into paragraphs.

    PDF extraction preserves the physical line breaks of a typeset column, so
    every line is an 8-to-12 word fragment. That is unreadable, cuts RAG chunks
    mid-sentence, and gives the coder fragments instead of prose.

    A line break is a real paragraph break only when all three hold: the line
    stops short of the column margin, it ends a sentence, and the next line
    starts like a fresh one. Everything else is a soft wrap and is joined.
    spaCy then repairs any break that still splits a sentence.
    """
    kept = [ln.strip() for ln in lines if ln.strip()]
    if not kept:
        return ""

    median_len = statistics.median([len(ln) for ln in kept]) if kept else 0
    paras: list[str] = []
    current: list[str] = []

    for i, line in enumerate(kept):
        nxt = kept[i + 1] if i + 1 < len(kept) else None
        if current and current[-1].endswith("-") and re.match(r"^[a-z]", line):
            current[-1] = current[-1][:-1] + line          # rejoin hyphenated word
        else:
            current.append(line)

        if nxt is None:
            break
        ends_sentence = bool(re.search(r"[.!?][\"'’)\]]*$", line))
        stops_short = len(line) < median_len * 0.92
        next_is_fresh = bool(re.match(r"^[A-Z•·\-\d(\"'“]", nxt))
        running_words = sum(len(x.split()) for x in current)

        # Primary signal: the line stops short of the margin, completes a
        # sentence, and the next line starts fresh.
        is_break = ends_sentence and stops_short and next_is_fresh
        # Fallback cap. Requiring all three conditions misses roughly half of
        # real breaks, which produced 170-word average paragraphs. Once a block
        # is already longer than a real paragraph, any completed sentence
        # followed by a fresh start is treated as a break. This bounds the
        # failure instead of leaving whole sections fused into one block.
        if not is_break and running_words >= 140 and ends_sentence and next_is_fresh:
            is_break = True

        if is_break:
            paras.append(" ".join(current))
            current = []

    if current:
        paras.append(" ".join(current))

    # Repair breaks that split a sentence: if a paragraph does not end on
    # terminal punctuation, it was cut mid-sentence and belongs with the next.
    repaired: list[str] = []
    for para in paras:
        text = re.sub(r"\s+", " ", para).strip()
        if not text:
            continue
        # Merge forward ONLY a short dangling fragment. Merging any paragraph
        # that lacks terminal punctuation cascades: academic paragraphs often
        # end on a citation, "...(Dahl et al. 2015)", whose final character is a
        # bracket, and one bad merge then swallows the whole section (observed
        # at 3,560 words). Allow trailing brackets and quotes after the stop.
        prev_complete = bool(repaired and re.search(r"[.!?][\"'’)\]]*$", repaired[-1]))
        prev_short = bool(repaired and len(repaired[-1].split()) < 30)
        if repaired and prev_short and not prev_complete:
            repaired[-1] = repaired[-1] + " " + text
        else:
            repaired.append(text)
    return "\n\n".join(repaired)


def count_sentences(text: str, nlp) -> int:
    """Sentence count via spaCy, used as a readability diagnostic."""
    if not text.strip():
        return 0
    total = 0
    for chunk in text.split("\n\n"):
        if chunk.strip():
            total += sum(1 for _ in nlp(chunk[:100000]).sents)
    return total


def is_byline_ner(text: str, nlp) -> bool:
    """True for a short line that is mostly names/organisations and has no verb.

    Regexes catch known byline shapes; this catches the ones they miss, because
    an author or affiliation line is characterised by being entity-dense and
    verbless rather than by any fixed punctuation pattern.
    """
    if nlp is None:
        return False
    t = text.strip()
    if not t or len(t.split()) > 20:
        return False
    doc = nlp(t)
    if any(tok.pos_ == "VERB" for tok in doc):
        return False
    ent_chars = sum(len(e.text) for e in doc.ents
                    if e.label_ in ("PERSON", "ORG", "GPE", "FAC", "NORP"))
    return ent_chars / max(len(t), 1) >= 0.55


def _flatten(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For echo matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def _squash(s: str) -> str:
    """Lowercase alphanumerics only, no spaces at all.

    Scopus abstracts routinely lose spaces on export ("aremore likely",
    "shallowknowledge", "outperformnongeneralists") while the PDF has them.
    Removing whitespace entirely makes echo matching immune to that, and to
    line-wrap and hyphenation differences.
    """
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def build_echo_matchers(meta: dict) -> dict:
    """Authoritative title/abstract/keywords, flattened and space-free."""
    corpus = meta.get("corpus_metadata", {})
    raw = {
        "title": meta.get("title", ""),
        "abstract": corpus.get("Abstract", "") or meta.get("abstract", ""),
        "keywords": corpus.get("Author Keywords", "") or meta.get("keywords", ""),
    }
    return {
        "flat": {k: _flatten(v) for k, v in raw.items()},
        "sq": {k: _squash(v) for k, v in raw.items()},
    }


def is_metadata_echo(text: str, echo: dict, min_words: int = 6,
                     is_heading: bool = False) -> tuple[bool, str]:
    """True when a line merely repeats title/abstract/keywords we already hold.

    The PDF front matter re-supplies the title, the whole abstract and the
    keyword list, and running heads repeat the title on every page. Because the
    corpus already gives us those authoritatively, any line that is literally a
    fragment of them is redundant and can go. Matching is exact-substring after
    flattening, so only genuine repeats are removed, never paraphrase.
    """
    flat = _flatten(text)
    if not flat:
        return False, ""
    # "Abstract This paper introduces..." -> drop the label before comparing.
    flat = re.sub(r"^abstract\s+", "", flat)
    sq = _squash(flat)

    # A heading that merely repeats part of the paper title is a running head or
    # a sidebar fragment, not a section. Canonical section names are exempt so a
    # genuine "Methods" survives in a paper whose title contains that word.
    if is_heading and echo["flat"]["title"]:
        if (flat not in _CANONICAL_HEADINGS
                and len(flat.split()) >= 2
                and flat in echo["flat"]["title"]):
            return True, "title_echo"

    if len(flat.split()) < min_words:
        # Short lines only match the title, where a short repeat is a running head.
        if echo["flat"]["title"] and flat in echo["flat"]["title"] and len(flat.split()) >= 3:
            return True, "title_echo"
        return False, ""

    for key in ("abstract", "title", "keywords"):
        if echo["flat"][key] and flat in echo["flat"][key]:
            return True, f"{key}_echo"
    # Space-insensitive fallback for exported text with lost spaces. The length
    # floor keeps this from matching on incidental character runs.
    if len(sq) >= 40:
        for key in ("abstract", "title", "keywords"):
            if echo["sq"][key] and sq in echo["sq"][key]:
                return True, f"{key}_echo"
    return False, ""


def parse_md(text: str) -> list[dict]:
    """Split Markdown back into sections, preserving heading level."""
    sections: list[dict] = []
    current = {"level": 2, "title": "", "lines": []}
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in-dir", default=DEFAULT_IN)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help="used only to look up each paper's journal name")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--show", metavar="PAPER_ID", help="print removed lines for one paper and exit")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop-appendix", action="store_true",
                    help="drop Appendix sections (kept by default: code, tables, robustness checks)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    md_dir = in_dir / "md"
    if not md_dir.is_dir():
        raise SystemExit(f"No Markdown in {md_dir}. Run extract_fulltext.py first.")

    journals: dict[str, str] = {}
    mpath = Path(args.manifest)
    if mpath.is_file():
        with mpath.open(encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                pid = (r.get("paper_id") or "").replace("eid:", "")
                if pid:
                    journals[pid] = r.get("corpus_source_title") or r.get("zotero_journal") or ""

    files = sorted(md_dir.glob("*.md"))
    if args.show:
        files = [f for f in files if f.stem == args.show] or files[:0]
        if not files:
            raise SystemExit(f"No Markdown for paper {args.show}")
    elif args.limit:
        files = files[: args.limit]

    if not args.dry_run and not args.show:
        (out_dir / "md").mkdir(parents=True, exist_ok=True)
        (out_dir / "json").mkdir(parents=True, exist_ok=True)

    nlp = load_spacy()
    print("spaCy loaded (en_core_web_sm): paragraph repair + byline NER active")
    report, removed_rows = [], []
    reason_counts: Counter = Counter()
    tot_before = tot_after = 0
    total_refs = refs_with_doi = 0

    for path in files:
        pid = path.stem
        journal = journals.get(pid, "")
        jsig, jacronym = journal_signature(journal)
        sidecar = in_dir / "json" / f"{pid}.json"
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        echo = build_echo_matchers(meta)
        sections = parse_md(path.read_text(encoding="utf-8"))

        # Identify the authoritative prefix by POSITION, not by title text.
        # extract_fulltext.py writes the corpus title, Abstract and Keywords at
        # the head of every document. Trusting the title text instead would let
        # any PDF line that parses as a heading (a "#"-mangled copyright line,
        # or the paper's own "Abstract" heading) claim authority and exempt the
        # duplicated front matter beneath it from cleaning.
        expected_title = _flatten(meta.get("title", ""))
        auth_idx = set()
        for i, sec in enumerate(sections):
            t = sec["title"].strip()
            is_root = sec["level"] == 0 and expected_title and _flatten(t) == expected_title
            is_meta_sec = sec["level"] == 1 and t in ("Abstract", "Keywords")
            if is_root or is_meta_sec:
                auth_idx.add(i)
            else:
                break

        kept_sections, n_removed, n_lines = [], 0, 0
        dropped_non_content = False
        in_back_matter = False
        ref_lines: list[str] = []
        for sec_i, sec in enumerate(sections):
            norm_title = _flatten(_LEAD_NUM_RE.sub("", sec["title"])).strip()

            # Apparatus sections are dropped entirely, body included. Everything
            # after a References heading is back matter too, so once we are in
            # that tail we keep dropping unless a genuine content section
            # reappears (some journals place an Appendix with real material
            # after the bibliography).
            is_non_content = (
                norm_title in NON_CONTENT_SECTIONS
                or any(norm_title.startswith(p) for p in NON_CONTENT_PREFIXES)
            )
            if norm_title.startswith(BACK_MATTER_STARTS) and sec["level"] >= 1:
                in_back_matter = True
            elif norm_title.startswith(CONTENT_RESUME_PREFIXES):
                # A genuine body section after a reference block means that
                # block was mid-document, not the start of the back matter.
                in_back_matter = False
            # Appendices are kept by default: they carry analysis code, extra
            # tables and robustness checks that are genuine content.
            if not args.drop_appendix and norm_title.startswith("appendix"):
                in_back_matter = False
                is_non_content = False

            # Drop apparatus sections, and everything in the back-matter tail,
            # body included. This check runs BEFORE the junk-heading rule so a
            # DOI-fragment heading inside a bibliography cannot promote its
            # reference text back into the preceding content section.
            section_words = len(" ".join(sec["lines"]).split())
            absorbed_content = (
                is_non_content
                and not in_back_matter
                and not norm_title.startswith(LEGITIMATELY_LARGE)
                and section_words > ABSORBED_CONTENT_WORDS
            )
            if absorbed_content:
                # Drop the heading, KEEP the body. A short-statement heading
                # carrying hundreds of words has swallowed argument text, and
                # dropping it wholesale destroys a Conclusion or Discussion.
                n_removed += 1
                reason_counts["absorbed_content_heading"] += 1
                removed_rows.append({
                    "paper_id": pid, "journal": journal,
                    "reason": "absorbed_content_heading",
                    "line": f"[heading only] {sec['title'][:60]} ({section_words}w body kept)",
                })
                sec = {"level": sec["level"], "title": "", "lines": sec["lines"]}
                is_non_content = False

            if (is_non_content or in_back_matter) and sec["level"] >= 1:
                words = section_words
                n_removed += 1
                reason = "back_matter_tail" if (in_back_matter and not is_non_content) \
                    else "non_content_section"
                reason_counts[reason] += 1
                # Record a body excerpt, not just the title. Logging only a
                # summary row let a dropped "Competing interests" section hide
                # 353 words of Conclusion from every diagnostic that searched
                # removed_lines.csv for the missing sentences.
                excerpt = re.sub(r"\s+", " ", " ".join(sec["lines"]))[:220]
                removed_rows.append({
                    "paper_id": pid, "journal": journal, "reason": reason,
                    "line": f"[dropped section] {sec['title'][:60]} ({words}w) :: {excerpt}",
                })
                # Keep the text of bibliography sections for metadata even
                # though it leaves the body: the graph links papers by citation.
                if norm_title.startswith(BACK_MATTER_STARTS) or reason == "back_matter_tail":
                    ref_lines.extend(sec["lines"])
                dropped_non_content = True
                continue

            # A second "Abstract" or "Keywords" heading is the PDF's own copy;
            # the authoritative one already sits at the head of the document.
            # Drop the heading but KEEP its body: such sections routinely absorb
            # the introduction (observed at 4,810 words), so echo filtering
            # should remove the duplicated abstract text and leave the rest.
            duplicate_meta_heading = (
                norm_title in ("abstract", "keywords") and sec_i not in auth_idx
            )

            # Junk headings (table cells, DOI fragments, link lines) lose the
            # heading but keep their text, merged into the section above.
            if sec["title"] and (duplicate_meta_heading
                                 or _JUNK_HEADING_RE.match(sec["title"].strip())):
                n_removed += 1
                reason = "duplicate_meta_heading" if duplicate_meta_heading else "junk_heading"
                reason_counts[reason] += 1
                removed_rows.append({
                    "paper_id": pid, "journal": journal, "reason": reason,
                    "line": "#" * (sec["level"] + 1) + " " + sec["title"][:80],
                })
                if kept_sections:
                    kept_sections[-1]["lines"].extend(sec["lines"])
                    continue
                sec = {"level": sec["level"], "title": "", "lines": sec["lines"]}

            title_noise, title_reason = is_noise_line(sec["title"], jsig, jacronym)
            if not title_noise:
                title_noise, title_reason = is_metadata_echo(sec["title"], echo, is_heading=True)
            # Only the corpus-supplied prefix is authoritative and exempt.
            authoritative = sec_i in auth_idx
            if authoritative:
                title_noise = False
            # The corpus-supplied title, Abstract and Keywords are the
            # authoritative record and are kept verbatim. spec-v3 coded exactly
            # this string, copyright suffix included, so altering it would
            # change the input relative to the abstract-level arm.
            if authoritative:
                body = "\n".join(sec["lines"]).strip()
                kept_sections.append({"level": sec["level"], "title": sec["title"],
                                      "lines": [body] if body else []})
                continue

            def _drop(text: str, reason: str) -> None:
                nonlocal n_removed
                n_removed += 1
                reason_counts[reason] += 1
                removed_rows.append({"paper_id": pid, "journal": journal,
                                     "reason": reason, "line": text.strip()[:200]})

            # PASS 1, line level: remove only SHORT furniture. This must precede
            # reflow, or a page number would be glued into the middle of a
            # sentence when wrapped lines are joined.
            pass1: list[str] = []
            for line in sec["lines"]:
                t = line.strip()
                if not t:
                    continue
                n_lines += 1
                if len(t.split()) <= 8:
                    noise, reason = is_noise_line(t, jsig, jacronym)
                    if not noise:
                        noise, reason = is_metadata_echo(t, echo)
                    if noise:
                        _drop(t, reason)
                        continue
                pass1.append(line)

            # PASS 2: rebuild paragraphs from the surviving wrapped lines.
            body = reflow(pass1, nlp)

            # PASS 3, paragraph level. Furniture that wraps across two PDF lines
            # is only detectable once rejoined, and echo matching against the
            # abstract is far more reliable on a whole paragraph.
            kept_paras: list[str] = []
            for para in body.split("\n\n"):
                p = para.strip()
                if not p:
                    continue
                # Excise interleaved stamps first. Only then judge the
                # remainder, so a paragraph is never discarded for containing
                # furniture that could simply be cut out of it.
                excised, n_cut = excise_furniture(p)
                if n_cut and len(excised.split()) >= 15:
                    _drop(p[:200], "furniture_excised")
                    p = excised
                noise, reason = is_noise_line(p, jsig, jacronym)
                if not noise:
                    noise, reason = is_metadata_echo(p, echo)
                if not noise and sec_i <= 4 and is_byline_ner(p, nlp):
                    noise, reason = True, "byline_ner"
                if noise:
                    _drop(p, reason)
                else:
                    kept_paras.append(p)
            body = "\n\n".join(kept_paras).strip()
            if title_noise:
                n_removed += 1
                reason_counts[title_reason + "_heading"] += 1
                removed_rows.append({"paper_id": pid, "journal": journal,
                                     "reason": title_reason + "_heading",
                                     "line": "## " + sec["title"][:190]})
                # A furniture heading may still cover real text; keep the body
                # by promoting it into the previous section rather than losing it.
                if body and kept_sections:
                    kept_sections[-1]["lines"].append(body)
                continue
            if not body and not sec["title"]:
                continue
            kept_sections.append({"level": sec["level"], "title": sec["title"],
                                  "lines": [body] if body else []})

        parts = []
        for sec in kept_sections:
            body = "\n".join(sec["lines"]).strip()
            if not body and not sec["title"]:
                continue
            if sec["title"]:
                parts.append("{} {}\n\n".format("#" * (sec["level"] + 1), sec["title"]))
            if body:
                parts.append(body + "\n\n")
        cleaned = "".join(parts).strip() + "\n"

        before_words = len(path.read_text(encoding="utf-8").split())
        after_words = len(cleaned.split())
        tot_before += before_words
        tot_after += after_words

        if args.show:
            print(f"=== {pid}  journal={journal!r}  acronym={jacronym!r} ===")
            print(f"lines removed: {n_removed} of {n_lines}"
                  f"   words {before_words} -> {after_words}\n")
            for r in removed_rows:
                print(f"  [{r['reason']:<18}] {r['line']}")
            return 0

        refs = parse_references(ref_lines)
        total_refs += len(refs)
        refs_with_doi += sum(1 for r in refs if r["doi"])

        # Readability check: before reflow a "paragraph" was one wrapped line of
        # ~10 words. Real prose paragraphs run 40 to 120 words, so this number
        # is the direct evidence that reflow worked.
        body_paras = [p for p in cleaned.split("\n\n")
                      if p.strip() and not p.lstrip().startswith("#")]
        mean_para_words = round(
            sum(len(p.split()) for p in body_paras) / max(len(body_paras), 1), 1)

        report.append({
            "paper_id": pid, "journal": journal[:60],
            "paragraphs": len(body_paras),
            "mean_words_per_paragraph": mean_para_words,
            "references": len(refs),
            "references_with_doi": sum(1 for r in refs if r["doi"]),
            "lines_total": n_lines, "lines_removed": n_removed,
            "pct_removed": round(100 * n_removed / n_lines, 2) if n_lines else 0,
            "words_before": before_words, "words_after": after_words,
            "words_dropped": before_words - after_words,
            "sections_before": len(sections), "sections_after": len(kept_sections),
            "est_input_tokens": int(after_words * TOKENS_PER_WORD),
        })

        if not args.dry_run:
            (out_dir / "md" / f"{pid}.md").write_text(cleaned, encoding="utf-8")
            src_json = in_dir / "json" / f"{pid}.json"
            meta = json.loads(src_json.read_text(encoding="utf-8")) if src_json.is_file() else {}
            meta["words"] = after_words
            meta["est_input_tokens"] = int(after_words * TOKENS_PER_WORD)
            meta["sections"] = [
                {"title": s["title"], "level": s["level"],
                 "words": len("\n".join(s["lines"]).split())}
                for s in kept_sections
            ]
            meta["references"] = refs
            meta["reference_count"] = len(refs)
            meta["references_with_doi"] = sum(1 for r in refs if r["doi"])
            meta["cleaning"] = {
                "cleaned_at": datetime.now(timezone.utc).isoformat(),
                "lines_removed": n_removed,
                "words_before": before_words,
                "words_after": after_words,
                "journal_acronym": jacronym,
            }
            (out_dir / "json" / f"{pid}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8")

    if not report:
        raise SystemExit("Nothing processed.")

    if not args.dry_run:
        with (out_dir / "cleaning_report.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
            w.writeheader()
            w.writerows(report)
        if removed_rows:
            with (out_dir / "removed_lines.csv").open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(removed_rows[0].keys()))
                w.writeheader()
                w.writerows(removed_rows)
        (out_dir / "cleaning_summary.json").write_text(json.dumps({
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "papers": len(report),
            "words_before": tot_before, "words_after": tot_after,
            "pct_words_dropped": round(100 * (tot_before - tot_after) / max(tot_before, 1), 2),
            "removal_reasons": dict(reason_counts),
        }, indent=2), encoding="utf-8")

    pct = 100 * (tot_before - tot_after) / max(tot_before, 1)
    print(("DRY RUN, nothing written\n" if args.dry_run else "") + "--- cleaning ---")
    print(f"{'papers':>22}: {len(report)}")
    print(f"{'words before':>22}: {tot_before:,}")
    print(f"{'words after':>22}: {tot_after:,}")
    print(f"{'dropped':>22}: {pct:.2f}%")
    all_para_means = [r["mean_words_per_paragraph"] for r in report if r["paragraphs"]]
    print(f"{'mean words/paragraph':>22}: "
          f"{round(statistics.mean(all_para_means), 1) if all_para_means else 0}"
          "   (pre-reflow this was ~10; real prose is 40-120)")
    print(f"{'references captured':>22}: {total_refs:,} "
          f"({refs_with_doi:,} with DOI, {100*refs_with_doi/max(total_refs,1):.0f}%)")
    print("\nremoval reasons:")
    for k, v in reason_counts.most_common():
        print(f"  {k:>22}: {v}")

    worst = sorted(report, key=lambda r: -r["pct_removed"])[:8]
    print("\nhighest removal rate (check these for over-cleaning):")
    for r in worst:
        print(f"  {r['pct_removed']:>6}%  {r['paper_id']:<22} {r['journal'][:38]}")

    # Guard rail. A mid-document reference block once latched back-matter mode
    # and deleted Methods, Discussion and Conclusion from a paper. That must
    # never pass unnoticed again.
    suspect = [r for r in report if r["pct_removed"] > 25]
    if suspect:
        print(f"\n*** WARNING: {len(suspect)} paper(s) lost more than 25% of words. "
              "Inspect before using; check removed_lines.csv for dropped sections "
              "with content-sounding titles (Discussion, Results, Methods). ***")
        for r in suspect:
            print(f"  {r['pct_removed']:>6}%  {r['paper_id']}")

    long_paras = [r for r in report if r.get("mean_words_per_paragraph", 0) > 250]
    if long_paras:
        print(f"\nNOTE: {len(long_paras)} paper(s) average over 250 words per paragraph, "
              "which suggests reflow is still merging paragraphs.")
    if not args.dry_run:
        print(f"\nwrote -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
