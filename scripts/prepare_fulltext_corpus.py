#!/usr/bin/env python3
"""Stage 5 prep: build a full-text processing manifest from a Zotero library.

Reads a Zotero SQLite library READ-ONLY (via a temporary copy, so it is safe to
run while Zotero is open), extracts the items in one or more collections,
resolves their PDF attachments on disk, and matches them against the canonical
22,345-paper master corpus and the previously-read workbook set.

This script is non-mutating. It never writes to the Zotero library, never
touches data/interim/spec_cache/**, and never calls a paid API. Its only output
is a manifest plus a summary under --out-dir.

Typical use:

    python scripts/prepare_fulltext_corpus.py --list-collections
    python scripts/prepare_fulltext_corpus.py --collection "Project Monika and Frederik"
    python scripts/prepare_fulltext_corpus.py --collection "Project Monika and Frederik" --stage-pdfs

Outputs:
    <out-dir>/fulltext_manifest.csv   one row per Zotero item
    <out-dir>/fulltext_summary.json   coverage counts and provenance
    <out-dir>/pdfs/<paper_id>.pdf     staged copies (only with --stage-pdfs)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

DEFAULT_ZOTERO_DIRS = [
    "/mnt/c/Users/kudzy/Desktop/Zotero",
    "C:/Users/kudzy/Desktop/Zotero",
    str(Path.home() / "Zotero"),
]
DEFAULT_CORPUS = "data/processed/master_corpus.csv"
DEFAULT_WORKBOOK = "data/interim/theory_elaboration/workbook_paper_match_audit.csv"
DEFAULT_OUT = "data/interim/fulltext_prep"

# Zotero attachment linkMode values
LINK_IMPORTED_FILE, LINK_IMPORTED_URL, LINK_LINKED_FILE, LINK_LINKED_URL = 0, 1, 2, 3


# ---------------------------------------------------------------- normalizers

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def norm_doi(value):
    if not value:
        return ""
    s = _DOI_PREFIX.sub("", str(value).strip()).strip().lower()
    return s.rstrip(".")


def norm_title(value):
    if not value:
        return ""
    return _NON_ALNUM.sub(" ", str(value).lower()).strip()


# ------------------------------------------------------------------- zotero io


def open_zotero_readonly(zotero_dir, tmpdir):
    """Copy zotero.sqlite to tmpdir and open it read-only.

    Copying is deliberate: reading the live file while Zotero holds a write lock
    can raise "database is locked", and we must never risk writing to it.
    """
    src = zotero_dir / "zotero.sqlite"
    if not src.is_file():
        raise SystemExit("No zotero.sqlite in {}".format(zotero_dir))
    dst = tmpdir / "zotero_copy.sqlite"
    shutil.copy2(src, dst)
    return sqlite3.connect("file:{}?mode=ro".format(dst.as_posix()), uri=True)


def collection_closure(con, names, all_collections):
    """Return {collectionID: full/path/name} for named collections and descendants."""
    rows = con.execute(
        "SELECT collectionID, collectionName, parentCollectionID FROM collections"
    ).fetchall()
    by_id = {cid: (nm, parent) for cid, nm, parent in rows}

    def path_of(cid):
        parts, seen = [], set()
        while cid in by_id and cid not in seen:
            seen.add(cid)
            nm, parent = by_id[cid]
            parts.append(nm)
            cid = parent
        return "/".join(reversed(parts))

    if all_collections:
        return {cid: path_of(cid) for cid in by_id}

    wanted = {n.strip().lower() for n in names}
    roots = [cid for cid, (nm, _) in by_id.items() if nm.strip().lower() in wanted]
    found = {by_id[c][0].strip().lower() for c in roots}
    missing = wanted - found
    if missing:
        available = sorted(nm for nm, _ in by_id.values())
        raise SystemExit(
            "Collection(s) not found: "
            + ", ".join(sorted(missing))
            + "\nAvailable: "
            + ", ".join(available)
        )
    closure = set(roots)
    changed = True
    while changed:
        changed = False
        for cid, (_, parent) in by_id.items():
            if parent in closure and cid not in closure:
                closure.add(cid)
                changed = True
    return {cid: path_of(cid) for cid in closure}


def load_items(con, collection_ids):
    """Return list of item dicts for regular (non-attachment/note) items."""
    trashed = {r[0] for r in con.execute("SELECT itemID FROM deletedItems")}
    types = dict(
        con.execute("SELECT itemID, typeName FROM items JOIN itemTypes USING(itemTypeID)")
    )

    placeholders = ",".join("?" * len(collection_ids))
    membership = {}
    query = (
        "SELECT collectionID, itemID FROM collectionItems WHERE collectionID IN ({})".format(
            placeholders
        )
    )
    for cid, iid in con.execute(query, tuple(collection_ids)):
        membership.setdefault(iid, set()).add(collection_ids[cid])

    item_ids = {
        i
        for i in membership
        if i not in trashed and types.get(i) not in ("attachment", "note", "annotation")
    }
    if not item_ids:
        return []

    fields = dict(con.execute("SELECT fieldID, fieldName FROM fields"))
    ph = ",".join("?" * len(item_ids))

    data = {}
    for iid, fid, val in con.execute(
        "SELECT itemData.itemID, itemData.fieldID, itemDataValues.value "
        "FROM itemData JOIN itemDataValues USING(valueID) "
        "WHERE itemData.itemID IN ({})".format(ph),
        tuple(item_ids),
    ):
        data.setdefault(iid, {})[fields.get(fid, str(fid))] = val

    creators = {}
    for iid, last, first in con.execute(
        "SELECT itemCreators.itemID, creators.lastName, creators.firstName "
        "FROM itemCreators JOIN creators USING(creatorID) "
        "WHERE itemCreators.itemID IN ({}) "
        "ORDER BY itemCreators.itemID, itemCreators.orderIndex".format(ph),
        tuple(item_ids),
    ):
        creators.setdefault(iid, []).append("{} {}".format(last or "", first or "").strip())

    keys = dict(
        con.execute("SELECT itemID, key FROM items WHERE itemID IN ({})".format(ph), tuple(item_ids))
    )

    # PDF attachments, including the attachment's own key (its storage subfolder)
    atts = {}
    for parent, akey, link_mode, ctype, path in con.execute(
        "SELECT ia.parentItemID, i.key, ia.linkMode, ia.contentType, ia.path "
        "FROM itemAttachments ia JOIN items i ON i.itemID = ia.itemID "
        "WHERE ia.parentItemID IS NOT NULL"
    ):
        is_pdf = ctype == "application/pdf" or str(path or "").lower().endswith(".pdf")
        if parent in item_ids and is_pdf:
            atts.setdefault(parent, []).append(
                {"key": akey, "link_mode": link_mode, "path": path or ""}
            )

    out = []
    for iid in sorted(item_ids):
        d = data.get(iid, {})
        out.append(
            {
                "zotero_item_id": iid,
                "zotero_key": keys.get(iid, ""),
                "item_type": types.get(iid, ""),
                "collections": sorted(membership.get(iid, ())),
                "title": d.get("title", ""),
                "doi": d.get("DOI", ""),
                "year": (d.get("date", "") or "")[:4],
                "journal": d.get("publicationTitle", "") or d.get("proceedingsTitle", ""),
                "authors": "; ".join(creators.get(iid, [])),
                "attachments": atts.get(iid, []),
            }
        )
    return out


def resolve_pdf(zotero_dir, att):
    path = att["path"]
    if att["link_mode"] == LINK_LINKED_FILE and path and not path.startswith("storage:"):
        p = Path(path)
        return p if p.is_file() else None
    if path.startswith("storage:"):
        p = zotero_dir / "storage" / att["key"] / path[len("storage:"):]
        return p if p.is_file() else None
    return None


# ------------------------------------------------------------------ corpus io


def load_corpus(path):
    """Index the master corpus by normalized DOI and normalized title.

    The COMPLETE row is retained, not a hand-picked subset. Downstream stages
    need far more than the matching keys: the knowledge graph builds Author,
    Institution, Journal, Year, Keyword and SearchQuery nodes, and the
    full-text documents reuse the exact title, abstract and author keywords
    that spec-v3 defined as its evidentiary unit. Selecting columns here means
    rebuilding the manifest every time a later stage needs one more field.
    """
    by_doi, by_title, total = {}, {}, 0
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            total += 1
            rec = {k: (v or "") for k, v in row.items()}
            d, t = norm_doi(rec.get("DOI")), norm_title(rec.get("Title"))
            if d:
                by_doi.setdefault(d, rec)
            if t:
                by_title.setdefault(t, rec)
    return by_doi, by_title, total


def load_workbook(path):
    """Return (paper_ids, dois, titles) for the previously-read workbook set."""
    ids, dois, titles = set(), set(), set()
    if not path.is_file():
        return ids, dois, titles
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("paper_id"):
                ids.add(row["paper_id"])
            for key in ("current_doi", "workbook_old_doi"):
                if norm_doi(row.get(key)):
                    dois.add(norm_doi(row[key]))
            for key in ("current_title", "workbook_title"):
                if norm_title(row.get(key)):
                    titles.add(norm_title(row[key]))
    return ids, dois, titles


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--zotero-dir", default=None, help="Zotero data dir (contains zotero.sqlite)")
    ap.add_argument("--collection", action="append", default=[], help="collection name (repeatable)")
    ap.add_argument("--all-collections", action="store_true", help="use every collection")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--workbook", default=DEFAULT_WORKBOOK)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--stage-pdfs", action="store_true", help="copy matched PDFs to <out-dir>/pdfs/")
    ap.add_argument("--stage-unmatched", action="store_true", help="also stage unmatched PDFs")
    ap.add_argument("--hash-pdfs", action="store_true", help="record sha256 per PDF (slower)")
    ap.add_argument("--list-collections", action="store_true", help="print collections and exit")
    args = ap.parse_args()

    candidates = [args.zotero_dir] if args.zotero_dir else DEFAULT_ZOTERO_DIRS
    zdir = None
    for cand in candidates:
        if cand and (Path(cand) / "zotero.sqlite").is_file():
            zdir = Path(cand)
            break
    if zdir is None:
        raise SystemExit("Zotero library not found. Pass --zotero-dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="zotero_ro_") as td:
        con = open_zotero_readonly(zdir, Path(td))

        if args.list_collections:
            everything = collection_closure(con, [], True)
            for cid, path in sorted(everything.items(), key=lambda kv: kv[1]):
                n = con.execute(
                    "SELECT COUNT(*) FROM collectionItems WHERE collectionID=?", (cid,)
                ).fetchone()[0]
                print("{:>6}  {}".format(n, path))
            con.close()
            return 0

        if not args.collection and not args.all_collections:
            raise SystemExit("Pass --collection NAME (repeatable) or --all-collections.")

        cols = collection_closure(con, args.collection, args.all_collections)
        items = load_items(con, cols)
        con.close()

    print("Zotero: {}".format(zdir))
    print("Collections in scope: {} | regular items: {}".format(len(cols), len(items)))

    corpus_path, wb_path = Path(args.corpus), Path(args.workbook)
    if not corpus_path.is_file():
        raise SystemExit("Corpus not found: {}".format(corpus_path))
    by_doi, by_title, corpus_n = load_corpus(corpus_path)
    print("Corpus: {} rows ({} DOIs, {} titles)".format(corpus_n, len(by_doi), len(by_title)))
    wb_ids, wb_dois, wb_titles = load_workbook(wb_path)
    print("Workbook previously-read set: {} paper_ids, {} DOIs".format(len(wb_ids), len(wb_dois)))

    pdf_dir = out_dir / "pdfs"
    if args.stage_pdfs:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # Complete corpus record per matched paper. The manifest stays a readable
    # working index; this carries every field the graph and RAG stages need.
    full_meta = {}
    stats = {
        "items": len(items), "with_pdf": 0, "with_doi": 0,
        "matched": 0, "matched_by_doi": 0, "matched_by_title": 0,
        "matched_with_pdf": 0, "in_workbook": 0, "in_workbook_with_pdf": 0,
        "staged": 0, "ready_for_fulltext": 0,
    }

    for it in items:
        nd, nt = norm_doi(it["doi"]), norm_title(it["title"])
        hit = by_doi.get(nd) if nd else None
        how = "doi" if hit else ""
        if not hit and nt:
            hit = by_title.get(nt)
            how = "title" if hit else ""

        pdf = None
        for att in it["attachments"]:
            pdf = resolve_pdf(zdir, att)
            if pdf:
                break

        in_wb = bool(
            (hit and hit.get("paper_id") in wb_ids)
            or (nd and nd in wb_dois)
            or (nt and nt in wb_titles)
        )

        if pdf:
            stats["with_pdf"] += 1
        if nd:
            stats["with_doi"] += 1
        if hit:
            stats["matched"] += 1
            stats["matched_by_" + how] += 1
            if pdf:
                stats["matched_with_pdf"] += 1
        if in_wb:
            stats["in_workbook"] += 1
            if pdf:
                stats["in_workbook_with_pdf"] += 1

        paper_id = (hit or {}).get("paper_id", "")
        if hit and paper_id:
            full_meta[paper_id] = hit
        staged = ""
        if args.stage_pdfs and pdf and (hit or args.stage_unmatched):
            stem = paper_id.replace("eid:", "") if paper_id else "zotero_" + it["zotero_key"]
            target = pdf_dir / (stem + ".pdf")
            if not target.exists():
                shutil.copy2(pdf, target)
            staged = str(target)
            stats["staged"] += 1

        ready = bool(pdf and hit)
        if ready:
            stats["ready_for_fulltext"] += 1

        rows.append({
            "zotero_key": it["zotero_key"],
            "zotero_item_id": it["zotero_item_id"],
            "item_type": it["item_type"],
            "collections": " | ".join(it["collections"]),
            "zotero_title": it["title"],
            "zotero_authors": it["authors"],
            "zotero_year": it["year"],
            "zotero_journal": it["journal"],
            "zotero_doi": it["doi"],
            "has_pdf": int(bool(pdf)),
            "pdf_path": str(pdf) if pdf else "",
            "pdf_bytes": pdf.stat().st_size if pdf else "",
            "pdf_sha256": sha256(pdf) if (pdf and args.hash_pdfs) else "",
            "staged_pdf_path": staged,
            "match_status": "matched" if hit else "not_in_corpus",
            "match_method": how,
            "paper_id": paper_id,
            "corpus_eid": (hit or {}).get("EID", ""),
            "corpus_doi": (hit or {}).get("DOI", ""),
            "corpus_title": (hit or {}).get("Title", ""),
            "corpus_abstract": (hit or {}).get("Abstract", ""),
            "corpus_keywords": (hit or {}).get("Author Keywords", ""),
            "corpus_year": (hit or {}).get("Year", ""),
            "corpus_source_title": (hit or {}).get("Source title", ""),
            "in_query_1": (hit or {}).get("in_query_1", ""),
            "in_query_2": (hit or {}).get("in_query_2", ""),
            "in_query_3": (hit or {}).get("in_query_3", ""),
            "in_query_4": (hit or {}).get("in_query_4", ""),
            "corpus_relevant": (hit or {}).get("corpus_relevant", ""),
            "ai_ent_relevant": (hit or {}).get("ai_ent_relevant", ""),
            "in_workbook_previously_read": int(in_wb),
            "ready_for_fulltext": int(ready),
        })

    # Deduplicate by paper_id. The same paper is routinely filed as several
    # Zotero records, one per subcollection, so row counts overstate the sample
    # (154 rows were only 131 distinct papers). Exactly one row per paper may be
    # staged, extracted and coded. The duplicates' collection membership is
    # merged into the survivor, since subcollection names are thematic metadata
    # worth keeping.
    by_paper: dict[str, list] = {}
    for r in rows:
        if r["paper_id"]:
            by_paper.setdefault(r["paper_id"], []).append(r)

    duplicates_suppressed = 0
    for pid, group in by_paper.items():
        merged = sorted({c for r in group for c in r["collections"].split(" | ") if c})
        # Prefer a row that actually has a PDF, then the largest file.
        primary = max(group, key=lambda r: (r["has_pdf"], int(r["pdf_bytes"] or 0)))
        for r in group:
            r["collections"] = " | ".join(merged)
            r["is_primary"] = int(r is primary)
            r["duplicate_of"] = "" if r is primary else pid
            if r is not primary:
                r["ready_for_fulltext"] = 0
                r["staged_pdf_path"] = ""
                duplicates_suppressed += 1

    for r in rows:
        r.setdefault("is_primary", 1)
        r.setdefault("duplicate_of", "")

    # Recompute from the deduplicated rows so the counts describe PAPERS.
    stats["ready_for_fulltext"] = sum(r["ready_for_fulltext"] for r in rows)
    stats["staged"] = sum(1 for r in rows if r["staged_pdf_path"])
    stats["unique_papers_matched"] = len(by_paper)
    stats["duplicate_rows_suppressed"] = duplicates_suppressed

    manifest = out_dir / "fulltext_manifest.csv"
    fieldnames = list(rows[0].keys()) if rows else ["zotero_key"]
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    meta_path = out_dir / "paper_metadata.json"
    meta_path.write_text(json.dumps(full_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    corpus_fields = sorted({k for rec in full_meta.values() for k in rec})

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "zotero_dir": str(zdir),
        "paper_metadata": str(meta_path),
        "paper_metadata_records": len(full_meta),
        "corpus_fields_carried": corpus_fields,
        "collections_in_scope": sorted(cols.values()),
        "corpus_path": str(corpus_path),
        "corpus_rows": corpus_n,
        "workbook_path": str(wb_path) if wb_path.is_file() else None,
        "counts": stats,
        "manifest": str(manifest),
        "notes": [
            "Zotero library read from a temporary copy; the live library is never written.",
            "ready_for_fulltext = has a resolvable PDF AND matches the master corpus.",
            "Corpus match is DOI-first, then exact normalized title.",
            "This script performs no coding and calls no API.",
        ],
    }
    (out_dir / "fulltext_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- coverage ---")
    for k, v in stats.items():
        print("{:>28}: {}".format(k, v))
    print("\nmanifest -> {}".format(manifest))
    print("summary  -> {}".format(out_dir / "fulltext_summary.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
