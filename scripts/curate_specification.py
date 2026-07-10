"""Stage 2A.5 curation UI: review LLM-coded dimensions, per dimension.

Usage (from the project root, graphrag env):

    python scripts/curate_specification.py --model llama3.2            # review queue
    python scripts/curate_specification.py --model llama3.2 --report   # status counts only
    python scripts/curate_specification.py --model llama3.2 --export   # write curated CSV
    python scripts/curate_specification.py --model llama3.2 --threshold 0.7

How it works (agreed 2026-07-10):
- Codes with evidence_type 'stated' and confidence >= threshold (default 0.8)
  are auto-accepted; only the rest queue for review.
- The queue is grouped by dimension, least confident first, so the reader
  works through one dimension's judgment at a time.
- Decisions are saved to data/processed/specification/
  curation_overrides_<model>.json after every keystroke (resumable); the LLM
  cache under data/interim/spec_cache/<model>/ is never modified.

Review commands:
    Enter / a   accept the LLM code
    1..N        override with the numbered allowed value
    e           show the paper's abstract
    s           skip (stays llm_unreviewed)
    d           defer this DIMENSION corpus-wide (e.g. needs full text)
    B           batch-accept every remaining queued item of this dimension
    q           save and quit

Outputs:
    data/processed/specification/curation_overrides_<model>.json
    data/processed/specification/paper_specifications_curated.csv  (--export)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.specification.curation import (  # noqa: E402
    DEFAULT_ACCEPT_THRESHOLD,
    build_review_queue,
    curated_frame,
    load_coded_records,
    load_overrides,
    record_decision,
    save_overrides,
    status_report,
)
from aecsp.specification.llm_coder import model_cache_dir  # noqa: E402

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPEC_DIR = PROCESSED_DIR / "specification"
CACHE_ROOT = PROJECT_ROOT / "data" / "interim" / "spec_cache"

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"


def resolve_cache_dir(model: str | None) -> tuple[Path, str]:
    """Cache dir for --model, or the only coded model when unambiguous."""

    available = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir()) if CACHE_ROOT.exists() else []
    if model:
        cache_dir = model_cache_dir(CACHE_ROOT, model)
        if not cache_dir.exists():
            sys.exit(f"No cache for model '{model}'. Coded models: {available or 'none'}")
        return cache_dir, cache_dir.name
    if len(available) == 1:
        return CACHE_ROOT / available[0], available[0]
    sys.exit(f"Pass --model. Coded models: {available or 'none'}")


def load_paper_lookup(paper_ids: set[str]) -> dict[str, dict[str, str]]:
    """Title and abstract per paper_id, for display during review."""

    master = pd.read_csv(
        PROCESSED_DIR / "master_corpus.csv",
        dtype=str,
        keep_default_na=False,
        usecols=["paper_id", "Title", "Abstract", "Source title", "Year"],
    )
    master = master[master["paper_id"].isin(paper_ids)]
    return {
        row["paper_id"]: {
            "title": row["Title"],
            "abstract": row["Abstract"],
            "journal": row["Source title"],
            "year": row["Year"],
        }
        for _, row in master.iterrows()
    }


def show_item(item: dict, paper: dict[str, str], position: int, total: int) -> None:
    confidence = item["confidence"]
    confidence_str = f"{confidence:.2f}" if confidence is not None else "n/a"
    colour = GREEN if item["evidence_type"] == "stated" else YELLOW
    print(f"\n{BOLD}[{position}/{total}] {item['column']}{RESET}  "
          f"({item['evidence_type'] or 'missing'}, confidence {confidence_str})")
    print(f"{CYAN}{paper.get('title', item['paper_id'])}{RESET}")
    print(f"  {paper.get('journal', '')} ({paper.get('year', '')})  {item['paper_id']}")
    print(f"  Q: {item['question']}")
    print(f"  LLM code: {colour}{item['code'] or '(empty)'}{RESET}")
    print(f"  Evidence: {item['evidence'] or '(none)'}")
    print("  Override options:")
    for index, value in enumerate(item["allowed_values"], start=1):
        print(f"    {index:2d}. {value}")


def review(queue: list[dict], lookup: dict, overrides: dict, overrides_path: Path) -> None:
    total = len(queue)
    decided = 0
    position = 0
    while position < len(queue):
        item = queue[position]
        column = item["column"]
        if column in overrides["dimension_deferrals"]:
            position += 1
            continue
        show_item(item, lookup.get(item["paper_id"], {}), position + 1, total)
        choice = input(f"{BOLD}[Enter=accept, 1-N=override, e=abstract, s=skip, "
                       f"d=defer dim, B=batch-accept dim, q=quit]{RESET} ").strip()

        if choice in ("", "a"):
            record_decision(overrides, item["paper_id"], column, "accept")
            decided += 1
        elif choice.isdigit() and 1 <= int(choice) <= len(item["allowed_values"]):
            new_code = item["allowed_values"][int(choice) - 1]
            record_decision(overrides, item["paper_id"], column, "override", code=new_code)
            print(f"  {GREEN}Overridden to: {new_code}{RESET}")
            decided += 1
        elif choice == "e":
            abstract = lookup.get(item["paper_id"], {}).get("abstract", "(no abstract)")
            print(f"\n{abstract}\n")
            continue  # re-show the same item
        elif choice == "s":
            pass  # stays llm_unreviewed
        elif choice == "d":
            overrides["dimension_deferrals"].append(column)
            print(f"  {YELLOW}Deferred '{column}' corpus-wide.{RESET}")
        elif choice == "B":
            remaining = [q for q in queue[position:] if q["column"] == column]
            for pending in remaining:
                record_decision(overrides, pending["paper_id"], column, "accept")
            decided += len(remaining)
            print(f"  {GREEN}Batch-accepted {len(remaining)} remaining "
                  f"'{column}' items.{RESET}")
        elif choice == "q":
            save_overrides(overrides_path, overrides)
            print(f"Saved. {decided} decisions this session.")
            return
        else:
            print(f"  {RED}Unknown command.{RESET}")
            continue

        save_overrides(overrides_path, overrides)
        position += 1

    save_overrides(overrides_path, overrides)
    print(f"\nQueue complete. {decided} decisions this session.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Coding model whose cache to curate.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD,
                        help="Auto-accept confidence threshold (default 0.8).")
    parser.add_argument("--report", action="store_true", help="Status counts only.")
    parser.add_argument("--export", action="store_true",
                        help="Write paper_specifications_curated.csv and exit.")
    args = parser.parse_args()

    cache_dir, model_slug = resolve_cache_dir(args.model)
    records = load_coded_records(cache_dir)
    if not records:
        sys.exit(f"No coded papers in {cache_dir}")
    overrides_path = SPEC_DIR / f"curation_overrides_{model_slug}.json"
    overrides = load_overrides(overrides_path)
    print(f"Model: {model_slug} | coded papers: {len(records):,} | "
          f"threshold: {args.threshold} | overrides: {overrides_path.name}")

    if args.report or args.export:
        report = status_report(records, overrides, args.threshold)
        print(report.to_string(index=False))
        if args.export:
            frame = curated_frame(records, overrides, args.threshold)
            SPEC_DIR.mkdir(parents=True, exist_ok=True)
            out_path = SPEC_DIR / "paper_specifications_curated.csv"
            frame.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"Curated dataset -> {out_path}")
        return

    queue = build_review_queue(records, args.threshold, overrides)
    if not queue:
        print("Nothing to review: every dimension is auto-accepted, decided, or deferred.")
        return
    print(f"Review queue: {len(queue):,} paper-dimension items "
          f"(grouped by dimension, least confident first).")
    review(queue, load_paper_lookup({item["paper_id"] for item in queue}), overrides, overrides_path)


if __name__ == "__main__":
    main()
