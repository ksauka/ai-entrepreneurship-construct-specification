"""Stage 2A.5: LLM-assisted AI specification coding (OpenAI-compatible API).

Runs directly on master_corpus.csv: coding is per paper and query-invariant,
so it is independent of topic modeling. Scopes only matter later, at
analysis time, when the coded columns are contrasted across the query views.

Setup for real runs (one-time): create .env in the project root (gitignored):
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4.1-nano       # optional override

Usage (from the project root, graphrag env):
    python scripts/run_specification.py --dry-run            # count + cost estimate
    python scripts/run_specification.py --local --limit 5    # free test via Ollama
    python scripts/run_specification.py --limit 25           # optional smoke check
    python scripts/run_specification.py                      # full corpus for this model
    python scripts/run_specification.py --scope query_3      # one query view only

Agreed cost ladder (2026-07-10): test end-to-end on local Ollama (free),
Every study model is run over the full corpus and preserved independently for
inter-rater reliability and five-scope comparison. Limited runs are plumbing
checks only; they are not a model-selection tournament.

Coding is cached per paper AND per model in data/interim/spec_cache/<model>/,
so local test codings never contaminate the final run, interrupted runs
resume for free, and recoding only happens if the cache file is deleted.

Outputs:
    data/processed/specification/paper_specifications_<model>.csv
    data/processed/specification/specification_report_<model>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.scopes import SCOPE_BY_ID, scope_frame  # noqa: E402
from aecsp.progress import ProgressReporter, format_duration  # noqa: E402
from aecsp.specification.llm_coder import (  # noqa: E402
    DEFAULT_MODEL,
    cache_key,
    code_paper,
    load_env,
    model_cache_dir,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPEC_DIR = PROCESSED_DIR / "specification"
CACHE_ROOT = PROJECT_ROOT / "data" / "interim" / "spec_cache"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LOCAL_MODEL = "llama3.2"

# (USD per 1M input tokens, USD per 1M output tokens) for the --dry-run
# estimate; unknown models fall back to gpt-4o-mini rates, local runs are $0.
MODEL_PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
}
EST_INPUT_TOKENS = 1400  # system prompt + dimension briefing + abstract
EST_OUTPUT_TOKENS = 700  # structured profile with evidence + confidence


def load_corpus() -> pd.DataFrame:
    path = PROCESSED_DIR / "master_corpus.csv"
    print(f"Loading corpus: {path.name}", flush=True)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        default="full_corpus",
        choices=sorted(SCOPE_BY_ID),
        help="Papers to code (default: full_corpus so every query view is covered).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max papers this run.")
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL/.env.")
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Use local Ollama at {OLLAMA_BASE_URL} (free testing; model "
        f"defaults to {DEFAULT_LOCAL_MODEL}, check 'ollama list').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Estimate only; no API calls.")
    args = parser.parse_args()

    env = load_env(PROJECT_ROOT / ".env")
    if args.local:
        base_url = env.get("OLLAMA_BASE_URL") or OLLAMA_BASE_URL
        api_key = "ollama"  # Ollama ignores the key but the client requires one
        model = args.model or env.get("OLLAMA_MODEL") or DEFAULT_LOCAL_MODEL
    else:
        base_url = env.get("OPENAI_BASE_URL")  # None -> api.openai.com
        api_key = env.get("OPENAI_API_KEY")
        model = args.model or env.get("OPENAI_MODEL") or DEFAULT_MODEL
    cache_dir = model_cache_dir(CACHE_ROOT, model)

    master = load_corpus()
    scoped = scope_frame(master, args.scope)
    todo = scoped[~scoped["paper_id"].map(lambda pid: (cache_dir / cache_key(pid)).exists())]
    cached = len(scoped) - len(todo)
    if args.limit:
        todo = todo.head(args.limit)
    if args.local:
        est_line = "est. cost: $0.00 (local Ollama)"
    else:
        in_price, out_price = MODEL_PRICES.get(model, MODEL_PRICES["gpt-4o-mini"])
        est_cost = len(todo) * (
            EST_INPUT_TOKENS * in_price + EST_OUTPUT_TOKENS * out_price
        ) / 1_000_000
        est_line = f"est. cost ({model}): ${est_cost:.2f}"
    print(f"Scope '{args.scope}': {len(scoped):,} papers | cached for {model}: "
          f"{cached:,} | to code now: {len(todo):,} | {est_line}", flush=True)

    if args.dry_run:
        return

    if not args.local and not api_key:
        sys.exit("OPENAI_API_KEY missing. Create .env in the project root "
                 "(see module docstring), or test free with --local.")

    from openai import OpenAI  # imported late so --dry-run works without a key

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Coding {len(todo):,} papers with {model}"
          f"{' via ' + base_url if base_url else ''}...", flush=True)

    started = time.time()
    failures: list[str] = []
    progress = ProgressReporter("Specification", len(todo), every=1, started=started)
    if todo.empty:
        progress.update(0, detail="all papers cached", force=True)
    for done, (_, row) in enumerate(todo.iterrows(), start=1):
        paper = {
            "paper_id": row["paper_id"],
            "title": row.get("Title", ""),
            "abstract": row.get("Abstract", ""),
            "keywords": row.get("Author Keywords", ""),
            "journal": row.get("Source title", ""),
            "year": row.get("Year", ""),
        }
        print(
            f"\n  START {done:,}/{len(todo):,} | {paper['paper_id']} | "
            f"{paper['title'][:90]}",
            flush=True,
        )
        paper_started = time.time()
        try:
            code_paper(client, model, paper, cache_dir)
            outcome = "DONE"
        except Exception as error:  # keep going; failures are re-tried next run
            failures.append(f"{paper['paper_id']}: {error}")
            outcome = "FAILED"
            print(f"  FAILED {paper['paper_id']}: {error}", flush=True)
        print(
            f"  {outcome} {paper['paper_id']} in "
            f"{format_duration(time.time() - paper_started)}",
            flush=True,
        )
        progress.update(
            done,
            failures=len(failures),
            detail=paper["paper_id"],
        )

    # Assemble the full coded dataset for this scope from the cache.
    print("Assembling coded dataset from cache...", flush=True)
    records = []
    for pid in scoped["paper_id"]:
        cache_path = cache_dir / cache_key(pid)
        if cache_path.exists():
            records.append(json.loads(cache_path.read_text(encoding="utf-8")))
    coded = pd.DataFrame(records)

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = cache_dir.name
    out_path = SPEC_DIR / f"paper_specifications_{model_slug}.csv"
    coded.to_csv(out_path, index=False, encoding="utf-8-sig")

    report = {
        "timestamp": datetime.now().isoformat(),
        "scope": args.scope,
        "model": model,
        "base_url": base_url,
        "papers_in_scope": len(scoped),
        "coded_total": len(coded),
        "coded_this_run": len(todo) - len(failures),
        "failures": failures,
        "runtime_seconds": round(time.time() - started, 1),
    }
    with open(
        SPEC_DIR / f"specification_report_{model_slug}.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    print(f"Done: {len(coded):,} coded papers -> {out_path}")
    if failures:
        print(f"{len(failures)} failures (will retry automatically on next run)")


if __name__ == "__main__":
    main()
