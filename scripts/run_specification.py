"""Run model-specific paper-level AI specification coding.

Inputs: the master corpus, model configuration, and any existing per-model
cache. Outputs: cached JSON records, a coded CSV, and a specification report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.corpus.scopes import SCOPE_BY_ID, scope_frame  # noqa: E402
from aecsp.progress import ProgressReporter, format_duration  # noqa: E402
from aecsp.specification.llm_coder import (  # noqa: E402
    PROTOCOL_ID,
    REQUEST_TIMEOUT_SECONDS,
    SDK_MAX_RETRIES,
    DEFAULT_MODEL,
    cache_key,
    code_paper,
    load_env,
    model_cache_dir,
    protocol_fingerprint,
    protocol_for_model,
    protocol_parameters,
    sanitize_lone_surrogates,
)
from aecsp.specification.analysis_columns import enrich_for_analysis  # noqa: E402

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
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
}
EST_INPUT_TOKENS = 1400  # system prompt + dimension briefing + abstract
EST_OUTPUT_TOKENS = 700  # structured profile with evidence + confidence

# HTTP 429 is transient transport state, never rater non-response: back off
# and retry inside the run instead of writing failures.jsonl noise. Sleeping
# holds the worker slot, so the pool self-paces at the account's TPM ceiling
# (~95 papers/min at 200,000 TPM and ~2,100 tokens per paper). Bounded so a
# genuinely stuck account surfaces as a real failure after a few minutes.
RATE_LIMIT_MAX_ATTEMPTS = 12

CONTENT_FAILURE_MARKERS = (
    "Structured response exceeded",
    "Unterminated string",
    "Expecting value",
    "Expecting property name",
    "Extra data",
    "Invalid JSON",
    "JSON decode",
)


def load_corpus() -> pd.DataFrame:
    path = PROCESSED_DIR / "master_corpus.csv"
    print(f"Loading corpus: {path.name}", flush=True)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def file_sha256(path: Path) -> str:
    """Return a stable checksum for an experiment input file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_failure_counts(path: Path) -> dict[str, int]:
    """Count model-content failures by paper, excluding transport errors."""

    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        error = str(event.get("error", ""))
        if not any(marker.lower() in error.lower() for marker in CONTENT_FAILURE_MARKERS):
            continue
        paper_id = str(event.get("paper_id", ""))
        if paper_id:
            counts[paper_id] = counts.get(paper_id, 0) + 1
    return counts


def write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV without ever exposing a truncated final artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        default="full_corpus",
        choices=sorted(SCOPE_BY_ID),
        help="Papers to code (default: full_corpus so every query view is covered).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max papers this run.")
    parser.add_argument(
        "--paper-ids-file",
        type=Path,
        default=None,
        help="CSV containing paper_id values to code, in file order.",
    )
    parser.add_argument(
        "--max-content-failures",
        type=int,
        default=None,
        help=(
            "Validation-only retry ceiling: skip papers with this many prior "
            "model-content failures; transport failures never count."
        ),
    )
    parser.add_argument("--model", default=None, help="Override OPENAI_MODEL/.env.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the provider endpoint (useful for an SSH-tunneled Ollama).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent requests (default: 1 local, 10 remote).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=f"Use local Ollama at {OLLAMA_BASE_URL} (free testing; model "
        f"defaults to {DEFAULT_LOCAL_MODEL}, check 'ollama list').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Estimate only; no API calls.")
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (1 if args.local else 10)
    if workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_content_failures is not None and args.max_content_failures < 1:
        parser.error("--max-content-failures must be at least 1")

    env = load_env(PROJECT_ROOT / ".env")
    if args.local:
        base_url = (
            args.base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or env.get("OLLAMA_BASE_URL")
            or OLLAMA_BASE_URL
        )
        api_key = "ollama"  # Ollama ignores the key but the client requires one
        model = (
            args.model
            or os.environ.get("OLLAMA_MODEL")
            or env.get("OLLAMA_MODEL")
            or DEFAULT_LOCAL_MODEL
        )
    else:
        base_url = (
            args.base_url
            or os.environ.get("OPENAI_BASE_URL")
            or env.get("OPENAI_BASE_URL")
        )  # None -> api.openai.com
        api_key = os.environ.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
        model = (
            args.model
            or os.environ.get("OPENAI_MODEL")
            or env.get("OPENAI_MODEL")
            or DEFAULT_MODEL
        )
    protocol_id, max_output_tokens = protocol_for_model(model)
    cache_dir = model_cache_dir(CACHE_ROOT, model, protocol_id)

    master = load_corpus()
    scoped = scope_frame(master, args.scope)
    if args.paper_ids_file is not None:
        selection = pd.read_csv(
            args.paper_ids_file, dtype=str, keep_default_na=False
        )
        if "paper_id" not in selection.columns:
            parser.error("--paper-ids-file must contain a paper_id column")
        requested = selection["paper_id"].drop_duplicates().tolist()
        available = set(scoped["paper_id"])
        missing = [paper_id for paper_id in requested if paper_id not in available]
        if missing:
            parser.error(
                f"{len(missing)} selected paper IDs are outside scope; first: {missing[0]}"
            )
        order = {paper_id: index for index, paper_id in enumerate(requested)}
        scoped = scoped[scoped["paper_id"].isin(requested)].copy()
        scoped["_selection_order"] = scoped["paper_id"].map(order)
        scoped = scoped.sort_values("_selection_order").drop(columns="_selection_order")
    failure_log_path = cache_dir / "failures.jsonl"
    prior_content_failures = content_failure_counts(failure_log_path)
    skipped_nonresponses: dict[str, int] = {}
    if args.max_content_failures is not None:
        scoped_ids = set(scoped["paper_id"])
        skipped_nonresponses = {
            paper_id: count
            for paper_id, count in prior_content_failures.items()
            if count >= args.max_content_failures and paper_id in scoped_ids
        }
    todo = scoped[
        ~scoped["paper_id"].map(lambda pid: (cache_dir / cache_key(pid)).exists())
        & ~scoped["paper_id"].isin(skipped_nonresponses)
    ]
    cached = sum(
        (cache_dir / cache_key(paper_id)).exists()
        for paper_id in scoped["paper_id"]
    )
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
    cache_display = cache_dir.relative_to(PROJECT_ROOT)
    print(
        f"Protocol '{protocol_id}' | scope '{args.scope}': {len(scoped):,} papers | "
        f"cached for {model}: {cached:,} | model non-responses skipped: "
        f"{len(skipped_nonresponses):,} | to code now: {len(todo):,} | "
        f"workers: {workers} | {est_line}\n"
        f"Cache: {cache_display}",
        flush=True,
    )

    if args.dry_run:
        return

    if not args.local and not api_key:
        sys.exit("OPENAI_API_KEY missing. Create .env in the project root "
                 "(see module docstring), or test free with --local.")

    from openai import OpenAI, RateLimitError  # imported late so --dry-run works without a key

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=SDK_MAX_RETRIES,
    )
    corpus_path = PROCESSED_DIR / "master_corpus.csv"
    manifest = {
        "protocol_id": protocol_id,
        "protocol_fingerprint": protocol_fingerprint(protocol_id, max_output_tokens),
        "parameters": protocol_parameters(max_output_tokens),
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "sdk_max_retries": SDK_MAX_RETRIES,
        "model": model,
        "provider_mode": "ollama" if args.local else "openai_compatible",
        "base_url": base_url,
        "openai_sdk_version": importlib.metadata.version("openai"),
        "python_version": sys.version,
        "corpus_path": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "papers_in_scope": len(scoped),
        "paper_ids_file": (
            str(args.paper_ids_file.resolve()) if args.paper_ids_file else None
        ),
        "max_content_failures": args.max_content_failures,
        "skipped_model_nonresponses": skipped_nonresponses,
        "workers": workers,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "protocol_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Coding {len(todo):,} papers with {model} using {workers} worker(s)"
          f" under {protocol_id}"
          f"{' via ' + base_url if base_url else ''}...", flush=True)

    started = time.time()
    failures: list[str] = []
    progress = ProgressReporter(
        "Specification", len(todo), every=1, started=started, in_place=True
    )
    if todo.empty:
        progress.update(0, detail="all papers cached", force=True)
    else:
        progress.update(
            0,
            detail=f"STARTING {min(workers, len(todo))} request(s)",
            force=True,
        )
    def code_row(row) -> tuple[str, float, str | None]:
        paper = {
            "paper_id": row["paper_id"],
            "title": row.get("Title", ""),
            "abstract": row.get("Abstract", ""),
            "keywords": row.get("Author Keywords", ""),
            "journal": row.get("Source title", ""),
            "year": row.get("Year", ""),
        }
        paper_started = time.time()
        attempts = 0
        while True:
            try:
                code_paper(
                    client,
                    model,
                    paper,
                    cache_dir,
                    local=args.local,
                    protocol_id=protocol_id,
                    max_output_tokens=max_output_tokens,
                )
                return paper["paper_id"], time.time() - paper_started, None
            except RateLimitError as error:
                attempts += 1
                if attempts >= RATE_LIMIT_MAX_ATTEMPTS:
                    return (
                        paper["paper_id"],
                        time.time() - paper_started,
                        f"rate limit persisted through {attempts} backoff attempts: {error}",
                    )
                # Exponential backoff with jitter, capped at 30s. The server
                # hint is sub-second; the cap covers a saturated TPM window.
                delay = min(30.0, 0.5 * (2 ** attempts)) * (0.5 + random.random())
                time.sleep(delay)
            except Exception as error:  # keep going; failures are re-tried next run
                return paper["paper_id"], time.time() - paper_started, str(error)

    executor = ThreadPoolExecutor(max_workers=workers)
    row_iterator = (row for _, row in todo.iterrows())
    pending = set()
    completed = 0
    try:
        for _ in range(min(len(todo), workers * 2)):
            pending.add(executor.submit(code_row, next(row_iterator)))
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                completed += 1
                paper_id, duration, error = future.result()
                if error is not None:
                    failures.append(f"{paper_id}: {error}")
                    with failure_log_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "timestamp": datetime.now().isoformat(),
                                    "paper_id": paper_id,
                                    "error": error,
                                }
                            )
                            + "\n"
                        )
                    if len(failures) <= 5 or len(failures) % 100 == 0:
                        print(
                            f"\nERROR {paper_id} ({len(failures)} failures): {error}",
                            flush=True,
                        )
                outcome = "FAILED" if error is not None else "DONE"
                progress.update(
                    completed,
                    failures=len(failures),
                    detail=(
                        f"{outcome} {paper_id} in {format_duration(duration)} | "
                        f"workers {workers}"
                    ),
                )
                try:
                    row = next(row_iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(code_row, row))
                if completed >= 20 and len(failures) / completed >= 0.5:
                    raise RuntimeError(
                        "Circuit breaker: at least 50% of the first "
                        f"{completed} requests failed. See {failure_log_path}."
                    )
    except KeyboardInterrupt:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except RuntimeError:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    # Assemble the full coded dataset for this scope from the cache.
    print("Assembling coded dataset from cache...", flush=True)
    records = []
    for pid in scoped["paper_id"]:
        cache_path = cache_dir / cache_key(pid)
        if cache_path.exists():
            records.append(
                sanitize_lone_surrogates(
                    json.loads(cache_path.read_text(encoding="utf-8"))
                )
            )
    coded = enrich_for_analysis(pd.DataFrame(records))

    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = cache_dir.name
    experiment_slug = f"{model_slug}_{protocol_id}"
    out_path = SPEC_DIR / f"paper_specifications_{experiment_slug}.csv"
    write_csv_atomically(coded, out_path)

    report = {
        "timestamp": datetime.now().isoformat(),
        **manifest,
        "scope": args.scope,
        "coded_total": len(coded),
        "coded_this_run": len(todo) - len(failures),
        "failures": failures,
        "runtime_seconds": round(time.time() - started, 1),
    }
    with open(
        SPEC_DIR / f"specification_report_{experiment_slug}.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
    print(f"Done: {len(coded):,} coded papers -> {out_path}")
    if failures:
        print(f"{len(failures)} failures (will retry automatically on next run)")


if __name__ == "__main__":
    main()
