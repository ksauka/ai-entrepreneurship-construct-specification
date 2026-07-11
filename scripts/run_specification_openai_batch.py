"""OpenAI Batch orchestrator for the full-corpus specification run.

Same frozen spec-v3 instrument and the SAME per-paper cache as the live
runner, at 50% of live prices. The tier's Batch queue limit is a cap on
tokens ENQUEUED AT ONE TIME, so the corpus is processed as sequential
chunks: submit one batch under the budget, poll to a terminal state, fetch
results into the cache, repeat until no uncached papers remain.

Usage (from the project root, graphrag env):

    python scripts/run_specification_openai_batch.py preview
    python scripts/run_specification_openai_batch.py run --yes     # PAID
    python scripts/run_specification_openai_batch.py status
    python scripts/run_specification_openai_batch.py fetch

- `preview` estimates chunks and cost; makes no API call.
- `run` refuses to submit without --yes (paid execution stays explicit).
  It is resumable: cached papers are skipped, an in-flight batch is polled
  rather than resubmitted, and Ctrl+C between batches loses nothing.
- `status` and `fetch` are safe at any time; `fetch` collects any completed
  batches that have not been written to the cache yet.

Options: --model (default gpt-5.4-mini-2026-03-17), --token-budget (default
1,850,000 to stay under the 2M enqueued cap), --poll-seconds (default 60),
--paper-ids-file (restrict to a manifest, e.g. the 50-paper challenge set).
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

from aecsp.specification.llm_coder import (  # noqa: E402
    cache_key,
    load_env,
    model_cache_dir,
    protocol_fingerprint,
    protocol_for_model,
    protocol_parameters,
)
from aecsp.specification.analysis_columns import enrich_for_analysis  # noqa: E402
from aecsp.specification.openai_batch import (  # noqa: E402
    BATCH_DISCOUNT,
    DEFAULT_TOKEN_BUDGET,
    custom_id_for,
    estimate_cost,
    pack_batches,
    parse_result_line,
    request_line,
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_ROOT = PROJECT_ROOT / "data" / "interim" / "spec_cache"
DEFAULT_BATCH_MODEL = "gpt-5.4-mini-2026-03-17"
LIVE_PRICES = {  # USD per 1M input/output tokens at LIVE rates
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "gpt-4.1-nano-2025-04-14": (0.10, 0.40),
}
TERMINAL = {"completed", "failed", "expired", "cancelled"}


def load_papers(paper_ids_file: Path | None) -> list[dict[str, str]]:
    master = pd.read_csv(PROCESSED_DIR / "master_corpus.csv", dtype=str, keep_default_na=False)
    if paper_ids_file is not None:
        selection = pd.read_csv(paper_ids_file, dtype=str, keep_default_na=False)
        if "paper_id" not in selection.columns:
            sys.exit("--paper-ids-file must contain a paper_id column")
        wanted = selection["paper_id"].drop_duplicates().tolist()
        order = {pid: i for i, pid in enumerate(wanted)}
        master = master[master["paper_id"].isin(wanted)].copy()
        master["_o"] = master["paper_id"].map(order)
        master = master.sort_values("_o")
    return [
        {
            "paper_id": row["paper_id"],
            "title": row.get("Title", ""),
            "abstract": row.get("Abstract", ""),
            "keywords": row.get("Author Keywords", ""),
            "journal": row.get("Source title", ""),
            "year": row.get("Year", ""),
        }
        for _, row in master.iterrows()
    ]


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"batches": []}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def pending_paper_ids(state: dict) -> set[str]:
    """Papers inside submitted batches that have not been fetched yet."""

    ids: set[str] = set()
    for batch in state["batches"]:
        if not batch.get("fetched"):
            ids.update(batch.get("paper_ids", []))
    return ids


def log_failure(cache_dir: Path, paper_id: str, error: str) -> None:
    with (cache_dir / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "paper_id": paper_id,
            "error": error,
            "transport": "openai_batch",
        }) + "\n")


def fetch_batch(client, batch_entry: dict, cache_dir: Path, model: str,
                protocol_id: str, max_output_tokens: int) -> tuple[int, int]:
    """Write one completed batch's results into the cache. Returns (ok, failed)."""

    id_by_custom = {custom_id_for(pid): pid for pid in batch_entry["paper_ids"]}
    seen: set[str] = set()
    ok = failed = 0
    for file_key in ("output_file_id", "error_file_id"):
        file_id = batch_entry.get(file_key)
        if not file_id:
            continue
        content = client.files.content(file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            paper_id = id_by_custom.get(obj.get("custom_id"))
            if paper_id is None:
                continue
            seen.add(paper_id)
            coded, error = parse_result_line(obj)
            if error is not None:
                failed += 1
                log_failure(cache_dir, paper_id, error)
                continue
            coded["paper_id"] = paper_id
            coded["coding_model"] = model
            coded["coding_protocol"] = protocol_id
            coded["coding_protocol_fingerprint"] = protocol_fingerprint(
                protocol_id, max_output_tokens
            )
            coded["coding_parameters_json"] = json.dumps(
                protocol_parameters(max_output_tokens), sort_keys=True
            )
            (cache_dir / cache_key(paper_id)).write_text(
                json.dumps(coded, indent=2), encoding="utf-8"
            )
            ok += 1
    for paper_id in set(batch_entry["paper_ids"]) - seen:
        failed += 1
        log_failure(cache_dir, paper_id, "no result line returned for this request")
    return ok, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preview", "run", "status", "fetch", "export"])
    parser.add_argument("--model", default=DEFAULT_BATCH_MODEL)
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                        help="Estimated input tokens per chunk; keep under the enqueued cap.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--paper-ids-file", type=Path, default=None)
    parser.add_argument("--yes", action="store_true",
                        help="Required for `run`: confirms paid batch submission.")
    args = parser.parse_args()

    protocol_id, max_output_tokens = protocol_for_model(args.model)
    cache_dir = model_cache_dir(CACHE_ROOT, args.model, protocol_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state_path = cache_dir / "openai_batch_state.json"
    state = load_state(state_path)

    papers = load_papers(args.paper_ids_file)
    uncached = [p for p in papers if not (cache_dir / cache_key(p["paper_id"])).exists()]
    in_flight = pending_paper_ids(state)
    todo = [p for p in uncached if p["paper_id"] not in in_flight]
    chunks = pack_batches(todo, args.token_budget)
    prices = LIVE_PRICES.get(args.model)
    cost = estimate_cost(len(todo), prices) if prices else None
    print(f"Protocol '{protocol_id}' | model {args.model} | papers {len(papers):,} | "
          f"cached {len(papers) - len(uncached):,} | in submitted batches {len(in_flight):,} | "
          f"to submit {len(todo):,} in {len(chunks)} chunk(s)"
          + (f" | est. batch cost ${cost:.2f} ({int(BATCH_DISCOUNT*100)}% of live)" if cost else ""),
          flush=True)

    if args.command == "export":
        records = []
        for paper in papers:
            path = cache_dir / cache_key(paper["paper_id"])
            if path.exists():
                records.append(json.loads(path.read_text(encoding="utf-8")))
        coded = enrich_for_analysis(pd.DataFrame(records))
        output = PROCESSED_DIR / "specification" / f"paper_specifications_{cache_dir.name}_{protocol_id}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        coded.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"Exported {len(coded):,}/{len(papers):,} cached papers to {output}")
        return

    if args.command == "preview":
        return

    env = load_env(PROJECT_ROOT / ".env")
    api_key = env.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY missing from .env")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    if args.command == "status":
        for entry in state["batches"]:
            batch = client.batches.retrieve(entry["id"])
            entry["status"] = batch.status
            entry["output_file_id"] = batch.output_file_id
            entry["error_file_id"] = batch.error_file_id
            counts = batch.request_counts
            print(f"  {entry['id']}: {batch.status} | total {counts.total} "
                  f"done {counts.completed} failed {counts.failed} | fetched={entry.get('fetched', False)}",
                  flush=True)
        save_state(state_path, state)
        return

    if args.command == "fetch":
        for entry in state["batches"]:
            if entry.get("fetched") or entry.get("status") not in TERMINAL:
                continue
            ok, failed = fetch_batch(client, entry, cache_dir, args.model,
                                     protocol_id, max_output_tokens)
            entry["fetched"] = True
            save_state(state_path, state)
            print(f"  fetched {entry['id']}: {ok} cached, {failed} failures", flush=True)
        return

    # ---- run: sequential submit -> poll -> fetch loop ----------------------
    if not args.yes:
        sys.exit("`run` submits PAID batches. Re-run with --yes to confirm.")

    manifest = {
        "protocol_id": protocol_id,
        "protocol_fingerprint": protocol_fingerprint(protocol_id, max_output_tokens),
        "parameters": protocol_parameters(max_output_tokens),
        "model": args.model,
        "provider_mode": "openai_batch",
        "token_budget_per_batch": args.token_budget,
        "papers_in_scope": len(papers),
        "experiment_register": "configs/experiment_register.json",
    }
    (cache_dir / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    def poll_to_terminal(batch_id: str) -> object:
        while True:
            batch = client.batches.retrieve(batch_id)
            counts = batch.request_counts
            print(f"  {batch_id}: {batch.status} | {counts.completed}/{counts.total} "
                  f"done, {counts.failed} failed", flush=True)
            if batch.status in TERMINAL:
                return batch
            time.sleep(args.poll_seconds)

    # Resume any non-terminal or unfetched batch first.
    for entry in state["batches"]:
        if not entry.get("fetched"):
            if entry.get("status") not in TERMINAL:
                batch = poll_to_terminal(entry["id"])
                entry["status"] = batch.status
                entry["output_file_id"] = batch.output_file_id
                entry["error_file_id"] = batch.error_file_id
                save_state(state_path, state)
            ok, failed = fetch_batch(client, entry, cache_dir, args.model,
                                     protocol_id, max_output_tokens)
            entry["fetched"] = True
            save_state(state_path, state)
            print(f"  resumed {entry['id']}: {ok} cached, {failed} failures", flush=True)

    inputs_dir = cache_dir / "batch_inputs"
    inputs_dir.mkdir(exist_ok=True)
    for index, chunk in enumerate(chunks, start=1):
        jsonl_path = inputs_dir / f"chunk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for paper in chunk:
                handle.write(json.dumps(request_line(args.model, paper, max_output_tokens)) + "\n")
        uploaded = client.files.create(file=jsonl_path.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        entry = {
            "id": batch.id,
            "input_file_id": uploaded.id,
            "submitted_at": datetime.now().isoformat(),
            "n": len(chunk),
            "paper_ids": [p["paper_id"] for p in chunk],
            "status": batch.status,
            "fetched": False,
        }
        state["batches"].append(entry)
        save_state(state_path, state)
        print(f"submitted chunk {index}/{len(chunks)}: batch {batch.id} ({len(chunk)} papers)",
              flush=True)
        batch = poll_to_terminal(batch.id)
        entry["status"] = batch.status
        entry["output_file_id"] = batch.output_file_id
        entry["error_file_id"] = batch.error_file_id
        save_state(state_path, state)
        ok, failed = fetch_batch(client, entry, cache_dir, args.model,
                                 protocol_id, max_output_tokens)
        entry["fetched"] = True
        save_state(state_path, state)
        print(f"chunk {index}/{len(chunks)} complete: {ok} cached, {failed} failures", flush=True)

    total = sum(1 for p in papers if (cache_dir / cache_key(p["paper_id"])).exists())
    print(f"Batch run finished: {total:,}/{len(papers):,} papers cached in {cache_dir}",
          flush=True)


if __name__ == "__main__":
    main()
