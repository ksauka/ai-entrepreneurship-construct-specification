"""Prepare, submit, monitor, and collect Anthropic specification batches.

Inputs: the master corpus, frozen spec-v3 prompt/schema, ANTHROPIC_API_KEY, and
an explicit action. Outputs: request shards, batch state, per-paper cache JSON,
failure logs, and model-specific CSV/report files. The default action is the
offline, non-billable preparation step; submission requires --confirm-submit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.specification.analysis_columns import enrich_for_analysis  # noqa: E402

from aecsp.specification.llm_coder import (  # noqa: E402
    MAX_OUTPUT_TOKENS,
    PROTOCOL_ID,
    SYSTEM_PROMPT,
    build_user_prompt,
    cache_key,
    flatten_profile,
    load_env,
    model_cache_dir,
    response_json_schema,
)

API_ROOT = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_TARGET = PROJECT_ROOT / "data/interim/proprietary_validation/proprietary_rater_target_2276_papers.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CACHE_ROOT = PROJECT_ROOT / "data" / "interim" / "spec_cache"
SPEC_DIR = PROCESSED_DIR / "specification"

# Official Claude API Batch prices through 2026-08-31, USD per 1M tokens.
SONNET5_INTRO_BATCH_INPUT = 1.00
SONNET5_INTRO_BATCH_OUTPUT = 5.00
SONNET5_STANDARD_BATCH_INPUT = 1.50
SONNET5_STANDARD_BATCH_OUTPUT = 7.50
SONNET5_INTRO_END = date(2026, 8, 31)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_batch_dir(cache_dir: Path, paper_ids_file: Path) -> Path:
    """Keep pilot and full-target transport state in separate directories."""

    return cache_dir / "anthropic_batches" / paper_ids_file.stem


def anthropic_parameters(model: str) -> dict[str, Any]:
    """Provider request settings, including unsupported spec-v3 controls."""

    return {
        "model": model,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
        "response_format": "output_config.format.json_schema",
        "temperature": "provider default; Sonnet 5 rejects non-default sampling",
        "top_p": "provider default; Sonnet 5 rejects non-default sampling",
        "seed": "unsupported by Anthropic Messages API",
        "frequency_penalty": "unsupported by Anthropic Messages API",
        "presence_penalty": "unsupported by Anthropic Messages API",
        "thinking": "disabled",
    }


def anthropic_fingerprint(model: str) -> str:
    payload = {
        "protocol_id": PROTOCOL_ID,
        "provider": "anthropic_message_batches",
        "system_prompt": SYSTEM_PROMPT,
        "response_schema": response_json_schema()["schema"],
        "parameters": anthropic_parameters(model),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def paper_from_row(row: pd.Series) -> dict[str, str]:
    return {
        "paper_id": row["paper_id"],
        "title": row.get("Title", ""),
        "abstract": row.get("Abstract", ""),
        "keywords": row.get("Author Keywords", ""),
        "journal": row.get("Source title", ""),
        "year": row.get("Year", ""),
    }


def custom_id(paper_id: str) -> str:
    return "paper_" + hashlib.sha1(str(paper_id).encode()).hexdigest()


def request_for_paper(model: str, paper: dict[str, str]) -> dict[str, Any]:
    return {
        "custom_id": custom_id(paper["paper_id"]),
        "params": {
            "model": model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": build_user_prompt(
                        paper["title"],
                        paper["abstract"],
                        paper["keywords"],
                        paper["journal"],
                        paper["year"],
                    ),
                }
            ],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": response_json_schema()["schema"],
                }
            },
            "thinking": {"type": "disabled"},
        },
    }


def validate_one(api_key: str, model: str, paper: dict[str, str], batch_dir: Path) -> None:
    """Run one paid live request before permitting a Batch experiment."""

    response = api_request(
        api_key, "POST", "/v1/messages", request_for_paper(model, paper)["params"]
    )
    text = "".join(
        block.get("text", "") for block in response.get("content", [])
        if block.get("type") == "text"
    )
    error = None
    try:
        flatten_profile(json.loads(text))
    except Exception as exc:
        error = f"response did not parse as the profile schema: {exc}"
    report = {
        "timestamp": datetime.now().isoformat(), "provider": "anthropic_live_gate",
        "model": model, "paper_id": paper["paper_id"], "passed": error is None,
        "error": error, "provider_fingerprint": anthropic_fingerprint(model),
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "live_validation_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if error:
        raise SystemExit(f"Claude one-paper validation FAILED: {error}")
    print("Claude one-paper validation PASSED: schema parsed successfully")


def api_request(
    api_key: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    jsonl: bool = False,
) -> Any:
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        API_ROOT + path,
        data=body,
        method=method,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API {error.code}: {detail}") from error
    if jsonl:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return json.loads(raw)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"batches": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def progress_line(label: str, completed: int, total: int, failures: int = 0) -> str:
    """Render provider and collection progress as a percentage bar."""

    total = max(total, 0)
    completed = max(0, min(completed, total)) if total else 0
    ratio = completed / total if total else 0.0
    width = 30
    filled = min(width, int(ratio * width))
    bar = "=" * filled + (">" if filled < width and completed else "")
    bar = bar.ljust(width, "-")
    return (
        f"{label} [{bar}] {completed:,}/{total:,} "
        f"({ratio * 100:6.2f}%) | failures {failures:,}"
    )


def observed_cost_estimate(todo_count: int) -> tuple[float, float, float]:
    """Estimate Batch cost from nano cache usage and Sonnet tokenizer guidance."""

    claude_reference = model_cache_dir(CACHE_ROOT, DEFAULT_MODEL, PROTOCOL_ID)
    reference = claude_reference if any(claude_reference.glob("*.json")) else model_cache_dir(
        CACHE_ROOT, "gpt-4.1-nano-2025-04-14", PROTOCOL_ID
    )
    prompt_total = output_total = count = 0
    for path in reference.glob("*.json"):
        if path.name == "protocol_manifest.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        prompt = record.get("prompt_tokens")
        output = record.get("output_tokens")
        if isinstance(prompt, (int, float)) and isinstance(output, (int, float)):
            prompt_total += prompt
            output_total += output
            count += 1
    if count and reference == claude_reference:
        input_tokens = prompt_total / count
        output_tokens = output_total / count
    elif count:
        # Anthropic documents approximately 30% more tokens for Sonnet 5's
        # tokenizer. This remains an estimate until count_tokens is called.
        input_tokens = (prompt_total / count) * 1.30
        output_tokens = (output_total / count) * 1.30
    else:
        input_tokens, output_tokens = 6_326.0, 910.0
    if date.today() <= SONNET5_INTRO_END:
        input_price = SONNET5_INTRO_BATCH_INPUT
        output_price = SONNET5_INTRO_BATCH_OUTPUT
    else:
        input_price = SONNET5_STANDARD_BATCH_INPUT
        output_price = SONNET5_STANDARD_BATCH_OUTPUT
    cost = todo_count * (
        input_tokens * input_price + output_tokens * output_price
    ) / 1_000_000
    return input_tokens, output_tokens, cost


def prepare(
    model: str,
    cache_dir: Path,
    batch_dir: Path,
    chunk_size: int,
    limit: int | None,
    paper_ids_file: Path,
) -> None:
    corpus_path = PROCESSED_DIR / "master_corpus.csv"
    corpus = pd.read_csv(corpus_path, dtype=str, keep_default_na=False)
    selection = pd.read_csv(paper_ids_file, dtype=str, keep_default_na=False)
    if "paper_id" not in selection:
        raise SystemExit("--paper-ids-file must contain a paper_id column")
    requested = selection["paper_id"].drop_duplicates().tolist()
    missing = sorted(set(requested) - set(corpus["paper_id"]))
    if missing:
        raise SystemExit(f"{len(missing)} target IDs are missing from master corpus; first: {missing[0]}")
    order = {paper_id: index for index, paper_id in enumerate(requested)}
    corpus = corpus[corpus["paper_id"].isin(requested)].copy()
    corpus["_target_order"] = corpus["paper_id"].map(order)
    corpus = corpus.sort_values("_target_order")
    todo = corpus[
        ~corpus["paper_id"].map(lambda pid: (cache_dir / cache_key(pid)).exists())
    ]
    if limit is not None:
        todo = todo.head(limit)
    batch_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    requests: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    for _, row in todo.iterrows():
        paper = paper_from_row(row)
        cid = custom_id(paper["paper_id"])
        mapping[cid] = paper["paper_id"]
        requests.append(request_for_paper(model, paper))
        if len(requests) == chunk_size:
            shards.append(_write_shard(batch_dir, len(shards), requests))
            requests = []
    if requests:
        shards.append(_write_shard(batch_dir, len(shards), requests))

    mapping_path = batch_dir / "custom_id_map.json"
    mapping_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    input_tokens, output_tokens, cost = observed_cost_estimate(len(todo))
    manifest = {
        "created_at": datetime.now().isoformat(),
        "protocol_id": PROTOCOL_ID,
        "provider": "anthropic_message_batches",
        "provider_fingerprint": anthropic_fingerprint(model),
        "model": model,
        "parameters": anthropic_parameters(model),
        "corpus_path": str(corpus_path.relative_to(PROJECT_ROOT)),
        "corpus_sha256": file_sha256(corpus_path),
        "paper_ids_file": str(paper_ids_file.relative_to(PROJECT_ROOT)),
        "paper_ids_file_sha256": file_sha256(paper_ids_file),
        "papers_in_target": len(corpus),
        "papers_prepared": len(todo),
        "chunk_size": chunk_size,
        "shards": shards,
        "cost_estimate": {
            "estimated_input_tokens_per_paper": round(input_tokens, 1),
            "estimated_output_tokens_per_paper": round(output_tokens, 1),
            "estimated_batch_cost_usd": round(cost, 2),
            "warning": "Estimate uses observed Claude pilot usage when available; actual cost varies by paper length and response tokens.",
        },
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared {len(todo):,} papers in {len(shards)} shard(s) under {batch_dir}.\n"
        f"Estimated Batch cost: ${cost:,.2f} (not submitted; no API charge)."
    )


def _write_shard(
    batch_dir: Path, index: int, requests: list[dict[str, Any]]
) -> dict[str, Any]:
    path = batch_dir / f"requests_{index:04d}.json"
    path.write_text(json.dumps({"requests": requests}), encoding="utf-8")
    return {
        "index": index,
        "path": path.name,
        "request_count": len(requests),
        "sha256": file_sha256(path),
    }


def submit(api_key: str, batch_dir: Path, confirm: bool) -> None:
    if not confirm:
        raise SystemExit(
            "Submission can incur charges. Re-run with --confirm-submit after reviewing batch_manifest.json."
        )
    manifest = json.loads((batch_dir / "batch_manifest.json").read_text())
    gate_path = batch_dir / "live_validation_gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if (
        not gate.get("passed")
        or gate.get("provider_fingerprint") != manifest["provider_fingerprint"]
    ):
        raise SystemExit(
            "Run the one-paper --action validate gate successfully before Batch submission."
        )
    state_path = batch_dir / "batch_state.json"
    state = load_state(state_path)
    submitted_paths = {entry["request_path"] for entry in state["batches"]}
    for shard in manifest["shards"]:
        if shard["path"] in submitted_paths:
            continue
        payload = json.loads((batch_dir / shard["path"]).read_text())
        response = api_request(
            api_key, "POST", "/v1/messages/batches", payload
        )
        state["batches"].append(
            {
                "batch_id": response["id"],
                "request_path": shard["path"],
                "request_count": shard["request_count"],
                "submitted_at": datetime.now().isoformat(),
                "processing_status": response.get("processing_status"),
                "collected": False,
            }
        )
        save_state(state_path, state)
        print(f"Submitted {shard['path']} -> {response['id']}")


def status(api_key: str, batch_dir: Path) -> None:
    state_path = batch_dir / "batch_state.json"
    state = load_state(state_path)
    if not state["batches"]:
        raise SystemExit("No submitted batches. Run --action submit explicitly.")
    for entry in state["batches"]:
        response = api_request(
            api_key, "GET", f"/v1/messages/batches/{entry['batch_id']}"
        )
        entry["processing_status"] = response.get("processing_status")
        entry["request_counts"] = response.get("request_counts")
        counts = entry.get("request_counts") or {}
        succeeded = int(counts.get("succeeded", 0) or 0)
        errored = int(counts.get("errored", 0) or 0)
        canceled = int(counts.get("canceled", 0) or 0)
        expired = int(counts.get("expired", 0) or 0)
        failures = errored + canceled + expired
        completed = succeeded + failures
        print(
            f"{entry['batch_id']}: {entry['processing_status']} "
            f"{entry.get('request_counts', {})}"
        )
        print(
            progress_line(
                "Claude Batch", completed, entry["request_count"], failures
            )
        )
    save_state(state_path, state)


def watch(
    api_key: str,
    model: str,
    cache_dir: Path,
    batch_dir: Path,
    poll_seconds: int,
) -> None:
    """Continuously poll all shards, then collect, validate and export."""

    state_path = batch_dir / "batch_state.json"
    state = load_state(state_path)
    if not state["batches"]:
        raise SystemExit("No submitted batches. Run --action submit explicitly.")
    total = sum(int(entry["request_count"]) for entry in state["batches"])
    print("Watching Claude Batch continuously; Ctrl+C detaches without cancelling it.")
    while True:
        completed = failures = 0
        all_ended = True
        for entry in state["batches"]:
            response = api_request(
                api_key, "GET", f"/v1/messages/batches/{entry['batch_id']}"
            )
            entry["processing_status"] = response.get("processing_status")
            entry["request_counts"] = response.get("request_counts") or {}
            counts = entry["request_counts"]
            succeeded = int(counts.get("succeeded", 0) or 0)
            shard_failures = sum(
                int(counts.get(key, 0) or 0)
                for key in ("errored", "canceled", "expired")
            )
            completed += succeeded + shard_failures
            failures += shard_failures
            all_ended = all_ended and entry["processing_status"] == "ended"
        save_state(state_path, state)
        print(
            "\r" + progress_line("Claude Batch", completed, total, failures),
            end="",
            flush=True,
        )
        if all_ended:
            print()
            break
        time.sleep(poll_seconds)
    collect(api_key, model, cache_dir, batch_dir)


def collect(api_key: str, model: str, cache_dir: Path, batch_dir: Path) -> None:
    state_path = batch_dir / "batch_state.json"
    state = load_state(state_path)
    mapping = json.loads((batch_dir / "custom_id_map.json").read_text())
    failure_path = cache_dir / "failures.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    collected = failed = 0
    for entry in state["batches"]:
        if entry.get("collected"):
            continue
        batch = api_request(
            api_key, "GET", f"/v1/messages/batches/{entry['batch_id']}"
        )
        if batch.get("processing_status") != "ended":
            print(f"Skipping {entry['batch_id']}: still in progress")
            continue
        results = api_request(
            api_key,
            "GET",
            f"/v1/messages/batches/{entry['batch_id']}/results",
            jsonl=True,
        )
        print(progress_line("Claude Collect", 0, len(results)))
        shard_failed = 0
        for index, item in enumerate(results, start=1):
            cid = item["custom_id"]
            paper_id = mapping[cid]
            result = item.get("result", {})
            if result.get("type") != "succeeded":
                _append_failure(failure_path, paper_id, result)
                failed += 1
                shard_failed += 1
                print(progress_line("Claude Collect", index, len(results), shard_failed))
                continue
            message = result["message"]
            if message.get("stop_reason") == "max_tokens":
                _append_failure(failure_path, paper_id, "max_tokens")
                failed += 1
                shard_failed += 1
                print(progress_line("Claude Collect", index, len(results), shard_failed))
                continue
            text = "".join(
                block.get("text", "")
                for block in message.get("content", [])
                if block.get("type") == "text"
            )
            try:
                coded = flatten_profile(json.loads(text))
            except Exception as error:
                _append_failure(failure_path, paper_id, f"parse error: {error}")
                failed += 1
                shard_failed += 1
                print(progress_line("Claude Collect", index, len(results), shard_failed))
                continue
            usage = message.get("usage", {})
            coded.update(
                {
                    "paper_id": paper_id,
                    "coding_model": model,
                    "prompt_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "coding_protocol": PROTOCOL_ID,
                    "coding_protocol_fingerprint": anthropic_fingerprint(model),
                    "coding_parameters_json": json.dumps(
                        anthropic_parameters(model), sort_keys=True
                    ),
                }
            )
            (cache_dir / cache_key(paper_id)).write_text(
                json.dumps(coded, indent=2), encoding="utf-8"
            )
            collected += 1
            print(progress_line("Claude Collect", index, len(results), shard_failed))
        entry["collected"] = True
        entry["collected_at"] = datetime.now().isoformat()
        save_state(state_path, state)
    print(f"Collected {collected:,} successful records; {failed:,} failures.")
    assemble_outputs(model, cache_dir, batch_dir)


def _append_failure(path: Path, paper_id: str, error: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
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


def assemble_outputs(model: str, cache_dir: Path, batch_dir: Path) -> None:
    records = []
    for path in cache_dir.glob("*.json"):
        if path.name == "protocol_manifest.json":
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if not records:
        raise SystemExit(
            "No cached Claude records exist for this target. Wait for the "
            "batch to end, then run collect before export."
        )
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    slug = cache_dir.name
    out_path = SPEC_DIR / f"paper_specifications_{slug}_{PROTOCOL_ID}.csv"
    enrich_for_analysis(pd.DataFrame(records)).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    report = {
        "timestamp": datetime.now().isoformat(),
        "protocol_id": PROTOCOL_ID,
        "provider": "anthropic_message_batches",
        "provider_fingerprint": anthropic_fingerprint(model),
        "model": model,
        "parameters": anthropic_parameters(model),
        "coded_total": len(records),
        "batch_directory": str(batch_dir.relative_to(PROJECT_ROOT)),
    }
    (SPEC_DIR / f"specification_report_{slug}_{PROTOCOL_ID}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Dataset -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("prepare", "validate", "submit", "run", "status", "watch", "collect", "export"),
        default="prepare",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--paper-ids-file", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--confirm-submit", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    args.paper_ids_file = args.paper_ids_file.resolve()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")

    env = load_env(PROJECT_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or env.get("ANTHROPIC_API_KEY")
    cache_dir = model_cache_dir(CACHE_ROOT, args.model, PROTOCOL_ID)
    batch_dir = target_batch_dir(cache_dir, args.paper_ids_file)

    if args.action == "prepare":
        prepare(
            args.model,
            cache_dir,
            batch_dir,
            args.chunk_size,
            args.limit,
            args.paper_ids_file,
        )
        return
    if args.action == "export":
        assemble_outputs(args.model, cache_dir, batch_dir)
        return
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required for this API action.")
    if args.action == "run":
        if not args.confirm_submit:
            raise SystemExit("`run` may submit PAID work. Re-run with --confirm-submit.")
        if not (batch_dir / "batch_manifest.json").exists():
            prepare(
                args.model, cache_dir, batch_dir, args.chunk_size, args.limit,
                args.paper_ids_file,
            )
        gate_path = batch_dir / "live_validation_gate.json"
        gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
        if gate.get("provider_fingerprint") != anthropic_fingerprint(args.model) or not gate.get("passed"):
            corpus = pd.read_csv(PROCESSED_DIR / "master_corpus.csv", dtype=str, keep_default_na=False)
            target = pd.read_csv(args.paper_ids_file, dtype=str, keep_default_na=False)
            row = corpus[corpus["paper_id"] == target.iloc[0]["paper_id"]].iloc[0]
            validate_one(api_key, args.model, paper_from_row(row), batch_dir)
        # `submit` records every accepted shard immediately and skips those
        # request paths on a later invocation. Calling it unconditionally makes
        # `run` resumable after an insufficient-credit or other submission
        # error interrupts a multi-shard launch.
        submit(api_key, batch_dir, True)
        watch(api_key, args.model, cache_dir, batch_dir, args.poll_seconds)
        return
    if args.action == "validate":
        if not args.confirm_submit:
            raise SystemExit(
                "`validate` makes one PAID live request. Re-run with --confirm-submit."
            )
        corpus = pd.read_csv(PROCESSED_DIR / "master_corpus.csv", dtype=str, keep_default_na=False)
        target = pd.read_csv(args.paper_ids_file, dtype=str, keep_default_na=False)
        row = corpus[corpus["paper_id"] == target.iloc[0]["paper_id"]].iloc[0]
        validate_one(api_key, args.model, paper_from_row(row), batch_dir)
        return
    if args.action == "submit":
        submit(api_key, batch_dir, args.confirm_submit)
    elif args.action == "status":
        status(api_key, batch_dir)
    elif args.action == "watch":
        watch(api_key, args.model, cache_dir, batch_dir, args.poll_seconds)
    else:
        collect(api_key, args.model, cache_dir, batch_dir)


if __name__ == "__main__":
    main()
