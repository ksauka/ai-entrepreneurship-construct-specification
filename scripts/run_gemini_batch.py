"""Prepare, submit, monitor, fetch, and export the Gemini validation batch.

Offline commands (`preview`, `prepare`) never call the API. `submit` is paid
and requires `--yes`. The default target is the frozen 2,276-paper manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.specification.analysis_columns import enrich_for_analysis  # noqa: E402
from aecsp.specification.gemini_batch import (  # noqa: E402
    custom_id_for,
    estimate_cost,
    generation_config,
    parse_result_line,
    request_line,
)
from aecsp.specification.llm_coder import (  # noqa: E402
    PROTOCOL_ID,
    SYSTEM_PROMPT,
    cache_key,
    load_env,
    model_cache_dir,
    response_json_schema,
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TARGET = PROJECT_ROOT / "data/interim/proprietary_validation/proprietary_rater_target_2276_papers.csv"
CACHE_ROOT = PROJECT_ROOT / "data/interim/spec_cache"
PROCESSED = PROJECT_ROOT / "data/processed"
TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
MAX_TIER_1_SAFE_DIRECT_PAPERS = 2_500


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_batch_dir(cache_dir: Path, target: Path) -> Path:
    """Keep pilot and full-target transport state in separate directories."""

    return cache_dir / "gemini_batches" / target.stem


def provider_fingerprint(model: str) -> str:
    payload = {
        "provider": "gemini_batch",
        "model": model,
        "protocol": PROTOCOL_ID,
        "system_prompt": SYSTEM_PROMPT,
        "schema": response_json_schema()["schema"],
        "generation_config": generation_config(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def ensure_direct_batch_size(papers: int) -> None:
    if papers > MAX_TIER_1_SAFE_DIRECT_PAPERS:
        raise SystemExit(
            f"Refusing one Gemini job for {papers:,} uncached papers. Use "
            "scripts/run_gemini_full_batch.py so Tier 1 queue safety does not "
            "depend on the account tier."
        )


def load_papers(target: Path) -> list[dict[str, str]]:
    selection = pd.read_csv(target, dtype=str, keep_default_na=False)
    if "paper_id" not in selection:
        raise SystemExit("--paper-ids-file must contain a paper_id column")
    master = pd.read_csv(PROCESSED / "master_corpus.csv", dtype=str, keep_default_na=False)
    wanted = selection["paper_id"].drop_duplicates().tolist()
    missing = sorted(set(wanted) - set(master["paper_id"]))
    if missing:
        raise SystemExit(f"{len(missing)} target IDs missing from master; first: {missing[0]}")
    order = {paper_id: index for index, paper_id in enumerate(wanted)}
    master = master[master["paper_id"].isin(wanted)].copy()
    master["_order"] = master["paper_id"].map(order)
    master = master.sort_values("_order")
    return [
        {
            "paper_id": row["paper_id"], "title": row["Title"],
            "abstract": row["Abstract"], "keywords": row["Author Keywords"],
            "journal": row["Source title"], "year": row["Year"],
        }
        for _, row in master.iterrows()
    ]


def prepare(model: str, papers: list[dict[str, str]], cache_dir: Path, target: Path) -> Path:
    batch_dir = target_batch_dir(cache_dir, target)
    batch_dir.mkdir(parents=True, exist_ok=True)
    todo = [paper for paper in papers if not (cache_dir / cache_key(paper["paper_id"])).exists()]
    ensure_direct_batch_size(len(todo))
    input_path = batch_dir / "requests.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for paper in todo:
            handle.write(json.dumps(request_line(model, paper)) + "\n")
    mapping = {custom_id_for(paper["paper_id"]): paper["paper_id"] for paper in todo}
    (batch_dir / "custom_id_map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(), "provider": "gemini_batch",
        "model": model, "protocol": PROTOCOL_ID,
        "provider_fingerprint": provider_fingerprint(model),
        "target": str(target.relative_to(PROJECT_ROOT)), "target_sha256": sha256(target),
        "papers_in_target": len(papers), "papers_prepared": len(todo),
        "input_file": input_path.name, "input_sha256": sha256(input_path),
        "generation_config": generation_config(),
        "estimated_batch_cost_usd": round(estimate_cost(len(todo)), 2),
    }
    (batch_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(todo):,}/{len(papers):,} papers in {input_path}")
    print(f"Estimated Gemini Batch cost: ${manifest['estimated_batch_cost_usd']:.2f} (not submitted)")
    return batch_dir


def load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def ensure_validation_gate(cache_dir: Path, batch_dir: Path, model: str) -> bool:
    """Reuse a passed gate only when the complete provider fingerprint matches."""

    fingerprint = provider_fingerprint(model)
    local_path = batch_dir / "live_validation_gate.json"
    local = load_state(local_path)
    if local.get("passed") and local.get("provider_fingerprint") == fingerprint:
        return True

    batches_root = cache_dir / "gemini_batches"
    if not batches_root.exists():
        return False
    for candidate in sorted(batches_root.glob("*/live_validation_gate.json")):
        if candidate == local_path:
            continue
        gate = load_state(candidate)
        if gate.get("passed") and gate.get("provider_fingerprint") == fingerprint:
            batch_dir.mkdir(parents=True, exist_ok=True)
            reused = dict(gate)
            reused["reused_at"] = datetime.now().isoformat()
            reused["reused_from"] = str(candidate.relative_to(cache_dir))
            local_path.write_text(json.dumps(reused, indent=2), encoding="utf-8")
            print(f"Reused validated Gemini request gate from {candidate.parent.name}")
            return True
    return False


def progress_line(label: str, completed: int, total: int, failures: int = 0) -> str:
    """Render a stable percentage bar for provider and local fetch progress."""

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


def completion_counts(job: object, total: int) -> tuple[int, int]:
    """Read provider processing counts, not validated paper outcomes."""

    stats = getattr(job, "completion_stats", None)
    successful = int(getattr(stats, "successful_count", 0) or 0)
    failed = int(getattr(stats, "failed_count", 0) or 0)
    incomplete = int(getattr(stats, "incomplete_count", 0) or 0)
    completed = successful + failed
    return min(completed, total), failed


def validate_one(client, types, model: str, paper: dict[str, str], batch_dir: Path) -> None:
    """Run one paid live request before permitting a Batch experiment."""

    from aecsp.specification.llm_coder import build_user_prompt

    config = generation_config().copy()
    response = client.models.generate_content(
        model=model,
        contents=build_user_prompt(
            paper["title"], paper["abstract"], paper["keywords"],
            paper["journal"], paper["year"],
        ),
        config=types.GenerateContentConfig(
            **config, system_instruction=SYSTEM_PROMPT
        ),
    )
    raw = json.loads(response.model_dump_json(by_alias=True, exclude_none=True))
    coded, error = parse_result_line({"response": raw})
    report = {
        "timestamp": datetime.now().isoformat(), "provider": "gemini_live_gate",
        "model": model, "paper_id": paper["paper_id"], "passed": error is None,
        "error": error, "provider_fingerprint": provider_fingerprint(model),
    }
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "live_validation_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if error:
        raise SystemExit(f"Gemini one-paper validation FAILED: {error}")
    print("Gemini one-paper validation PASSED: schema parsed successfully")


def append_failure(cache_dir: Path, paper_id: str, error: str) -> None:
    with (cache_dir / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": datetime.now().isoformat(), "paper_id": paper_id, "error": error, "transport": "gemini_batch"}) + "\n")


def export(model: str, papers: list[dict[str, str]], cache_dir: Path) -> None:
    records = []
    for paper in papers:
        path = cache_dir / cache_key(paper["paper_id"])
        if path.exists():
            records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit(
            "No cached Gemini records exist for this target. Wait for "
            "JOB_STATE_SUCCEEDED, then run fetch before export."
        )
    frame = enrich_for_analysis(pd.DataFrame(records))
    output = PROCESSED / "specification" / f"paper_specifications_{cache_dir.name}_{PROTOCOL_ID}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Exported {len(frame):,}/{len(papers):,} records to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preview", "prepare", "validate", "submit", "run", "status", "watch", "fetch", "export"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--paper-ids-file", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    target = args.paper_ids_file.resolve()
    papers = load_papers(target)
    cache_dir = model_cache_dir(CACHE_ROOT, args.model, PROTOCOL_ID)
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = target_batch_dir(cache_dir, target)
    state_path = batch_dir / "batch_state.json"
    if args.command in {"preview", "prepare"}:
        prepare(args.model, papers, cache_dir, target)
        return
    if args.command == "export":
        export(args.model, papers, cache_dir)
        return
    env = load_env(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is required in ETV_V2/.env")
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise SystemExit("Install provider dependency: pip install -e '.[providers]'") from error
    client = genai.Client(api_key=api_key)
    state = load_state(state_path)
    if args.command == "run":
        if not args.yes:
            raise SystemExit("`run` may submit PAID work. Re-run with --yes.")
        if not state.get("job_name"):
            prepare(args.model, papers, cache_dir, target)
            if not ensure_validation_gate(cache_dir, batch_dir, args.model):
                validate_one(client, types, args.model, papers[0], batch_dir)
            uploaded = client.files.upload(
                file=str(batch_dir / "requests.jsonl"),
                config=types.UploadFileConfig(display_name="etv-spec-v3-gemini", mime_type="jsonl"),
            )
            job = client.batches.create(
                model=args.model, src=uploaded.name,
                config={"display_name": "etv-spec-v3-gemini"},
            )
            state = {
                "job_name": job.name, "uploaded_file": uploaded.name,
                "state": job.state.name, "submitted_at": datetime.now().isoformat(),
                "fetched": False,
            }
            save_state(state_path, state)
            print(f"Submitted {job.name} ({len(papers):,} target papers)")
    if args.command == "validate":
        if not args.yes:
            raise SystemExit("`validate` makes one PAID live request. Re-run with --yes.")
        validate_one(client, types, args.model, papers[0], batch_dir)
        return
    if args.command == "submit":
        if not args.yes:
            raise SystemExit("`submit` is PAID. Re-run with --yes after reviewing preview.")
        if not ensure_validation_gate(cache_dir, batch_dir, args.model):
            raise SystemExit("Run the one-paper `validate --yes` gate successfully before Batch submission.")
        if state.get("job_name"):
            raise SystemExit(f"A job is already recorded: {state['job_name']}")
        if not (batch_dir / "requests.jsonl").exists():
            prepare(args.model, papers, cache_dir, target)
        uploaded = client.files.upload(file=str(batch_dir / "requests.jsonl"), config=types.UploadFileConfig(display_name="etv-spec-v3-gemini", mime_type="jsonl"))
        job = client.batches.create(model=args.model, src=uploaded.name, config={"display_name": "etv-spec-v3-gemini"})
        state = {"job_name": job.name, "uploaded_file": uploaded.name, "state": job.state.name, "submitted_at": datetime.now().isoformat(), "fetched": False}
        save_state(state_path, state)
        print(f"Submitted {job.name} ({len(papers):,} target papers)")
        return
    if not state.get("job_name"):
        raise SystemExit("No submitted Gemini batch found")
    if args.command in {"watch", "run"}:
        print("Watching Gemini Batch continuously; Ctrl+C detaches without cancelling it.")
        while True:
            job = client.batches.get(name=state["job_name"])
            state["state"] = job.state.name
            state["checked_at"] = datetime.now().isoformat()
            save_state(state_path, state)
            completed, provider_failures = completion_counts(job, len(papers))
            line = progress_line(
                f"Gemini {job.state.name}", completed, len(papers), provider_failures
            )
            print(f"\r{line}", end="", flush=True)
            if job.state.name in TERMINAL:
                print()
                break
            time.sleep(args.poll_seconds)
    else:
        job = client.batches.get(name=state["job_name"])
    state["state"] = job.state.name
    state["checked_at"] = datetime.now().isoformat()
    save_state(state_path, state)
    completed, provider_failures = completion_counts(job, len(papers))
    print(f"{job.name}: {job.state.name}")
    if state.get("fetched"):
        print(progress_line(
            "Gemini outcomes", int(state.get("cached", 0)) + int(state.get("failed", 0)),
            len(papers), int(state.get("failed", 0)),
        ))
        print(f"Validated cache outcomes: {state.get('cached', 0)} successful, {state.get('failed', 0)} failed")
    else:
        print(progress_line("Gemini processed", completed, len(papers), provider_failures))
        print("Paper-level success is unknown until fetch validates every response.")
    if args.command == "status":
        return
    if job.state.name not in TERMINAL:
        raise SystemExit("Batch is not yet in a terminal state")
    if job.state.name != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"Batch ended as {job.state.name}: {job.error}")
    if state.get("fetched"):
        print("Results were already fetched")
        return
    content = client.files.download(file=job.dest.file_name).decode("utf-8")
    mapping = json.loads((batch_dir / "custom_id_map.json").read_text())
    ok = failed = 0
    lines = [line for line in content.splitlines() if line.strip()]
    print(progress_line("Gemini Fetch", 0, len(lines)))
    for index, line in enumerate(lines, start=1):
        item = json.loads(line)
        key = item.get("key") or item.get("metadata", {}).get("key")
        paper_id = mapping.get(key)
        if not paper_id:
            print(progress_line("Gemini Fetch", index, len(lines), failed))
            continue
        coded, error = parse_result_line(item)
        if error:
            append_failure(cache_dir, paper_id, error); failed += 1
            print(progress_line("Gemini Fetch", index, len(lines), failed))
            continue
        coded.update({"paper_id": paper_id, "coding_model": args.model, "coding_protocol": PROTOCOL_ID, "coding_protocol_fingerprint": provider_fingerprint(args.model), "coding_parameters_json": json.dumps(generation_config(), sort_keys=True)})
        (cache_dir / cache_key(paper_id)).write_text(json.dumps(coded, indent=2), encoding="utf-8")
        ok += 1
        print(progress_line("Gemini Fetch", index, len(lines), failed))
    state["fetched"] = True; state["fetched_at"] = datetime.now().isoformat(); state["cached"] = ok; state["failed"] = failed
    save_state(state_path, state)
    print(f"Fetched {ok:,} successful records; {failed:,} failures")
    if args.command in {"watch", "run"}:
        export(args.model, papers, cache_dir)


if __name__ == "__main__":
    main()
