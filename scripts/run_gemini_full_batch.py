"""Run Gemini full-corpus coding as sequential Tier-1-safe Batch shards.

Offline commands create deterministic paper-ID shards and never call Gemini.
The paid ``run`` command invokes ``run_gemini_batch.py`` for one shard at a
time, so active enqueued input remains well below the Tier 1 five-million-token
limit. Existing per-paper cache records are skipped automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from aecsp.specification.gemini_batch import estimate_cost  # noqa: E402
from aecsp.specification.llm_coder import (  # noqa: E402
    PROTOCOL_ID,
    cache_key,
    model_cache_dir,
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_TARGET = PROJECT_ROOT / "data/processed/master_corpus.csv"
DEFAULT_PLAN_ROOT = (
    PROJECT_ROOT / "data/interim/proprietary_validation/gemini_full_corpus_chunks"
)
DEFAULT_CHUNK_SIZE = 2_000
MAX_TIER_1_SAFE_CHUNK_SIZE = 2_500
OBSERVED_INPUT_TOKENS_PER_PAPER = 1_439.23
TIER_1_ENQUEUED_TOKEN_LIMIT = 5_000_000
CACHE_ROOT = PROJECT_ROOT / "data/interim/spec_cache"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(paper_ids: list[str]) -> str:
    payload = "\n".join(paper_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_target_ids(target: Path) -> list[str]:
    frame = pd.read_csv(target, dtype=str, keep_default_na=False, usecols=["paper_id"])
    paper_ids = frame["paper_id"].astype(str).str.strip().tolist()
    if any(not paper_id for paper_id in paper_ids):
        raise SystemExit("The full-corpus target contains an empty paper_id")
    if len(paper_ids) != len(set(paper_ids)):
        raise SystemExit("The full-corpus target contains duplicate paper_id values")
    return paper_ids


def plan_directory(root: Path, target: Path, run_id: str) -> Path:
    return root / f"{target.stem}_{run_id}"


def create_plan(
    *,
    model: str,
    target: Path,
    root: Path,
    run_id: str,
    chunk_size: int,
) -> tuple[Path, dict]:
    """Create or validate a deterministic cache-aware full-corpus shard plan."""

    if chunk_size < 1 or chunk_size > MAX_TIER_1_SAFE_CHUNK_SIZE:
        raise SystemExit(
            f"--chunk-size must be between 1 and {MAX_TIER_1_SAFE_CHUNK_SIZE:,} "
            "for the Tier-1-safe workflow"
        )
    target = target.resolve()
    plan_dir = plan_directory(root.resolve(), target, run_id)
    manifest_path = plan_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "model": model,
            "protocol": PROTOCOL_ID,
            "target_sha256": file_sha256(target),
            "chunk_size": chunk_size,
        }
        mismatches = [
            key for key, value in expected.items() if manifest.get(key) != value
        ]
        if mismatches:
            raise SystemExit(
                "Existing Gemini full-corpus plan does not match: "
                + ", ".join(mismatches)
                + ". Use a new --run-id rather than changing a frozen plan."
            )
        for shard in manifest["shards"]:
            path = plan_dir / shard["path"]
            if not path.exists() or file_sha256(path) != shard["sha256"]:
                raise SystemExit(f"Frozen shard is missing or changed: {path}")
        return plan_dir, manifest

    paper_ids = load_target_ids(target)
    cache_dir = model_cache_dir(CACHE_ROOT, model, PROTOCOL_ID)
    cached = {
        paper_id for paper_id in paper_ids if (cache_dir / cache_key(paper_id)).exists()
    }
    todo = [paper_id for paper_id in paper_ids if paper_id not in cached]
    plan_dir.mkdir(parents=True, exist_ok=False)
    shards = []
    for index, start in enumerate(range(0, len(todo), chunk_size)):
        shard_ids = todo[start : start + chunk_size]
        path = plan_dir / f"chunk_{index:04d}.csv"
        pd.DataFrame({"paper_id": shard_ids}).to_csv(path, index=False)
        expected_tokens = len(shard_ids) * OBSERVED_INPUT_TOKENS_PER_PAPER
        shards.append(
            {
                "index": index,
                "path": path.name,
                "sha256": file_sha256(path),
                "paper_ids_sha256": ids_sha256(shard_ids),
                "papers": len(shard_ids),
                "expected_input_tokens": round(expected_tokens),
                "tier_1_headroom_tokens": round(
                    TIER_1_ENQUEUED_TOKEN_LIMIT - expected_tokens
                ),
            }
        )
    manifest = {
        "created_at": datetime.now().isoformat(),
        "run_id": run_id,
        "provider": "gemini_batch_sequential_shards",
        "model": model,
        "protocol": PROTOCOL_ID,
        "target": display_path(target),
        "target_sha256": file_sha256(target),
        "target_papers": len(paper_ids),
        "target_paper_ids_sha256": ids_sha256(paper_ids),
        "cached_before_plan": len(cached),
        "papers_planned": len(todo),
        "chunk_size": chunk_size,
        "observed_input_tokens_per_paper": OBSERVED_INPUT_TOKENS_PER_PAPER,
        "tier_1_enqueued_token_limit": TIER_1_ENQUEUED_TOKEN_LIMIT,
        "submission_policy": "one shard active at a time",
        "estimated_batch_cost_usd": round(estimate_cost(len(todo)), 2),
        "shards": shards,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return plan_dir, manifest


def print_plan(plan_dir: Path, manifest: dict) -> None:
    print(
        f"Gemini full-corpus plan: {manifest['papers_planned']:,} uncached papers "
        f"in {len(manifest['shards']):,} sequential shard(s)"
    )
    print(
        f"Cached before plan: {manifest['cached_before_plan']:,}/"
        f"{manifest['target_papers']:,}"
    )
    print(f"Estimated Batch cost: ${manifest['estimated_batch_cost_usd']:.2f}")
    print(f"Manifest: {plan_dir / 'manifest.json'}")
    for shard in manifest["shards"]:
        print(
            f"  {shard['path']}: {shard['papers']:,} papers, "
            f"~{shard['expected_input_tokens']:,} input tokens, "
            f"~{shard['tier_1_headroom_tokens']:,} Tier 1 headroom"
        )


def batch_state_path(model: str, shard_path: Path) -> Path:
    cache_dir = model_cache_dir(CACHE_ROOT, model, PROTOCOL_ID)
    return cache_dir / "gemini_batches" / shard_path.stem / "batch_state.json"


def invoke_runner(
    command: str,
    *,
    model: str,
    paper_ids_file: Path,
    yes: bool = False,
    poll_seconds: int = 15,
) -> None:
    args = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_gemini_batch.py"),
        command,
        "--model",
        model,
        "--paper-ids-file",
        str(paper_ids_file),
        "--poll-seconds",
        str(poll_seconds),
    ]
    if yes:
        args.append("--yes")
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def run_sequential(
    plan_dir: Path,
    manifest: dict,
    *,
    model: str,
    poll_seconds: int,
    between_shards: int,
) -> None:
    for position, shard in enumerate(manifest["shards"], start=1):
        shard_path = plan_dir / shard["path"]
        print(
            f"\nGemini shard {position:,}/{len(manifest['shards']):,}: "
            f"{shard_path.name} ({shard['papers']:,} papers)"
        )
        invoke_runner(
            "run",
            model=model,
            paper_ids_file=shard_path,
            yes=True,
            poll_seconds=poll_seconds,
        )
        if position < len(manifest["shards"]) and between_shards:
            print(f"Waiting {between_shards}s before enqueuing the next shard")
            time.sleep(between_shards)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preview", "prepare", "run", "status", "export"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--paper-ids-file", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--run-id", default="initial")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--between-shards", type=int, default=30)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error("--poll-seconds must be at least 5")
    if args.between_shards < 0:
        parser.error("--between-shards cannot be negative")

    target = args.paper_ids_file.resolve()
    plan_dir, manifest = create_plan(
        model=args.model,
        target=target,
        root=args.plan_root,
        run_id=args.run_id,
        chunk_size=args.chunk_size,
    )
    print_plan(plan_dir, manifest)
    if args.command in {"preview", "prepare"}:
        print("No Gemini API request was made.")
        return
    if args.command == "export":
        invoke_runner("export", model=args.model, paper_ids_file=target)
        return
    if args.command == "status":
        for shard in manifest["shards"]:
            shard_path = plan_dir / shard["path"]
            state_path = batch_state_path(args.model, shard_path)
            if not state_path.exists():
                print(f"{shard_path.name}: NOT_SUBMITTED")
                continue
            invoke_runner("status", model=args.model, paper_ids_file=shard_path)
        return
    if not args.yes:
        raise SystemExit(
            "`run` submits PAID sequential Gemini batches. Review the manifest, "
            "confirm funding, then re-run with --yes."
        )
    run_sequential(
        plan_dir,
        manifest,
        model=args.model,
        poll_seconds=args.poll_seconds,
        between_shards=args.between_shards,
    )
    invoke_runner("export", model=args.model, paper_ids_file=target)


if __name__ == "__main__":
    main()
