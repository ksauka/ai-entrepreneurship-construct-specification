"""Offline contracts for tier-independent Gemini full-corpus sharding."""

import importlib.util
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from aecsp.specification.llm_coder import PROTOCOL_ID, cache_key


def load_script():
    path = Path(__file__).parents[1] / "scripts/run_gemini_full_batch.py"
    spec = importlib.util.spec_from_file_location("run_gemini_full_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_plan_is_deterministic_cache_aware_and_tier_one_safe(tmp_path):
    module = load_script()
    module.CACHE_ROOT = tmp_path / "cache"
    target = tmp_path / "full.csv"
    ids = [f"eid:{index}" for index in range(5_001)]
    pd.DataFrame({"paper_id": ids}).to_csv(target, index=False)

    cache_dir = (
        module.CACHE_ROOT / PROTOCOL_ID / module.DEFAULT_MODEL
    )
    cache_dir.mkdir(parents=True)
    (cache_dir / cache_key(ids[0])).write_text("{}")

    plan_dir, manifest = module.create_plan(
        model=module.DEFAULT_MODEL,
        target=target,
        root=tmp_path / "plans",
        run_id="initial",
        chunk_size=2_000,
    )
    assert manifest["target_papers"] == 5_001
    assert manifest["cached_before_plan"] == 1
    assert manifest["papers_planned"] == 5_000
    assert [item["papers"] for item in manifest["shards"]] == [2_000, 2_000, 1_000]
    assert all(item["tier_1_headroom_tokens"] > 0 for item in manifest["shards"])

    second_dir, second = module.create_plan(
        model=module.DEFAULT_MODEL,
        target=target,
        root=tmp_path / "plans",
        run_id="initial",
        chunk_size=2_000,
    )
    assert second_dir == plan_dir
    assert second == manifest


def test_frozen_plan_rejects_changed_target(tmp_path):
    module = load_script()
    module.CACHE_ROOT = tmp_path / "cache"
    target = tmp_path / "full.csv"
    pd.DataFrame({"paper_id": ["eid:1", "eid:2"]}).to_csv(target, index=False)
    module.create_plan(
        model=module.DEFAULT_MODEL,
        target=target,
        root=tmp_path / "plans",
        run_id="initial",
        chunk_size=2_000,
    )
    pd.DataFrame({"paper_id": ["eid:1", "eid:3"]}).to_csv(target, index=False)
    with pytest.raises(SystemExit, match="target_sha256"):
        module.create_plan(
            model=module.DEFAULT_MODEL,
            target=target,
            root=tmp_path / "plans",
            run_id="initial",
            chunk_size=2_000,
        )


def test_plan_refuses_oversized_tier_one_shard(tmp_path):
    module = load_script()
    target = tmp_path / "full.csv"
    pd.DataFrame({"paper_id": ["eid:1"]}).to_csv(target, index=False)
    with pytest.raises(SystemExit, match="Tier-1-safe"):
        module.create_plan(
            model=module.DEFAULT_MODEL,
            target=target,
            root=tmp_path / "plans",
            run_id="initial",
            chunk_size=2_501,
        )


def test_sequential_run_rebuilds_full_aggregate_after_each_shard(
    tmp_path, monkeypatch
):
    module = load_script()
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    shard = plan_dir / "chunk_0000.csv"
    pd.DataFrame({"paper_id": ["eid:1"]}).to_csv(shard, index=False)
    target = tmp_path / "full.csv"
    pd.DataFrame({"paper_id": ["eid:0", "eid:1"]}).to_csv(target, index=False)
    calls = []

    def record(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(module, "invoke_runner", record)
    module.run_sequential(
        plan_dir,
        {"shards": [{"path": shard.name, "papers": 1}]},
        model=module.DEFAULT_MODEL,
        poll_seconds=30,
        between_shards=0,
        aggregate_target=target,
    )
    assert calls[0][0] == "run"
    assert calls[0][1]["paper_ids_file"] == shard
    assert calls[0][1]["skip_export"] is True
    assert calls[1][0] == "export"
    assert calls[1][1]["paper_ids_file"] == target


def test_sequential_run_waits_and_retries_only_quota_rejection(
    tmp_path, monkeypatch
):
    module = load_script()
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    shard = plan_dir / "chunk_0000.csv"
    pd.DataFrame({"paper_id": ["eid:1"]}).to_csv(shard, index=False)
    target = tmp_path / "full.csv"
    pd.DataFrame({"paper_id": ["eid:1"]}).to_csv(target, index=False)
    calls = []
    sleeps = []

    def quota_then_success(command, **kwargs):
        calls.append(command)
        if command == "run" and calls.count("run") == 1:
            raise subprocess.CalledProcessError(
                module.QUOTA_RETRY_EXIT_CODE, ["gemini"]
            )

    monkeypatch.setattr(module, "invoke_runner", quota_then_success)
    monkeypatch.setattr(module.time, "sleep", sleeps.append)
    module.run_sequential(
        plan_dir,
        {"shards": [{"path": shard.name, "papers": 1}]},
        model=module.DEFAULT_MODEL,
        poll_seconds=30,
        between_shards=0,
        aggregate_target=target,
        quota_retry_seconds=300,
    )
    assert calls == ["run", "run", "export"]
    assert sleeps == [300]


def test_sequential_run_does_not_retry_nonquota_failure(tmp_path, monkeypatch):
    module = load_script()
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    shard = plan_dir / "chunk_0000.csv"
    pd.DataFrame({"paper_id": ["eid:1"]}).to_csv(shard, index=False)
    target = tmp_path / "full.csv"
    pd.DataFrame({"paper_id": ["eid:1"]}).to_csv(target, index=False)

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, ["gemini"])

    monkeypatch.setattr(module, "invoke_runner", fail)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_sequential(
            plan_dir,
            {"shards": [{"path": shard.name, "papers": 1}]},
            model=module.DEFAULT_MODEL,
            poll_seconds=30,
            between_shards=0,
            aggregate_target=target,
            quota_retry_seconds=300,
        )
