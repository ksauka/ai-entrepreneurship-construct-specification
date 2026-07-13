"""Offline contracts for the Anthropic Batch request shape."""

import importlib.util
from pathlib import Path

from aecsp.specification.llm_coder import SYSTEM_PROMPT, response_json_schema


def load_script():
    path = Path(__file__).parents[1] / "scripts/run_anthropic_batch.py"
    spec = importlib.util.spec_from_file_location("run_anthropic_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_anthropic_request_uses_schema_and_disables_thinking():
    module = load_script()
    paper = {"paper_id": "eid:1", "title": "AI", "abstract": "A", "keywords": "K", "journal": "J", "year": "2025"}
    params = module.request_for_paper("claude-sonnet-5", paper)["params"]
    assert params["system"] == SYSTEM_PROMPT
    assert params["output_config"]["format"]["schema"] == response_json_schema()["schema"]
    assert params["thinking"] == {"type": "disabled"}
    assert "temperature" not in params


def test_target_batch_directory_isolated_by_manifest():
    module = load_script()
    cache = Path("cache")
    assert module.target_batch_dir(cache, Path("pilot.csv")) == cache / "anthropic_batches" / "pilot"
    assert module.target_batch_dir(cache, Path("full.csv")) != module.target_batch_dir(cache, Path("pilot.csv"))


def test_progress_line_reports_percentage_and_failures():
    module = load_script()
    line = module.progress_line("Claude Batch", 25, 50, 2)
    assert "25/50" in line
    assert "50.00%" in line
    assert "failures 2" in line
