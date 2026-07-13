"""Contract tests for Gemini Batch request/response logic."""

import json
import importlib.util
from pathlib import Path

from aecsp.specification.gemini_batch import (
    custom_id_for,
    generation_config,
    parse_result_line,
    request_line,
)
from aecsp.specification.llm_coder import SYSTEM_PROMPT, response_json_schema

PAPER = {
    "paper_id": "eid:1", "title": "AI ventures", "abstract": "AI predicts demand.",
    "keywords": "AI", "journal": "J", "year": "2025",
}


def load_script():
    path = Path(__file__).parents[1] / "scripts/run_gemini_batch.py"
    spec = importlib.util.spec_from_file_location("run_gemini_batch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_request_uses_frozen_prompt_schema_and_controls():
    line = request_line("gemini-3.1-pro-preview", PAPER)
    request = line["request"]
    assert line["key"] == custom_id_for("eid:1")
    assert request["system_instruction"]["parts"][0]["text"] == SYSTEM_PROMPT
    config = request["generation_config"]
    assert config["response_json_schema"] == response_json_schema()["schema"]
    assert config["temperature"] == 1.0
    assert config["seed"] == 42
    assert config["max_output_tokens"] == 4096
    assert config["thinking_config"] == {"thinking_level": "low"}


def test_parse_result_success_and_truncation():
    profile = {"ai_role_function": {"evidence": "AI predicts", "evidence_type": "stated", "code": "AI as tool", "confidence": 0.9}}
    item = {"response": {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(profile)}]}}], "usageMetadata": {"promptTokenCount": 3000, "candidatesTokenCount": 400, "thoughtsTokenCount": 20}}}
    coded, error = parse_result_line(item)
    assert error is None
    assert coded["ai_role_function"] == "AI as tool"
    assert coded["thinking_tokens"] == 20

    coded, error = parse_result_line({"response": {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]}})
    assert coded is None and "output ceiling" in error


def test_target_batch_directory_isolated_by_manifest():
    module = load_script()
    cache = Path("cache")
    assert module.target_batch_dir(cache, Path("pilot.csv")) == cache / "gemini_batches" / "pilot"
    assert module.target_batch_dir(cache, Path("full.csv")) != module.target_batch_dir(cache, Path("pilot.csv"))


def test_progress_line_reports_percentage_and_failures():
    module = load_script()
    line = module.progress_line("Gemini Batch", 25, 50, 2)
    assert "25/50" in line
    assert "50.00%" in line
    assert "failures 2" in line


def test_job_success_never_infers_paper_success_without_counts():
    module = load_script()
    class State:
        name = "JOB_STATE_SUCCEEDED"
    class Job:
        state = State()
        completion_stats = None
    assert module.completion_counts(Job(), 50) == (0, 0)
