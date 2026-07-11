"""Contract tests for the OpenAI Batch transport (no API calls)."""

import json
import re

from aecsp.specification.llm_coder import (
    SYSTEM_PROMPT,
    build_user_prompt,
    response_json_schema,
)
from aecsp.specification.openai_batch import (
    DEFAULT_TOKEN_BUDGET,
    build_body,
    custom_id_for,
    estimate_input_tokens,
    pack_batches,
    parse_result_line,
    request_line,
)

PAPER = {
    "paper_id": "eid:2-s2.0-000123/45?",
    "title": "AI and venture creation",
    "abstract": "Generative AI reshapes founding processes.",
    "keywords": "GenAI",
    "journal": "JBV",
    "year": "2025",
}


def test_batch_body_is_identical_to_the_live_instrument():
    body = build_body("gpt-5.4-mini-2026-03-17", PAPER, 4096)

    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert body["messages"][1]["content"] == build_user_prompt(
        PAPER["title"], PAPER["abstract"], PAPER["keywords"], PAPER["journal"], PAPER["year"]
    )
    assert body["response_format"] == {
        "type": "json_schema", "json_schema": response_json_schema()
    }
    # Frozen spec-v3 decoding settings, unchanged across transports.
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["seed"] == 42
    assert body["frequency_penalty"] == 0.0
    assert body["presence_penalty"] == 0.0
    assert body["n"] == 1
    assert body["stream"] is False
    assert body["max_completion_tokens"] == 4096


def test_custom_id_is_valid_stable_and_distinct():
    cid = custom_id_for(PAPER["paper_id"])
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid)
    assert cid == custom_id_for(PAPER["paper_id"])
    assert cid != custom_id_for("eid:2-s2.0-000123/45!")


def test_request_line_shape():
    line = request_line("gpt-5.4-mini-2026-03-17", PAPER, 4096)
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    assert line["custom_id"] == custom_id_for(PAPER["paper_id"])


def test_packing_respects_budget_and_preserves_order():
    papers = [dict(PAPER, paper_id=f"eid:{i}") for i in range(100)]
    per_paper = estimate_input_tokens(PAPER)
    budget = per_paper * 30 + 10
    chunks = pack_batches(papers, budget)

    assert all(len(chunk) <= 30 for chunk in chunks)
    flat = [p["paper_id"] for chunk in chunks for p in chunk]
    assert flat == [p["paper_id"] for p in papers]
    assert pack_batches(papers, DEFAULT_TOKEN_BUDGET)  # one big chunk is fine


def _line(body_overrides=None, error=None, status=200):
    body = {
        "choices": [{
            "finish_reason": "stop",
            "message": {"content": json.dumps({
                "ai_role_function": {"evidence": "q", "evidence_type": "stated",
                                     "code": "AI as tool", "confidence": 0.9},
            })},
        }],
        "usage": {"prompt_tokens": 3300, "completion_tokens": 480},
    }
    if body_overrides:
        body.update(body_overrides)
    return {"custom_id": "x", "error": error,
            "response": {"status_code": status, "body": body}}


def test_parse_result_success_flattens_and_carries_usage():
    coded, error = parse_result_line(_line())
    assert error is None
    assert coded["ai_role_function"] == "AI as tool"
    assert coded["prompt_tokens"] == 3300
    assert coded["output_tokens"] == 480


def test_parse_result_truncation_and_errors_never_cache():
    truncated = _line({"choices": [{"finish_reason": "length", "message": {"content": "{"}}]})
    coded, error = parse_result_line(truncated)
    assert coded is None and "output ceiling" in error

    coded, error = parse_result_line(_line(error={"code": "boom"}))
    assert coded is None and "batch request error" in error

    coded, error = parse_result_line(_line(status=500))
    assert coded is None and error.startswith("HTTP 500")
