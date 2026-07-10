"""Contract tests for the Stage 2A.5 LLM coder (no API calls)."""

import json
from pathlib import Path

from aecsp.specification.llm_coder import (
    SYSTEM_PROMPT,
    build_user_prompt,
    cache_key,
    code_paper,
    flatten_profile,
    load_env,
    model_cache_dir,
    response_json_schema,
)
from aecsp.specification.schema import (
    EVIDENCE_TYPES,
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
)


def test_schema_covers_all_seven_dimensions_with_epistemics():
    schema = response_json_schema()["schema"]

    for dimension in SPECIFICATION_DIMENSIONS:
        entry = schema["properties"][dimension.column]
        assert entry["type"] == "object"
        # Evidence must be generated before the code so codes stay grounded.
        assert list(entry["properties"]) == ["evidence", "evidence_type", "code", "confidence"]
        assert entry["properties"]["code"]["enum"] == list(dimension.allowed_values)
        assert entry["properties"]["evidence_type"]["enum"] == list(EVIDENCE_TYPES)
        assert entry["additionalProperties"] is False
    for column in (SPECIFICATION_PROBLEM_COLUMN, "theories_mentioned",
                   "needs_full_text", "adversarial_review", "ai_mechanism_logic"):
        assert column in schema["properties"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_system_prompt_carries_dimensions_diagnoses_and_discipline():
    # Normalise whitespace: prompt line wrapping must not hide an anchor.
    prose = " ".join(SYSTEM_PROMPT.split())
    for dimension in SPECIFICATION_DIMENSIONS:
        assert dimension.column in prose
        assert dimension.diagnosis, f"{dimension.id} is missing its diagnosis"
    for anchor in ("stated", "inferred", "absent", "Adversarial pass",
                   "needs_full_text", "Do not inflate"):
        assert anchor in prose


def test_user_prompt_contains_paper_fields():
    prompt = build_user_prompt(
        "AI and venture creation", "Generative AI reshapes founding.", "GenAI", "JBV", "2025"
    )
    assert "AI and venture creation" in prompt
    assert "Generative AI reshapes founding." in prompt
    assert "JBV (2025)" in prompt


def test_flatten_profile_produces_curatable_columns():
    raw = {
        "ai_role_function": {
            "evidence": "AI acts as a co-founder",
            "evidence_type": "stated",
            "code": "AI as actor/agent",
            "confidence": 1.7,  # out of range: must clamp to 1.0
        },
        "ai_mechanism_logic": "AI expands search, shifting the bottleneck to evaluation.",
        "theories_mentioned": ["Effectuation (Sarasvathy 2001)", "RBV (Barney 1991)"],
        "construct_clarity_score": 250,  # out of range: must clamp to 100
        SPECIFICATION_PROBLEM_COLUMN: ["role ambiguity", "mechanism missing"],
        "needs_full_text": ["definition_construct_clarity"],
        "adversarial_review": "Challenged actor coding; survived.",
    }

    flat = flatten_profile(raw)

    assert flat["ai_role_function"] == "AI as actor/agent"
    assert flat["ai_role_function_evidence"] == "AI acts as a co-founder"
    assert flat["ai_role_function_evidence_type"] == "stated"
    assert flat["ai_role_function_confidence"] == 1.0
    assert flat["construct_clarity_score"] == 100
    assert flat["theories_mentioned"] == "Effectuation (Sarasvathy 2001); RBV (Barney 1991)"
    assert flat[SPECIFICATION_PROBLEM_COLUMN] == "role ambiguity;mechanism missing"
    assert flat["needs_full_text"] == "definition_construct_clarity"
    # Dimensions absent from the response still get their columns.
    assert flat["ai_type_form"] == ""
    assert flat["ai_type_form_confidence"] is None


def test_cache_key_is_filesystem_safe_and_stable():
    key = cache_key("eid:2-s2.0-85/12345?x")
    assert "/" not in key and "?" not in key
    assert key == cache_key("eid:2-s2.0-85/12345?x")
    assert key != cache_key("eid:2-s2.0-85/12345?y")


def test_cache_is_isolated_per_model(tmp_path: Path):
    local = model_cache_dir(tmp_path, "llama3")
    paid = model_cache_dir(tmp_path, "gpt-4.1-nano")
    assert local != paid
    assert local.parent == paid.parent == tmp_path


def test_code_paper_uses_cache_without_client(tmp_path: Path):
    paper_id = "eid:test1"
    cached = {"paper_id": paper_id, "ai_role_function": "AI as tool"}
    (tmp_path / cache_key(paper_id)).write_text(json.dumps(cached), encoding="utf-8")

    result = code_paper(client=None, model="none", paper={"paper_id": paper_id}, cache_dir=tmp_path)

    assert result["ai_role_function"] == "AI as tool"


def test_load_env_parses_key_values(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nOPENAI_API_KEY='sk-abc'\nOPENAI_MODEL=gpt-4o-mini\n")

    env = load_env(env_file)

    assert env["OPENAI_API_KEY"] == "sk-abc"
    assert env["OPENAI_MODEL"] == "gpt-4o-mini"
