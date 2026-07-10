"""Stage 2A.5: LLM-assisted AI specification coding (OpenAI-compatible API).

Codes each paper's title/abstract/keywords across the seven specification
dimensions defined in :mod:`aecsp.specification.schema`, returning the exact
brief-aligned columns plus, per dimension, the evidence, its epistemic status
(stated / inferred / absent), and a confidence score the curation step uses
to accept or queue each dimension.

Design (2026-07-10, with Kudzai):
- Provider: any OpenAI-compatible endpoint. OpenAI GPT for paid pilots (key
  from .env, never committed); local Ollama (http://localhost:11434/v1) for
  free end-to-end testing. Paid-pilot default gpt-4.1-nano, overridable with
  OPENAI_MODEL or --model.
- Structured output: JSON schema so every response parses into the
  controlled vocabularies; evidence is generated BEFORE the code within each
  dimension object so codes stay grounded.
- Epistemics: evidence vs inference labelled per dimension; unverifiable
  codes kept but flagged (needs_full_text), never silently dropped; the
  coder runs an adversarial pass on its own load-bearing codes.
- Cache: one JSON per paper per model under data/interim/spec_cache/<model>/
  so test codings (local models) never contaminate the final run and reruns
  only pay for uncoded papers.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from aecsp.specification.schema import (
    AI_DISTINCTION_TARGETS,
    EVIDENCE_TYPES,
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
    SPECIFICATION_PROBLEM_VALUES,
)

DEFAULT_MODEL = "gpt-4.1-nano"

AUXILIARY_PROPERTIES: dict[str, dict] = {
    "ai_method_or_phenomenon": {
        "type": "string",
        "enum": ["method", "phenomenon", "both", "unclear"],
        "description": "Is AI the research method used by the authors, or the phenomenon being studied?",
    },
    "process_sequence_specified": {
        "type": "string",
        "enum": ["yes", "no"],
        "description": "Does the paper specify when/how AI enters the entrepreneurial process?",
    },
    "ai_definition_present": {
        "type": "string",
        "enum": ["yes", "no"],
        "description": "Does the abstract indicate the paper defines AI?",
    },
    "ai_distinction_present": {
        "type": "string",
        "enum": ["yes", "no"],
        "description": (
            "Does the paper distinguish AI from adjacent constructs: "
            + ", ".join(AI_DISTINCTION_TARGETS)
        ),
    },
    "construct_clarity_score": {
        "type": "integer",
        "description": "Overall clarity of the AI construct in this paper, integer 0-100.",
    },
}

# Fields the model may flag as unjudgeable from the abstract alone.
NEEDS_FULL_TEXT_OPTIONS: tuple[str, ...] = tuple(
    dimension.column for dimension in SPECIFICATION_DIMENSIONS
) + ("theories_mentioned",)


def _dimension_schema(dimension) -> dict[str, Any]:
    """One dimension as an object; key order makes evidence precede the code."""

    return {
        "type": "object",
        "properties": {
            "evidence": {
                "type": "string",
                "description": (
                    "Short quote or close paraphrase from the title/abstract/"
                    "keywords supporting the code; empty when the text is silent."
                ),
            },
            "evidence_type": {
                "type": "string",
                "enum": list(EVIDENCE_TYPES),
                "description": (
                    "'stated' = the text explicitly supports the code; "
                    "'inferred' = your reasonable inference from context; "
                    "'absent' = the text does not address this dimension."
                ),
            },
            "code": {
                "type": "string",
                "enum": list(dimension.allowed_values),
                "description": dimension.question,
            },
            "confidence": {
                "type": "number",
                "description": (
                    "0.0-1.0. 0.9+ explicit statement; 0.6-0.8 strong "
                    "inference; below 0.6 weak inference needing human review. "
                    "Do not inflate."
                ),
            },
        },
        "required": ["evidence", "evidence_type", "code", "confidence"],
        "additionalProperties": False,
    }


def response_json_schema() -> dict[str, Any]:
    """JSON schema constraining the model to the controlled vocabularies."""

    properties: dict[str, Any] = {}
    for dimension in SPECIFICATION_DIMENSIONS:
        properties[dimension.column] = _dimension_schema(dimension)

    properties["ai_mechanism_logic"] = {
        "type": "string",
        "description": (
            "One sentence stating the causal logic in the paper's own terms "
            "(what AI does that changes the outcome); empty if the mechanism "
            "is missing."
        ),
    }
    properties["theories_mentioned"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": (
            "Named theories or frameworks as the text states them, with "
            "originators when given, e.g. 'Technology Acceptance Model "
            "(Davis 1989)'; empty array if none are named."
        ),
    }
    properties.update(AUXILIARY_PROPERTIES)
    properties[SPECIFICATION_PROBLEM_COLUMN] = {
        "type": "array",
        "items": {"type": "string", "enum": list(SPECIFICATION_PROBLEM_VALUES)},
        "description": "All construct specification problems diagnosed for this paper.",
    }
    properties["needs_full_text"] = {
        "type": "array",
        "items": {"type": "string", "enum": list(NEEDS_FULL_TEXT_OPTIONS)},
        "description": (
            "Dimensions that cannot be reliably coded from the abstract and "
            "need the full paper; typically definition fit and theory use."
        ),
    }
    properties["adversarial_review"] = {
        "type": "string",
        "description": (
            "One or two sentences: which load-bearing codes you attacked as a "
            "skeptic, and whether any were revised."
        ),
    }

    return {
        "name": "ai_specification_profile",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def _dimension_briefing() -> str:
    lines = []
    for dimension in SPECIFICATION_DIMENSIONS:
        lines.append(f"- {dimension.column}: {dimension.question}")
        if dimension.diagnosis:
            lines.append(f"  Diagnoses {dimension.diagnosis}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an expert construct-specification coder for a \
theory-elaboration study of how Artificial Intelligence is specified as a \
construct in entrepreneurship and business research. You code one paper at a \
time from its title, abstract, and author keywords only.

WHAT EACH DIMENSION DIAGNOSES (code with the diagnosis in mind):
{_dimension_briefing()}

CODING DISCIPLINE:
1. Evidence before code. For every dimension, first extract the evidence (a
   short quote or close paraphrase from the title/abstract/keywords), then
   choose the code. No evidence means an empty evidence field and the
   unspecified/missing code.
2. Separate what the text states from what you infer, and label it in
   evidence_type: 'stated' means the text explicitly supports the code
   (quote it); 'inferred' means the code is your reasonable inference from
   context (the evidence field says what you inferred from); 'absent' means
   the text does not address the dimension. Never present an inference as
   stated.
3. Score confidence per dimension, 0.0-1.0: 0.9+ explicit statement; 0.6-0.8
   strong inference; below 0.6 weak inference that a human must review. Do
   not inflate: an honest 0.4 is worth more than a false 0.8.
4. Unverifiable is not droppable. If a dimension truly requires the full
   paper (typically definition fit and theory use), still code your best
   estimate, mark evidence_type honestly, lower the confidence, and list the
   dimension in needs_full_text.
5. Adversarial pass before answering: identify your three most load-bearing
   codes for this paper, attack each as a skeptic paid to find the flaw
   (does the text actually say this, or am I pattern-matching on keywords?),
   revise any code that does not survive, and summarise the outcome in
   adversarial_review.
6. Conservative is not lazy: unspecified/missing is correct when the text is
   silent, and wrong when the text supports an inference you failed to make."""


def build_user_prompt(title: str, abstract: str, keywords: str, journal: str, year: str) -> str:
    return (
        "Code this paper across the seven AI construct-specification dimensions.\n\n"
        f"TITLE: {title}\n"
        f"JOURNAL: {journal} ({year})\n"
        f"KEYWORDS: {keywords}\n"
        f"ABSTRACT: {abstract}"
    )


def _clamp(value: Any, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(high, max(low, number))


def flatten_profile(raw: dict[str, Any]) -> dict[str, Any]:
    """Nested response -> flat cache/CSV record with predictable columns.

    Each dimension becomes <column>, <column>_evidence, <column>_evidence_type,
    <column>_confidence; array fields are ';'-joined ('; ' for theories, which
    contain commas).
    """

    flat: dict[str, Any] = {}
    for dimension in SPECIFICATION_DIMENSIONS:
        entry = raw.get(dimension.column)
        if not isinstance(entry, dict):
            entry = {}
        flat[dimension.column] = entry.get("code", "")
        flat[f"{dimension.column}_evidence"] = entry.get("evidence", "")
        flat[f"{dimension.column}_evidence_type"] = entry.get("evidence_type", "")
        flat[f"{dimension.column}_confidence"] = _clamp(entry.get("confidence"), 0.0, 1.0)

    flat["ai_mechanism_logic"] = raw.get("ai_mechanism_logic", "")
    flat["theories_mentioned"] = "; ".join(raw.get("theories_mentioned") or [])
    for column in AUXILIARY_PROPERTIES:
        flat[column] = raw.get(column, "")
    score = _clamp(raw.get("construct_clarity_score"), 0, 100)
    flat["construct_clarity_score"] = int(score) if score is not None else None
    flat[SPECIFICATION_PROBLEM_COLUMN] = ";".join(raw.get(SPECIFICATION_PROBLEM_COLUMN) or [])
    flat["needs_full_text"] = ";".join(raw.get("needs_full_text") or [])
    flat["adversarial_review"] = raw.get("adversarial_review", "")
    return flat


def load_env(env_path: Path) -> dict[str, str]:
    """Minimal .env parser (KEY=VALUE lines, # comments)."""

    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def model_cache_dir(cache_root: Path, model: str) -> Path:
    """Per-model cache directory so test models never pollute the real run."""

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cache_root / slug


def cache_key(paper_id: str) -> str:
    """Filesystem-safe cache filename for one paper."""

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(paper_id))[:80]
    digest = hashlib.sha1(str(paper_id).encode("utf-8")).hexdigest()[:10]
    return f"{safe}.{digest}.json"


def code_paper(
    client: Any,
    model: str,
    paper: dict[str, str],
    cache_dir: Path,
) -> dict[str, Any]:
    """Code one paper, using the cache when available."""

    cache_path = cache_dir / cache_key(paper["paper_id"])
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    paper.get("title", ""),
                    paper.get("abstract", ""),
                    paper.get("keywords", ""),
                    paper.get("journal", ""),
                    paper.get("year", ""),
                ),
            },
        ],
        response_format={"type": "json_schema", "json_schema": response_json_schema()},
        temperature=0,
    )
    coded = flatten_profile(json.loads(completion.choices[0].message.content))
    coded["paper_id"] = paper["paper_id"]
    coded["coding_model"] = model

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(coded, indent=2), encoding="utf-8")
    return coded
