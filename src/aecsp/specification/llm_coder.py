"""Code paper metadata against the AI specification schema.

Inputs: a paper record and an OpenAI-compatible model client.
Outputs: validated specification codes, evidence, epistemic labels, confidence
scores, and model-specific cache records.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aecsp.specification.schema import (
    AI_DISTINCTION_TARGETS,
    EVIDENCE_TYPES,
    SPECIFICATION_DIMENSIONS,
    SPECIFICATION_PROBLEM_COLUMN,
    SPECIFICATION_PROBLEM_VALUES,
)

DEFAULT_MODEL = "gpt-4.1-nano-2025-04-14"

# Frozen cross-provider coding protocol. Changing any value requires a new
# protocol ID and therefore a new cache namespace.
# Protocol history:
# spec-v1 (retired): 1,200-token output ceiling truncated valid responses -
#   successful codings cluster at ~900-1,200 tokens, so the ceiling sat
#   inside the instrument's length distribution.
# spec-v2 (retired): uniform 4,096-token ceiling for every rater (per-model
#   ceilings were rejected as a comparability hazard). The 4,568-paper audit
#   of spec-v2 output exposed a mechanism leak: raters coded substantive
#   mechanisms while leaving ai_mechanism_logic EMPTY (27% of nano's
#   substantive codes) and flagging 'mechanism missing' as a problem anyway
#   (52%), undercounting the paper's core black-box diagnosis 6x (5% coded
#   vs 31% under the empty-logic rule). needs_full_text was also flagged as
#   a routine caveat (>90% of papers), destroying its signal value.
# spec-v3 (2026-07-11, CURRENT): identical schema and decoding settings;
#   adds two coding-discipline rules - a mechanism/logic coupling gate and
#   a needs_full_text discipline rule. All earlier caches are pilots and are
#   never mixed with spec-v3 experiment data.
PROTOCOL_ID = "spec-v3"
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
MAX_OUTPUT_TOKENS = 4096
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
COMPLETIONS_PER_PAPER = 1
REQUEST_TIMEOUT_SECONDS = 300.0
SDK_MAX_RETRIES = 2

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
   silent, and wrong when the text supports an inference you failed to make.
7. Mechanism requires causal logic. Code a substantive ai_mechanism ONLY if
   you can also state the paper's causal logic in ai_mechanism_logic, in the
   paper's own terms. If ai_mechanism_logic would be empty, generic, or a
   restatement of the code, the correct code is 'mechanism missing'. Never
   code a substantive mechanism while also flagging 'mechanism missing' in
   specification_problem: those two must agree.
8. needs_full_text is a signal, not a caveat. Flag a dimension only when the
   title/abstract/keywords are genuinely insufficient to code it. A dimension
   you coded from 'stated' evidence at high confidence should almost never
   appear there. Flagging most dimensions on most papers destroys the flag's
   meaning."""


def build_user_prompt(title: str, abstract: str, keywords: str, journal: str, year: str) -> str:
    return (
        "Code this paper across the seven AI construct-specification dimensions.\n\n"
        f"TITLE: {title}\n"
        f"JOURNAL: {journal} ({year})\n"
        f"KEYWORDS: {keywords}\n"
        f"ABSTRACT: {abstract}"
    )


# --------------------------------------------------------------------------
# spec-ft-v1: the full-text arm.
#
# A SEPARATE protocol, not a revision of spec-v3. Everything above stays
# byte-identical so the spec-v3 fingerprint cannot move and its existing caches
# remain valid.
#
# Only what the evidentiary boundary requires is changed. No new guidance is
# added (no "mechanisms are usually in the discussion" hints), because the
# paired abstract-versus-full-text comparison is interpretable only if the text
# supplied is the single thing that differs between the two arms.
#
# Rule 7 (mechanism requires causal logic) is preserved word for word: the
# pre-registered empty-logic correction depends on it.
#
# Rules 4 and 8 necessarily change. Both concerned metadata insufficiency and
# are incoherent once the full document is supplied. needs_full_text is retired
# for this protocol; document quality is recorded by the extraction pipeline.
# --------------------------------------------------------------------------

FULLTEXT_PROTOCOL_ID = "spec-ft-v1"
FULLTEXT_MAX_OUTPUT_TOKENS = 8192

SYSTEM_PROMPT_FULLTEXT = f"""You are an expert construct-specification coder for a \
theory-elaboration study of how Artificial Intelligence is specified as a \
construct in entrepreneurship and business research. You code one paper at a \
time from the full text of the paper as supplied to you. The supplied document \
contains the title, abstract, author keywords and the body of the paper. Its \
reference list and publisher apparatus have been removed and are not available \
to you.

WHAT EACH DIMENSION DIAGNOSES (code with the diagnosis in mind):
{_dimension_briefing()}

CODING DISCIPLINE:
1. Evidence before code. For every dimension, first extract the evidence (a
   short quote or close paraphrase from the supplied document), then choose
   the code. No evidence means an empty evidence field and the
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
4. Code what the supplied document supports. You have the full body, so a
   dimension the abstract alone could not settle should be coded from the
   text wherever the text addresses it. Where the document is genuinely
   silent on a dimension, the unspecified/missing code with 'absent'
   evidence_type is correct.
5. Adversarial pass before answering: identify your three most load-bearing
   codes for this paper, attack each as a skeptic paid to find the flaw
   (does the text actually say this, or am I pattern-matching on keywords?),
   revise any code that does not survive, and summarise the outcome in
   adversarial_review.
6. Conservative is not lazy: unspecified/missing is correct when the text is
   silent, and wrong when the text supports an inference you failed to make.
7. Mechanism requires causal logic. Code a substantive ai_mechanism ONLY if
   you can also state the paper's causal logic in ai_mechanism_logic, in the
   paper's own terms. If ai_mechanism_logic would be empty, generic, or a
   restatement of the code, the correct code is 'mechanism missing'. Never
   code a substantive mechanism while also flagging 'mechanism missing' in
   specification_problem: those two must agree.
8. Judge the paper, not the extraction. The document was produced by automated
   PDF extraction and may carry artefacts: broken words, lost table structure,
   or a missing section. Code what is present. Do not treat an extraction
   artefact as evidence that the paper failed to specify something, and do not
   speculate about content you cannot see."""


def build_user_prompt_fulltext(
    title: str, full_text: str, keywords: str, journal: str, year: str
) -> str:
    return (
        "Code this paper across the seven AI construct-specification dimensions.\n\n"
        f"TITLE: {title}\n"
        f"JOURNAL: {journal} ({year})\n"
        f"KEYWORDS: {keywords}\n"
        f"FULL TEXT:\n{full_text}"
    )


@dataclass(frozen=True)
class Protocol:
    """One coding protocol, declared as data rather than as control flow.

    Everything that distinguishes one protocol from another lives here, so a new
    protocol is a new entry in PROTOCOLS and nothing else. Branching on the
    protocol id inside prompt builders meant every future protocol would need
    another `if` in every builder, which is how instruments drift apart.

    text_field  the key in the paper record carrying the evidence text
    text_prefix the label placed before it, INCLUDING its separator, so that
                spec-v3 keeps "ABSTRACT: <text>" on one line and spec-ft-v1
                puts the document on its own line. Byte-identical output for
                spec-v3 is a contract test, not an aspiration.
    """

    protocol_id: str
    system_prompt: str
    max_output_tokens: int
    text_field: str
    text_prefix: str


PROTOCOLS: dict[str, Protocol] = {
    PROTOCOL_ID: Protocol(
        protocol_id=PROTOCOL_ID,
        system_prompt=SYSTEM_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        text_field="abstract",
        text_prefix="ABSTRACT: ",
    ),
    FULLTEXT_PROTOCOL_ID: Protocol(
        protocol_id=FULLTEXT_PROTOCOL_ID,
        system_prompt=SYSTEM_PROMPT_FULLTEXT,
        max_output_tokens=FULLTEXT_MAX_OUTPUT_TOKENS,
        text_field="full_text",
        text_prefix="FULL TEXT:\n",
    ),
}


def get_protocol(protocol_id: str | None = None) -> Protocol:
    """Look up a protocol, defaulting to the frozen spec-v3."""
    resolved = protocol_id or PROTOCOL_ID
    try:
        return PROTOCOLS[resolved]
    except KeyError:
        known = ", ".join(sorted(PROTOCOLS))
        raise ValueError(f"Unknown protocol '{resolved}'. Known protocols: {known}") from None


def build_paper_record(
    row: Any,
    text_dir: Path | None = None,
    protocol_id: str | None = None,
) -> dict[str, str]:
    """Assemble the record handed to a coder, for any protocol or transport.

    Shared by the live runner and both Batch runners so the three transports
    cannot drift in how they assemble a paper.

    Identifying metadata always comes from the corpus row. For a full-text
    protocol the evidence text is additionally read from
    <text_dir>/<paper_id>.md. That document already carries the corpus title,
    abstract and keywords at its head, written there by extract_fulltext.py, so
    switching protocols does not lose the abstract: it arrives inside the
    document rather than as a separate field.
    """
    protocol = get_protocol(protocol_id)
    paper = {
        "paper_id": row["paper_id"],
        "title": row.get("Title", ""),
        "abstract": row.get("Abstract", ""),
        "keywords": row.get("Author Keywords", ""),
        "journal": row.get("Source title", ""),
        "year": row.get("Year", ""),
    }
    if protocol.text_field == "abstract":
        if text_dir is not None:
            # Silently ignoring text_dir would let a full-text run code
            # abstracts with nobody noticing, so this is an error.
            raise ValueError(
                f"text_dir was supplied but protocol '{protocol.protocol_id}' reads the "
                f"abstract. Pass protocol_id='{FULLTEXT_PROTOCOL_ID}' for full-text runs."
            )
        return paper
    if text_dir is None:
        raise ValueError(
            f"Protocol '{protocol.protocol_id}' needs document text; pass --text-dir."
        )
    stem = str(row["paper_id"]).replace("eid:", "")
    md_path = Path(text_dir) / f"{stem}.md"
    if not md_path.is_file():
        raise FileNotFoundError(f"No cleaned full text for {row['paper_id']}: {md_path}")
    paper[protocol.text_field] = md_path.read_text(encoding="utf-8")
    return paper


def system_prompt_for(protocol_id: str | None = None) -> str:
    """The system prompt for a protocol."""
    return get_protocol(protocol_id).system_prompt


def max_output_tokens_for(protocol_id: str | None = None) -> int:
    """The output ceiling for a protocol. Ceilings are per protocol, never per model."""
    return get_protocol(protocol_id).max_output_tokens


def build_user_prompt_for(protocol_id: str | None, paper: dict[str, str]) -> str:
    """Build the user prompt for a protocol from a paper record.

    The record is assembled by the caller, so the arms differ only in which text
    it carries: spec-v3 reads 'abstract', spec-ft-v1 reads 'full_text'.
    """
    protocol = get_protocol(protocol_id)
    return (
        "Code this paper across the seven AI construct-specification dimensions.\n\n"
        f"TITLE: {paper.get('title', '')}\n"
        f"JOURNAL: {paper.get('journal', '')} ({paper.get('year', '')})\n"
        f"KEYWORDS: {paper.get('keywords', '')}\n"
        f"{protocol.text_prefix}{paper.get(protocol.text_field, '')}"
    )


def _clamp(value: Any, low: float, high: float) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(high, max(low, number))


def sanitize_lone_surrogates(value: Any) -> Any:
    """Replace invalid standalone UTF-16 surrogate code points recursively.

    Local model output can occasionally contain an escaped lone surrogate.
    Python's JSON parser preserves it in memory, but UTF-8 CSV and JSON
    writers cannot encode it. Replacing only those invalid code points with
    the Unicode replacement character preserves every valid character and
    keeps a successful coding record usable.
    """

    if isinstance(value, str):
        return "".join(
            "\N{REPLACEMENT CHARACTER}"
            if 0xD800 <= ord(character) <= 0xDFFF
            else character
            for character in value
        )
    if isinstance(value, dict):
        return {
            key: sanitize_lone_surrogates(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_lone_surrogates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_lone_surrogates(item) for item in value)
    return value


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


def protocol_for_model(model: str) -> tuple[str, int]:
    """Return the declared protocol and output ceiling for a model.

    Since spec-v2 the protocol is uniform across raters by design: identical
    prompt, schema, decoding settings, and output ceiling for every model, so
    rater comparisons are never confounded by instrument differences.
    """

    return PROTOCOL_ID, MAX_OUTPUT_TOKENS


def protocol_parameters(max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict[str, Any]:
    """Return the frozen decoding settings recorded with every experiment."""

    return {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_output_tokens": max_output_tokens,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "n": COMPLETIONS_PER_PAPER,
        "stream": False,
        "response_format": "strict_json_schema",
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "sdk_max_retries": SDK_MAX_RETRIES,
    }


def protocol_fingerprint(
    protocol_id: str = PROTOCOL_ID,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
) -> str:
    """Hash the prompt, schema, and decoding settings for audit purposes."""

    payload = {
        "protocol_id": protocol_id,
        "system_prompt": SYSTEM_PROMPT,
        "response_schema": response_json_schema(),
        "parameters": protocol_parameters(max_output_tokens),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_cache_dir(
    cache_root: Path, model: str, protocol_id: str | None = None
) -> Path:
    """Return an isolated cache directory for a model and coding protocol."""

    slug = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    return cache_root / protocol_id / slug if protocol_id else cache_root / slug


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
    *,
    local: bool = False,
    protocol_id: str | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """Code one paper, using the cache when available."""

    cache_path = cache_dir / cache_key(paper["paper_id"])
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    resolved_protocol, resolved_max_tokens = protocol_for_model(model)
    protocol_id = protocol_id or resolved_protocol
    max_output_tokens = max_output_tokens or resolved_max_tokens
    request = {
        "model": model,
        "messages": [
            # Protocol-selected. For spec-v3 these resolve to exactly the same
            # strings as before, so the request body stays byte-identical and
            # existing caches remain valid.
            {"role": "system", "content": system_prompt_for(protocol_id)},
            {"role": "user", "content": build_user_prompt_for(protocol_id, paper)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": response_json_schema(),
        },
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "n": COMPLETIONS_PER_PAPER,
        "stream": False,
    }
    # Ollama's OpenAI-compatible chat endpoint supports max_tokens; OpenAI's
    # current Chat Completions API uses max_completion_tokens.
    request["max_tokens" if local else "max_completion_tokens"] = max_output_tokens
    completion = client.chat.completions.create(**request)
    if completion.choices[0].finish_reason == "length":
        raise RuntimeError(
            f"Structured response exceeded {max_output_tokens} output tokens"
        )
    coded = sanitize_lone_surrogates(
        flatten_profile(json.loads(completion.choices[0].message.content))
    )
    coded["paper_id"] = paper["paper_id"]
    coded["coding_model"] = model
    # Actual token usage per paper: lets the reproducibility appendix report
    # the output-length distribution and verify the ceiling has headroom.
    usage = getattr(completion, "usage", None)
    coded["prompt_tokens"] = getattr(usage, "prompt_tokens", None) if usage else None
    coded["output_tokens"] = getattr(usage, "completion_tokens", None) if usage else None
    coded["coding_protocol"] = protocol_id
    coded["coding_protocol_fingerprint"] = protocol_fingerprint(
        protocol_id, max_output_tokens
    )
    coded["coding_parameters_json"] = json.dumps(
        protocol_parameters(max_output_tokens), sort_keys=True
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(coded, indent=2), encoding="utf-8")
    return coded
