"""OpenAI Batch API transport for the specification protocol.

Same frozen spec-v3 instrument as the live path in llm_coder.code_paper -
identical system prompt, user prompt, strict JSON schema, and decoding
parameters - at 50% of live prices. Only the transport differs, so Batch and
live runs share one per-paper cache and resume each other.

Tier constraint (verified 2026-07-11): the Batch queue limit is the maximum
input tokens ENQUEUED AT ONE TIME (not per day). At a 2,000,000-token queue
limit the ~85M-token corpus is processed as sequential chunks, one batch
enqueued at a time, each under the queue budget. The orchestrator in
scripts/run_specification_openai_batch.py owns the submit/poll/fetch loop;
this module holds the testable pure logic.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from aecsp.specification.llm_coder import (
    COMPLETIONS_PER_PAPER,
    FREQUENCY_PENALTY,
    PRESENCE_PENALTY,
    SEED,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_user_prompt,
    flatten_profile,
    response_json_schema,
)

# Observed on completed spec-v3 records (nano and the 50-paper gpt-5.4-mini
# gate agree within a few percent): used for cost preview and chunk packing.
OBSERVED_INPUT_TOKENS = 3350
OBSERVED_OUTPUT_TOKENS = 500
# Fixed request overhead (system prompt + schema + template) in tokens; the
# per-paper remainder scales with the metadata text length.
REQUEST_OVERHEAD_TOKENS = 2900
BATCH_DISCOUNT = 0.5
# Keep each chunk safely under the 2,000,000-token enqueued budget.
DEFAULT_TOKEN_BUDGET = 1_850_000


def custom_id_for(paper_id: str) -> str:
    """Batch custom_id: <=64 chars of [A-Za-z0-9_-], stable and distinct."""

    safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(paper_id))[:40]
    digest = hashlib.sha1(str(paper_id).encode("utf-8")).hexdigest()[:16]
    return f"{safe}_{digest}"


def build_body(model: str, paper: dict[str, str], max_output_tokens: int) -> dict[str, Any]:
    """Chat Completions body identical to the live code_paper request."""

    return {
        "model": model,
        "messages": [
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
        "max_completion_tokens": max_output_tokens,
    }


def request_line(model: str, paper: dict[str, str], max_output_tokens: int) -> dict[str, Any]:
    """One JSONL line for the batch input file."""

    return {
        "custom_id": custom_id_for(paper["paper_id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": build_body(model, paper, max_output_tokens),
    }


def estimate_input_tokens(paper: dict[str, str]) -> int:
    """Conservative per-request input estimate for queue-budget packing."""

    text = " ".join(
        paper.get(key, "") for key in ("title", "abstract", "keywords", "journal")
    )
    return REQUEST_OVERHEAD_TOKENS + max(50, len(text) // 4)


def pack_batches(papers: list[dict[str, str]], token_budget: int = DEFAULT_TOKEN_BUDGET) -> list[list[dict[str, str]]]:
    """Split papers into ordered chunks whose estimated input stays under budget."""

    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    used = 0
    for paper in papers:
        cost = estimate_input_tokens(paper)
        if current and used + cost > token_budget:
            chunks.append(current)
            current, used = [], 0
        current.append(paper)
        used += cost
    if current:
        chunks.append(current)
    return chunks


def parse_result_line(line_obj: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """(flattened profile, error) for one batch output line.

    The caller resolves custom_id -> paper_id and adds protocol metadata.
    """

    error = line_obj.get("error")
    if error:
        return None, f"batch request error: {error}"
    response = line_obj.get("response") or {}
    if response.get("status_code") != 200:
        return None, f"HTTP {response.get('status_code')}: {json.dumps(response.get('body'))[:300]}"
    body = response.get("body") or {}
    choice = (body.get("choices") or [{}])[0]
    finish = choice.get("finish_reason")
    if finish == "length":
        return None, "structured response exceeded the output ceiling (finish_reason=length)"
    if finish == "content_filter":
        return None, "provider content filter (finish_reason=content_filter)"
    content = (choice.get("message") or {}).get("content")
    if not content:
        return None, f"empty response content (finish_reason={finish})"
    try:
        coded = flatten_profile(json.loads(content))
    except Exception as parse_error:
        return None, f"response did not parse as the profile schema: {parse_error}"
    usage = body.get("usage") or {}
    coded["prompt_tokens"] = usage.get("prompt_tokens")
    coded["output_tokens"] = usage.get("completion_tokens")
    return coded, None


def estimate_cost(papers: int, live_prices: tuple[float, float]) -> float:
    """Batch cost preview in USD from observed per-paper token averages."""

    in_price, out_price = live_prices
    live = papers * (
        OBSERVED_INPUT_TOKENS * in_price + OBSERVED_OUTPUT_TOKENS * out_price
    ) / 1_000_000
    return live * BATCH_DISCOUNT
