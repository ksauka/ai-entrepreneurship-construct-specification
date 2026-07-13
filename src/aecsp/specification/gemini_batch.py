"""Pure request and response logic for Gemini specification batches."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from aecsp.specification.llm_coder import (
    MAX_OUTPUT_TOKENS,
    SEED,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    build_user_prompt,
    flatten_profile,
    response_json_schema,
)

GEMINI_BATCH_INPUT_PRICE = 1.00
GEMINI_BATCH_OUTPUT_PRICE = 6.00


def custom_id_for(paper_id: str) -> str:
    return "paper_" + hashlib.sha1(str(paper_id).encode("utf-8")).hexdigest()


def generation_config(max_output_tokens: int = MAX_OUTPUT_TOKENS) -> dict[str, Any]:
    return {
        # Gemini 3.1 Pro is optimized for its provider-default temperature.
        "temperature": 1.0,
        "top_p": TOP_P,
        "seed": SEED,
        "max_output_tokens": max_output_tokens,
        "candidate_count": 1,
        "response_mime_type": "application/json",
        "response_json_schema": response_json_schema()["schema"],
        "thinking_config": {"thinking_level": "low"},
    }


def request_line(model: str, paper: dict[str, str]) -> dict[str, Any]:
    return {
        "key": custom_id_for(paper["paper_id"]),
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": build_user_prompt(
                                paper.get("title", ""),
                                paper.get("abstract", ""),
                                paper.get("keywords", ""),
                                paper.get("journal", ""),
                                paper.get("year", ""),
                            )
                        }
                    ],
                }
            ],
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generation_config": generation_config(),
        },
    }


def parse_result_line(item: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if item.get("error"):
        return None, f"batch request error: {item['error']}"
    response = item.get("response") or item.get("inlineResponse", {}).get("response") or {}
    candidates = response.get("candidates") or []
    if not candidates:
        return None, f"no response candidate: {json.dumps(item)[:300]}"
    candidate = candidates[0]
    finish = str(candidate.get("finishReason") or candidate.get("finish_reason") or "")
    if finish.upper() in {"MAX_TOKENS", "LENGTH"}:
        return None, "structured response exceeded the output ceiling"
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(str(part.get("text", "")) for part in parts if part.get("text"))
    if not text:
        return None, f"empty response content (finish_reason={finish})"
    try:
        coded = flatten_profile(json.loads(text))
    except Exception as error:
        return None, f"response did not parse as the profile schema: {error}"
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    coded["prompt_tokens"] = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    coded["output_tokens"] = usage.get("candidatesTokenCount") or usage.get("candidates_token_count")
    coded["thinking_tokens"] = usage.get("thoughtsTokenCount") or usage.get("thoughts_token_count")
    return coded, None


def estimate_cost(
    papers: int,
    input_tokens: float = 1444.6,
    output_tokens: float = 1578.6,
) -> float:
    """Estimate Batch cost from the 50-paper 3.1 Pro pilot.

    Output includes both visible response tokens and billable thinking tokens.
    """
    return papers * (
        input_tokens * GEMINI_BATCH_INPUT_PRICE
        + output_tokens * GEMINI_BATCH_OUTPUT_PRICE
    ) / 1_000_000
