"""Prepare, apply, and export human review of specification codes.

Inputs: model-coded cache records, confidence settings, and saved overrides.
Outputs: review queues, curation summaries, and curated paper-level data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aecsp.specification.schema import CURATABLE_SPECIFICATION_FIELDS

DEFAULT_ACCEPT_THRESHOLD = 0.8

# Per paper-dimension curation statuses on the exported dataset.
AUTO_ACCEPTED = "auto_accepted"
HUMAN_ACCEPTED = "human_accepted"
HUMAN_OVERRIDDEN = "human_overridden"
DEFERRED = "deferred"
LLM_UNREVIEWED = "llm_unreviewed"


def load_coded_records(cache_dir: Path) -> list[dict[str, Any]]:
    """All coded papers for one model, straight from the per-paper cache."""

    records = []
    for path in sorted(cache_dir.glob("*.json")):
        if path.name == "protocol_manifest.json":
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "paper_id" in record:
            records.append(record)
    return records


def is_auto_accepted(
    record: dict[str, Any], column: str, threshold: float = DEFAULT_ACCEPT_THRESHOLD
) -> bool:
    """Auto-accept only explicitly evidenced, high-confidence codes."""

    confidence = record.get(f"{column}_confidence")
    return (
        record.get(f"{column}_evidence_type") == "stated"
        and confidence is not None
        and float(confidence) >= threshold
    )


def empty_overrides() -> dict[str, Any]:
    return {"dimension_deferrals": [], "papers": {}}


def load_overrides(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_overrides()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("dimension_deferrals", [])
    data.setdefault("papers", {})
    return data


def save_overrides(path: Path, overrides: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def record_decision(
    overrides: dict[str, Any],
    paper_id: str,
    column: str,
    action: str,
    code: str | None = None,
    note: str = "",
) -> None:
    """Store one human decision ('accept' or 'override') in the overrides."""

    entry: dict[str, Any] = {"action": action, "decided_at": datetime.now().isoformat()}
    if code is not None:
        entry["code"] = code
    if note:
        entry["note"] = note
    overrides["papers"].setdefault(paper_id, {})[column] = entry


def build_review_queue(
    records: list[dict[str, Any]],
    threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Paper-dimension items needing a human, grouped by dimension and
    ordered least confident first within each dimension."""

    overrides = overrides or empty_overrides()
    queue: list[dict[str, Any]] = []
    for dimension in CURATABLE_SPECIFICATION_FIELDS:
        column = dimension.column
        if not any(column in record for record in records):
            continue
        if column in overrides["dimension_deferrals"]:
            continue
        items = []
        for record in records:
            paper_id = record.get("paper_id", "")
            if column in overrides["papers"].get(paper_id, {}):
                continue
            if is_auto_accepted(record, column, threshold):
                continue
            items.append(
                {
                    "paper_id": paper_id,
                    "column": column,
                    "question": dimension.question,
                    "allowed_values": list(dimension.allowed_values),
                    "code": record.get(column, ""),
                    "evidence": record.get(
                        f"{column}_evidence",
                        " | ".join(
                            value
                            for value in (
                                record.get("ai_role_function_evidence", ""),
                                record.get("ai_type_form_evidence", ""),
                            )
                            if value
                        ),
                    ),
                    "evidence_type": record.get(f"{column}_evidence_type", ""),
                    "confidence": record.get(f"{column}_confidence"),
                }
            )
        items.sort(
            key=lambda item: (item["confidence"] is not None, item["confidence"] or 0.0)
        )
        queue.extend(items)
    return queue


def curation_status(
    record: dict[str, Any],
    column: str,
    overrides: dict[str, Any],
    threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> tuple[str, str]:
    """Final (code, status) for one paper-dimension after curation.

    Precedence: human decision > corpus-wide deferral > auto-accept >
    unreviewed LLM code.
    """

    decision = overrides["papers"].get(record.get("paper_id", ""), {}).get(column)
    if decision is not None:
        if decision["action"] == "override":
            return decision.get("code", ""), HUMAN_OVERRIDDEN
        return record.get(column, ""), HUMAN_ACCEPTED
    if column in overrides["dimension_deferrals"]:
        return record.get(column, ""), DEFERRED
    if is_auto_accepted(record, column, threshold):
        return record.get(column, ""), AUTO_ACCEPTED
    return record.get(column, ""), LLM_UNREVIEWED


def curated_frame(
    records: list[dict[str, Any]],
    overrides: dict[str, Any],
    threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> pd.DataFrame:
    """Coded records with curation applied: final codes plus, per dimension,
    a <column>_curation status column."""

    rows = []
    for record in records:
        row = dict(record)
        for dimension in CURATABLE_SPECIFICATION_FIELDS:
            if dimension.column not in record:
                continue
            code, status = curation_status(record, dimension.column, overrides, threshold)
            row[dimension.column] = code
            row[f"{dimension.column}_curation"] = status
        rows.append(row)
    return pd.DataFrame(rows)


def status_report(
    records: list[dict[str, Any]],
    overrides: dict[str, Any],
    threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> pd.DataFrame:
    """Per-dimension counts of each curation status."""

    rows = []
    for dimension in CURATABLE_SPECIFICATION_FIELDS:
        if not any(dimension.column in record for record in records):
            continue
        counts = {
            AUTO_ACCEPTED: 0,
            HUMAN_ACCEPTED: 0,
            HUMAN_OVERRIDDEN: 0,
            DEFERRED: 0,
            LLM_UNREVIEWED: 0,
        }
        for record in records:
            _, status = curation_status(record, dimension.column, overrides, threshold)
            counts[status] += 1
        rows.append({"dimension": dimension.column, **counts})
    return pd.DataFrame(rows)
