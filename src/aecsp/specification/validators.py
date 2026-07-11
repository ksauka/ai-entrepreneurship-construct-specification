"""Non-mutating quality-control checks for specification records."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from aecsp.specification.schema import SPECIFICATION_DIMENSIONS


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    dimension: str | None
    severity: str
    message: str


def token_overlap(evidence: str, source: str) -> float:
    tokens = set(re.findall(r"[a-z0-9]+", str(evidence).lower()))
    source_tokens = set(re.findall(r"[a-z0-9]+", str(source).lower()))
    return len(tokens & source_tokens) / len(tokens) if tokens else 1.0


def validate_profile(record: dict, paper: dict, min_overlap: float = 0.5) -> dict:
    issues: list[ValidationIssue] = []
    scores: dict[str, float] = {}
    mechanism = str(record.get("ai_mechanism", "")).strip()
    logic = str(record.get("ai_mechanism_logic", "")).strip()
    if mechanism and mechanism != "mechanism missing" and not logic:
        issues.append(ValidationIssue("RULE7_EMPTY_LOGIC", "ai_mechanism", "error", "Substantive mechanism has no causal logic."))
    needs = {part.strip() for part in str(record.get("needs_full_text", "")).split(";") if part.strip()}
    source = " ".join(str(paper.get(key, "")) for key in ("title", "abstract", "keywords"))
    for dimension in SPECIFICATION_DIMENSIONS:
        column = dimension.column
        if column in needs and record.get(f"{column}_evidence_type") == "stated" and float(record.get(f"{column}_confidence") or 0) >= 0.8:
            issues.append(ValidationIssue("RULE8_NFT_CONFLICT", column, "warning", "High-confidence stated evidence conflicts with needs_full_text."))
        evidence = str(record.get(f"{column}_evidence", ""))
        score = token_overlap(evidence, source)
        scores[column] = round(score, 4)
        if evidence and score < min_overlap:
            issues.append(ValidationIssue("GROUNDING_LOW", column, "warning", f"Evidence token overlap is {score:.2f}."))
    return {
        "validation_passed": not any(issue.severity == "error" for issue in issues),
        "validation_error_count": sum(issue.severity == "error" for issue in issues),
        "validation_warning_count": sum(issue.severity == "warning" for issue in issues),
        "validation_issues": [asdict(issue) for issue in issues],
        "grounding_scores": scores,
    }
