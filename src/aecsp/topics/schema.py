"""Topic and keyword assignment records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicAssignment:
    """One topic, keyword, or keyphrase assignment for a publication."""

    publication_id: str
    label: str
    extraction_method: str
    score: float | None = None
    source_column: str | None = None
