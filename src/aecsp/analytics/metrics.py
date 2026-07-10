"""Metrics for construct specification analysis."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionMetric:
    """Summary of convergence/divergence for one categorical dimension."""

    total: int
    unique_values: int
    dominant_value: str | None
    dominant_share: float
    entropy: float
    convergence_score: float


def summarize_distribution(values: Iterable[str | None]) -> DistributionMetric:
    """Summarize how concentrated or fragmented a categorical variable is."""

    clean_values = [value.strip() for value in values if value and value.strip()]
    total = len(clean_values)
    if total == 0:
        return DistributionMetric(
            total=0,
            unique_values=0,
            dominant_value=None,
            dominant_share=0.0,
            entropy=0.0,
            convergence_score=0.0,
        )

    counts = Counter(clean_values)
    dominant_value, dominant_count = counts.most_common(1)[0]
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy else 0.0

    return DistributionMetric(
        total=total,
        unique_values=len(counts),
        dominant_value=dominant_value,
        dominant_share=dominant_count / total,
        entropy=normalized_entropy,
        convergence_score=1.0 - normalized_entropy,
    )
