"""Agreement metrics for human and model specification ratings.

Inputs: aligned categorical ratings keyed by paper and dimension. Outputs:
pairwise percent agreement and nominal Krippendorff alpha with missing ratings
excluded pairwise. These metrics do not treat any rater as an infallible gold
standard.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class PairwiseAgreement:
    comparable: int
    agreements: int
    percent_agreement: float | None


def pairwise_percent_agreement(
    left: Iterable[object], right: Iterable[object]
) -> PairwiseAgreement:
    """Calculate exact categorical agreement for aligned rating sequences."""

    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("rating sequences must have equal length")
    pairs = [
        (a, b)
        for a, b in zip(map(_clean, left_values), map(_clean, right_values))
        if a is not None and b is not None
    ]
    agreements = sum(a == b for a, b in pairs)
    return PairwiseAgreement(
        comparable=len(pairs),
        agreements=agreements,
        percent_agreement=(agreements / len(pairs)) if pairs else None,
    )


def krippendorff_alpha_nominal(units: Sequence[Sequence[object]]) -> float | None:
    """Calculate Krippendorff's alpha for nominal ratings by unit.

    Each inner sequence contains all available rater values for one paper.
    Units with fewer than two non-missing ratings do not contribute observed
    disagreement. Returns ``None`` when agreement is not estimable.
    """

    observed_disagreements = 0
    observed_pairs = 0
    pooled: Counter[str] = Counter()
    for unit in units:
        values = [value for value in map(_clean, unit) if value is not None]
        pooled.update(values)
        for index, left in enumerate(values):
            for right in values[index + 1 :]:
                observed_pairs += 1
                observed_disagreements += left != right
    if observed_pairs == 0:
        return None
    total = sum(pooled.values())
    if total < 2:
        return None
    expected_disagreement = 1.0 - sum(
        count * (count - 1) for count in pooled.values()
    ) / (total * (total - 1))
    observed_disagreement = observed_disagreements / observed_pairs
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1.0 - observed_disagreement / expected_disagreement
