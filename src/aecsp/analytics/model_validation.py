"""Model-comparison statistics for specification validation.

Inputs are paper-aligned categorical model outputs. Outputs are prevalence,
pairwise agreement, nominal Krippendorff alpha, bootstrap intervals, and
multi-rater agreement summaries without treating any model as ground truth.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence

import numpy as np

from aecsp.analytics.agreement import krippendorff_alpha_nominal


CORE_DIMENSIONS = (
    "ai_method_or_phenomenon",
    "ai_type_form",
    "ai_role_function",
    "ai_mechanism_analysis",
    "level_of_analysis",
    "scope_conditions",
)

EXPLORATORY_DIMENSIONS = (
    "entrepreneurial_process_stage",
    "definition_construct_clarity",
)

SUPPLEMENTARY_DIAGNOSTIC_DIMENSIONS = (
    "process_sequence_specified",
    "ai_definition_present",
    "ai_distinction_present",
)

DIMENSIONS = (
    *CORE_DIMENSIONS,
    *EXPLORATORY_DIMENSIONS,
    *SUPPLEMENTARY_DIAGNOSTIC_DIMENSIONS,
)


def clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def nominal_alpha_pair(left: Sequence[object], right: Sequence[object]) -> float | None:
    return krippendorff_alpha_nominal([list(pair) for pair in zip(left, right)])


def _alpha_from_codes(left: np.ndarray, right: np.ndarray, categories: int) -> float | None:
    """Vectorized nominal alpha for two complete categorical rating arrays."""

    n = len(left)
    if n == 0:
        return None
    observed = float(np.mean(left != right))
    counts = np.bincount(np.concatenate((left, right)), minlength=categories)
    total = 2 * n
    expected = 1.0 - float(np.sum(counts * (counts - 1))) / (total * (total - 1))
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % (2**32)


def pairwise_with_bootstrap(
    left: Sequence[object],
    right: Sequence[object],
    *,
    weights: Sequence[float] | None = None,
    repetitions: int = 2_000,
    seed: int = 20_260_711,
) -> dict[str, float | int | None]:
    """Exact agreement and nominal alpha with paper-level percentile CIs."""

    rows = [
        (clean(a), clean(b), float(weights[index]) if weights is not None else 1.0)
        for index, (a, b) in enumerate(zip(left, right))
        if clean(a) is not None and clean(b) is not None
    ]
    if not rows:
        return {"comparable": 0, "agreements": 0, "percent_agreement": None,
                "weighted_percent_agreement": None, "krippendorff_alpha": None,
                "agreement_ci_low": None, "agreement_ci_high": None,
                "alpha_ci_low": None, "alpha_ci_high": None}
    a = np.array([row[0] for row in rows], dtype=object)
    b = np.array([row[1] for row in rows], dtype=object)
    w = np.array([row[2] for row in rows], dtype=float)
    equal = a == b
    agreement = float(equal.mean())
    weighted = float(np.average(equal, weights=w)) if w.sum() else None
    categories = {value: index for index, value in enumerate(sorted(set(a) | set(b)))}
    a_codes = np.array([categories[value] for value in a], dtype=np.int32)
    b_codes = np.array([categories[value] for value in b], dtype=np.int32)
    alpha = _alpha_from_codes(a_codes, b_codes, len(categories))
    rng = np.random.default_rng(seed)
    agreement_draws = np.empty(repetitions)
    alpha_draws = np.empty(repetitions)
    for repetition in range(repetitions):
        index = rng.integers(0, len(a), len(a))
        aa, bb = a_codes[index], b_codes[index]
        agreement_draws[repetition] = np.mean(aa == bb)
        value = _alpha_from_codes(aa, bb, len(categories))
        alpha_draws[repetition] = np.nan if value is None else value
    valid_alpha = alpha_draws[~np.isnan(alpha_draws)]
    return {
        "comparable": len(a), "agreements": int(equal.sum()),
        "percent_agreement": agreement, "weighted_percent_agreement": weighted,
        "krippendorff_alpha": alpha,
        "agreement_ci_low": float(np.quantile(agreement_draws, 0.025)),
        "agreement_ci_high": float(np.quantile(agreement_draws, 0.975)),
        "alpha_ci_low": float(np.quantile(valid_alpha, 0.025)) if len(valid_alpha) else None,
        "alpha_ci_high": float(np.quantile(valid_alpha, 0.975)) if len(valid_alpha) else None,
    }


def multirater_summary(units: Sequence[Sequence[object]]) -> dict[str, float | int | None]:
    cleaned = [[value for value in map(clean, unit) if value is not None] for unit in units]
    valid = [unit for unit in cleaned if len(unit) >= 2]
    unanimous = sum(len(set(unit)) == 1 for unit in valid)
    return {
        "comparable_units": len(valid),
        "unanimous_units": unanimous,
        "unanimous_share": unanimous / len(valid) if valid else None,
        "krippendorff_alpha": krippendorff_alpha_nominal(valid),
    }


def normalized_entropy(values: Sequence[object], weights: Sequence[float] | None = None) -> float:
    cleaned = [clean(value) for value in values]
    if weights is None:
        counts = Counter(value for value in cleaned if value is not None)
        masses = np.array(list(counts.values()), dtype=float)
    else:
        weighted_counts: dict[str, float] = {}
        for value, weight in zip(cleaned, weights):
            if value is not None:
                weighted_counts[value] = weighted_counts.get(value, 0.0) + float(weight)
        masses = np.array(list(weighted_counts.values()), dtype=float)
    if len(masses) <= 1 or masses.sum() == 0:
        return 0.0
    probabilities = masses / masses.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(masses)))
